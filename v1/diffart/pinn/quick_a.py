"""Time-limited CPU ablation of a single sharp stepped axisymmetric tube.

Run with python -m diffart.pinn.quick_a. This is stage (a), not a parametric
surrogate. References are independent P1 FEM on a conforming sharp-step mesh.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import time

# Set these before importing numerical libraries, never acquire a GPU lease.
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_key] = '4'

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import torch
from torch import nn

from .uniform import C, plane_transfer

HERE = Path(__file__).resolve().parent
LENGTH = .17
PIECES = ((0., .08, .01), (.08, .17, .015))
Z = .15
SEED = 20261109
LABELS = ('基準', '+無次元化・出力尺度', '+入口 hard constraint',
          '+k 比例 Fourier', '+FEM 2% 教師')


def fem(frequency, spacing, uniform=False):
    """Assemble r-weighted weak Helmholtz, with natural rigid shoulders.

    Coincident nodes at the aperture are merged; the extra radial face on the
    wider piece remains a rigid wall. No smoothing/interpolation across the step.
    """
    pieces = ((0., LENGTH, .015),) if uniform else PIECES
    node_map, nodes, triangles, regions = {}, [], [], []
    for j, (start, end, radius) in enumerate(pieces):
        nx, nr = round((end-start)/spacing), round(radius/spacing)
        grid = np.empty((nx+1, nr+1), int)
        for i, x in enumerate(np.linspace(start, end, nx+1)):
            for h, r in enumerate(np.linspace(0, radius, nr+1)):
                key = (round(float(x), 12), round(float(r), 12))
                if key not in node_map:
                    node_map[key] = len(nodes)
                    nodes.append((x, r))
                    regions.append(j)
                grid[i, h] = node_map[key]
        a, b, c, d = (grid[:-1, :-1].ravel(), grid[1:, :-1].ravel(),
                      grid[1:, 1:].ravel(), grid[:-1, 1:].ravel())
        triangles.extend(np.stack((a, b, c), -1))
        triangles.extend(np.stack((a, c, d), -1))
    nodes, triangles = np.asarray(nodes), np.asarray(triangles)
    tri = nodes[triangles]
    delta = tri[:, 1:] - tri[:, :1]
    determinant = delta[:, 0, 0]*delta[:, 1, 1]-delta[:, 0, 1]*delta[:, 1, 0]
    area = abs(determinant)/2
    grad = np.stack((tri[:, [1, 2, 0], 1]-tri[:, [2, 0, 1], 1],
                     tri[:, [2, 0, 1], 0]-tri[:, [1, 2, 0], 0]), -1)/determinant[:, None, None]
    stiffness = area[:, None, None]*tri[:, :, 1].mean(-1)[:, None, None]*np.einsum('tia,tja->tij', grad, grad)
    bary = np.array([[1/3]*3, [.6, .2, .2], [.2, .6, .2], [.2, .2, .6]])
    quadrature = np.array([-27, 25, 25, 25])/48
    mass = area[:, None, None]*np.einsum('tq,q,qi,qj->tij', tri[:, :, 1]@bary.T, quadrature, bary, bary)
    k = 2*np.pi*frequency/C
    block = stiffness-k*k*mass
    rows = np.broadcast_to(triangles[:, :, None], block.shape).ravel()
    cols = np.broadcast_to(triangles[:, None, :], block.shape).ravel()
    values = block.astype(complex).ravel()
    rhs, weights = np.zeros(len(nodes), complex), np.zeros(len(nodes))
    np.add.at(weights, triangles.ravel(), mass.sum(-1).ravel())
    u, w = np.polynomial.legendre.leggauss(2)
    phi = np.stack(((1-u)/2, (1+u)/2), -1)
    for side in (0., LENGTH):
        indices = np.where(abs(nodes[:, 0]-side) < 1e-12)[0]
        indices = indices[np.argsort(nodes[indices, 1])]
        pairs = np.stack((indices[:-1], indices[1:]), -1)
        rad = nodes[pairs, 1]
        ds, rq = rad[:, 1]-rad[:, 0], rad@phi.T
        integrals = ds[:, None]/2*np.einsum('eq,q,qi->ei', rq, w, phi)
        if side == 0:
            np.add.at(rhs, pairs.ravel(), -integrals.ravel()/LENGTH)
        else:
            face = ds[:, None, None]/2*np.einsum('eq,q,qi,qj->eij', rq, w, phi, phi)
            rows = np.r_[rows, np.broadcast_to(pairs[:, :, None], face.shape).ravel()]
            cols = np.r_[cols, np.broadcast_to(pairs[:, None, :], face.shape).ravel()]
            values = np.r_[values, (1j*k/Z*face).ravel()]
            end_pairs, end_integrals = pairs, integrals
    matrix = sparse.coo_matrix((values, (rows, cols)), shape=(len(nodes), len(nodes))).tocsc()
    pressure = spsolve(matrix, rhs)
    transfer = -1j*k/Z*np.sum(end_integrals*pressure[end_pairs])*LENGTH/(pieces[0][2]**2/2)
    return dict(nodes=nodes, pressure=pressure, weights=weights, regions=np.asarray(regions),
                transfer=transfer, residual=float(np.linalg.norm(matrix@pressure-rhs)/np.linalg.norm(rhs)))


class Field(nn.Module):
    def __init__(self, frequency, arm, width=32):
        super().__init__()
        self.arm, self.k = arm, 2*np.pi*frequency/C
        self.scale = 1/(self.k*LENGTH) if arm >= 1 else 1.
        self.nets = nn.ModuleList([nn.Sequential(nn.Linear(8, width), nn.Tanh(),
            nn.Linear(width, width), nn.Tanh(), nn.Linear(width, width), nn.Tanh(),
            nn.Linear(width, 2)) for _ in PIECES])

    def forward(self, j, t, s):
        start, end, _ = PIECES[j]
        # t^2 parameterization imposes p_x(0,r)=1/L exactly in the inlet piece,
        # retaining a transverse-dependent inlet pressure and an even radial field.
        u = t.square() if self.arm >= 2 and j == 0 else t
        scales = t.new_tensor([.5, 1., 2.])*self.k*(end-start) if self.arm >= 3 else t.new_tensor([1., 2., 4.])*np.pi
        phase = u*scales
        p = self.scale*self.nets[j](torch.cat((u, s, phase.sin(), phase.cos()), -1))
        if self.arm >= 2 and j == 0:
            p = p+torch.cat((t*(end-start)/LENGTH, torch.zeros_like(t)), -1)
        return p


def derivative(p, x):
    return torch.cat([torch.autograd.grad(v.sum(), x, create_graph=True, retain_graph=True)[0]
                      for v in p.split(1, -1)], -1)


def coordinates(n, generator, t=None, s=None):
    a = torch.rand(n, 1, generator=generator) if t is None else torch.full((n, 1), t)
    b = torch.rand(n, 1, generator=generator) if s is None else torch.full((n, 1), s)
    return a.requires_grad_(), b.requires_grad_()


def physics(model, generator, count=64):
    terms = {}
    k, scale = model.k, model.scale
    # arm 0 reproduces source-pressure scaling; arm 1 uses q=p/(1/kL),
    # and dimensionless pressure/velocity residuals. Identical for later arms.
    pscale = scale if model.arm >= 1 else 1.
    for j, (start, end, radius) in enumerate(PIECES):
        length = end-start
        t, s = coordinates(count, generator)
        p = model(j, t, s)
        pt, ps = derivative(p, t), derivative(p, s)
        lap = derivative(pt, t)/length**2+4/radius**2*(s*derivative(ps, s)+ps)
        terms[f'pde{j}'] = ((lap+k*k*p)*LENGTH**2/(1+(k*LENGTH)**2)/pscale).square().mean()
        t, s = coordinates(24, generator, s=1.)
        p = model(j, t, s)
        terms[f'wall{j}'] = (2*derivative(p, s)/pscale).square().mean()
    t, s = coordinates(24, generator, t=0.)
    p = model(0, t, s)
    target = torch.tensor([1., 0.])
    terms['inlet'] = (derivative(p, t)*LENGTH/(PIECES[0][1]-PIECES[0][0])-target).square().mean()
    t, s = coordinates(24, generator, t=1.)
    p = model(1, t, s)
    px = derivative(p, t)/(PIECES[1][1]-PIECES[1][0])
    # z/(i k)*p_x, represented by two real channels.
    robin = p+Z/k*torch.stack((px[:, 1], -px[:, 0]), -1)
    terms['outlet'] = (robin/pscale).square().mean()
    t, s = coordinates(32, generator, t=1.)
    t2 = torch.zeros_like(t).requires_grad_()
    p0, p1 = model(0, t, s), model(1, t2, s*(.01/.015)**2)
    terms['interface_p'] = ((p0-p1)/pscale).square().mean()
    terms['interface_dx'] = (LENGTH*(derivative(p0, t)/.08-derivative(p1, t2)/.09)).square().mean()
    t, s = coordinates(24, generator, t=0.)
    # Annular shoulder: r in (.01,.015); s=r^2/R^2.
    s = s*(1-(.01/.015)**2)+(.01/.015)**2
    p = model(1, t, s)
    terms['shoulder'] = (LENGTH/.09*derivative(p, t)).square().mean()
    return terms


def predict(model, ref, indices):
    result = torch.empty(len(indices), 2)
    for j, (start, end, radius) in enumerate(PIECES):
        mask = ref['regions'][indices] == j
        xy = torch.from_numpy(ref['nodes'][indices[mask]])
        result[torch.from_numpy(mask)] = model(j, (xy[:, :1]-start)/(end-start), (xy[:, 1:]/radius)**2)
    return result


def evaluate(model, ref, indices):
    with torch.no_grad():
        p = predict(model, ref, indices).numpy()
    p = p[:, 0]+1j*p[:, 1]
    truth, weights = ref['pressure'][indices], ref['weights'][indices]
    return float(np.sqrt(np.sum(weights*abs(p-truth)**2)/np.sum(weights*abs(truth)**2)))


class TrialTimeout(Exception):
    pass


def train(frequency, arm, ref, teachers, evaluation, args, deadline):
    torch.manual_seed(SEED)
    model = Field(frequency, arm).double()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    generator = torch.Generator().manual_seed(SEED+1)
    out = args.out/f'{int(frequency)}hz_{arm}'
    out.mkdir(parents=True, exist_ok=True)
    (out/'training.jsonl').write_text('')
    start = time.monotonic()
    history, reason, step = [], 'step_limit', 0
    teacher_truth = np.stack((ref['pressure'][teachers].real, ref['pressure'][teachers].imag), -1)
    teacher_truth = torch.from_numpy(teacher_truth)
    teacher_weights = torch.from_numpy(ref['weights'][teachers])
    teacher_weights = teacher_weights/teacher_weights.sum()

    def record():
        # A fixed independent collocation sample makes residual trends comparable.
        heldout = torch.Generator().manual_seed(SEED+99)
        parts = {key: float(v.detach()) for key, v in physics(model, heldout, 128).items()}
        # Common units across arms: undo pressure normalization for reporting.
        common = {key: val*(model.scale**2 if arm >= 1 and
                  (key.startswith('pde') or key.startswith('wall') or key in ('outlet', 'interface_p')) else 1.)
                  for key, val in parts.items()}
        row = dict(step=step, elapsed_seconds=time.monotonic()-start,
                   residual_mse=sum(parts.values()), common_residual_mse=sum(common.values()),
                   residual_parts=parts, common_residual_parts=common,
                   fem_holdout_relative_l2=evaluate(model, ref, evaluation))
        history.append(row)
        with (out/'training.jsonl').open('a') as f:
            f.write(json.dumps(row)+'\n')
        print(json.dumps(dict(frequency_hz=frequency, arm=arm, **row)), flush=True)

    def alarm_handler(signum, frame):
        raise TrialTimeout()

    old = signal.signal(signal.SIGALRM, alarm_handler)
    # Leave >10 seconds for checkpoint and final metrics before the 5-min ceiling.
    signal.setitimer(signal.ITIMER_REAL, max(.1, min(args.trial_seconds-12, deadline-time.monotonic()-15)))
    try:
        record()
        for step in range(1, args.steps+1):
            terms = physics(model, generator)
            loss = sum(terms.values())
            if arm == 4:
                teacher_prediction = predict(model, ref, teachers)
                data = (teacher_weights[:, None]*((teacher_prediction-teacher_truth)/model.scale)**2).sum()/2
                loss = loss+10*data
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 100.)
            optimizer.step()
            if step % 100 == 0:
                record()
                if len(history) >= 5:
                    earlier, latest = history[-4], history[-1]
                    if (latest['common_residual_mse'] < .95*earlier['common_residual_mse'] and
                        latest['fem_holdout_relative_l2'] >= .995*earlier['fem_holdout_relative_l2']):
                        reason = 'residual_down_l2_stalled_300_steps'
                        break
            if time.monotonic() > min(start+args.trial_seconds-15, deadline-20):
                reason = 'time_limit'
                break
    except TrialTimeout:
        reason = 'time_limit'
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)
    if history[-1]['step'] != step:
        record()
    result = dict(frequency_hz=frequency, arm=arm, label=LABELS[arm], steps=step,
        elapsed_seconds=time.monotonic()-start, reason=reason, initial=history[0], final=history[-1],
        best_l2=min(r['fem_holdout_relative_l2'] for r in history),
        pressure_scale=model.scale, teacher_count=len(teachers) if arm == 4 else 0)
    torch.save(dict(model=model.state_dict(), frequency_hz=frequency, arm=arm), out/'field.pt')
    (out/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def write_report(summary, out):
    rows = summary['trials']
    lines = ['# C: 段階 (a) CPU 短時間試行', '',
        '2026-11-09。単一の軸対称段差管、周波数ごとにゼロから学習。工夫を一つずつ累積追加した比較（単独効果ではなく直前条件との差）。', '',
        '| Hz | 条件 | step | 秒 | 共通残差 MSE | FEM 相対 L2（初期→終了） | 直前条件からの L2 差 |',
        '| ---: | --- | ---: | ---: | ---: | ---: | ---: |']
    prev = {}
    for r in rows:
        f, final = r['frequency_hz'], r['final']
        delta = final['fem_holdout_relative_l2']-prev[f] if f in prev else None
        lines.append(f"| {f:g} | {r['label']} | {r['steps']} | {r['elapsed_seconds']:.1f} | {final['common_residual_mse']:.3e} | {r['initial']['fem_holdout_relative_l2']:.4f} → {final['fem_holdout_relative_l2']:.4f} | {delta:+.4f} |" if delta is not None else
            f"| {f:g} | {r['label']} | {r['steps']} | {r['elapsed_seconds']:.1f} | {final['common_residual_mse']:.3e} | {r['initial']['fem_holdout_relative_l2']:.4f} → {final['fem_holdout_relative_l2']:.4f} | — |")
        prev[f] = final['fem_holdout_relative_l2']
    lines += ['', '## 条件・読み方', '',
        '- 管長 170 mm、段差位置 80 mm、半径 10→15 mm。c=350 m/s、exp(+iωt)、入口 p_x=1/L、剛壁、出口 p+z/(ik)p_x=0、z=0.15（実定数）。追加壁損失なし。',
        '- 独立 P1 三角形 FEM は段差適合メッシュ。共通開口の節点を共有し、肩は自然 Neumann 条件。積分は軸対称の r 重み。',
        '- 2 つの区間に別 MLP（各 3 層、幅 32、tanh、複素圧力の実虚 2 出力）。全条件 float64、Adam lr=0.001、seed=20261109。各試行を同じ seed で初期化し直す。',
        '- 基準も既存コード同様の局所座標 t と s=(r/R)^2、dyadic Fourier を使用。工夫 (1) は圧力 p*=1/(kL) と q=p/p*、圧力型残差を p* で正規化。速度型残差は入口尺度を保持する。',
        '- 工夫 (2) は入口区間 p=t l/L+p* N(t²,s) とし、p_x(0,r)=1/L を構造で満たす。工夫 (3) は dyadic π[1,2,4] を k l [0.5,1,2] に置換（hard 制約内では t² に適用）。',
        '- 工夫 (4) は固定 seed の FEM 節点 2%（一様無作為、重複なし）を教師とし、r 重み付き q の MSE を重み 10 で追加。教師点は全条件で L2 評価から除外。FEM 値を最適化損失に使うのは最後の条件だけで、全条件の評価と早期停止には FEM L2 を使う。',
        '- L2=√[Σw|p−p_FEM|²/Σw|p_FEM|²]。残り 98% 全節点で評価。共通残差は基準と同じ単位・重みへ戻した固定独立点 MSE の和（教師損失を含まない）。正規化後の各残差も JSONL に保存。',
        '- 100 step ごとと初期・終了に記録。300 step 間で残差が 5% 以上改善し、L2 が 0.5% も改善しない場合は打ち切り。step 上限と時間上限も適用。最良 L2 での選別はせず、表は終了時の値。',
        f"- CPU のみ、数値演算 threads≤4、PyTorch interop=1。各試行上限 {summary['config']['trial_seconds']:g} 秒、全作業 30 分以内。実験経過 {summary['elapsed_seconds']:.1f} 秒。GPU・ポッド・B は使用・変更しない。", '',
        '## FEM 検証', '']
    for f, r in summary['reference_checks'].items():
        lines.append(f"- {f} Hz: FEM 細格子 {r['fine_nodes']} 点、粗→細の場 L2 差 {r['field_grid_change']:.3e}、伝達関数差 {r['transfer_grid_change']:.3e}、線形方程式残差 {r['algebraic_residual']:.3e}。")
    lines += ['', '## 判定', '']
    for f in sorted(prev):
        group = [r for r in rows if r['frequency_hz'] == f]
        best = min(group, key=lambda r:r['final']['fem_holdout_relative_l2'])
        value = best['final']['fem_holdout_relative_l2']
        lines.append(f"- {f:g} Hz の最小終了時 L2 は **{value:.4f}**（{best['label']}）。"+
                     ('5% 以下。' if value <= .05 else '5% に未到達。'))
        for r in group:
            lines.append(f"  - {r['label']}: 停止理由 `{r['reason']}`。")
    lines += ['- 単一 seed・短時間・累積比較のため一般化や各工夫の独立効果は未検証。段階 (b) 以降と B 用の教師作成は実施しない。', '',
        '再実行: `.venv/bin/python -m diffart.pinn.quick_a --steps 2000 --trial-seconds 110 --budget-seconds 1400`。',
        '生ログ: `<周波数>hz_<条件番号>/training.jsonl`、全設定・FEM 確認・結果: `summary.json`。', '']
    (out/'RESULT.md').write_text('\n'.join(lines))
    (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')


def main(args):
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float64)
    args.out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    deadline = start+args.budget_seconds
    summary = dict(started_utc=datetime.now(timezone.utc).isoformat(),
        config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        reference_checks={}, trials=[], elapsed_seconds=0.,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    for frequency in (1000., 4000.):
        coarse, ref = fem(frequency, .001), fem(frequency, .0005)
        # Coarse nodes are a subset of the fine grid; compare on those nodes.
        mapping = {tuple(np.round(xy, 12)): i for i, xy in enumerate(ref['nodes'])}
        ix = np.array([mapping[tuple(np.round(xy, 12))] for xy in coarse['nodes']])
        grid_change = np.sqrt(np.sum(coarse['weights']*abs(coarse['pressure']-ref['pressure'][ix])**2)/
                              np.sum(coarse['weights']*abs(ref['pressure'][ix])**2))
        summary['reference_checks'][str(int(frequency))] = dict(
            coarse_nodes=len(coarse['nodes']), fine_nodes=len(ref['nodes']),
            field_grid_change=float(grid_change), transfer_grid_change=float(abs(coarse['transfer']/ref['transfer']-1)),
            algebraic_residual=ref['residual'])
        rng = np.random.default_rng(SEED+2)
        teachers = np.sort(rng.choice(len(ref['nodes']), round(.02*len(ref['nodes'])), replace=False))
        evaluation = np.setdiff1d(np.arange(len(ref['nodes'])), teachers)
        np.savez_compressed(args.out/f'fem_{int(frequency)}.npz', **ref, teachers=teachers, evaluation=evaluation)
        for arm in range(5):
            if time.monotonic() > deadline-25:
                break
            result = train(frequency, arm, ref, teachers, evaluation, args, deadline)
            summary['trials'].append(result)
            summary['elapsed_seconds'] = time.monotonic()-start
            write_report(summary, args.out)
    summary['elapsed_seconds'] = time.monotonic()-start
    write_report(summary, args.out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=HERE/'results/quick_a')
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--trial-seconds', type=float, default=110.)
    parser.add_argument('--budget-seconds', type=float, default=1400.)
    args = parser.parse_args()
    if not 15 < args.trial_seconds <= 300 or not 0 < args.budget_seconds <= 1600:
        parser.error('trial seconds in (15,300], budget seconds in (0,1600]')
    main(args)
