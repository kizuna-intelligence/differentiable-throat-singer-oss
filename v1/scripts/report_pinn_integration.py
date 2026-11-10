"""Collect isolated C measurements, plots and a concise reproducible report."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=Path('artifacts/pinn_integration'))
    args=ap.parse_args();out=args.out
    data=json.loads((out/'full/summary.json').read_text())
    surrogate=json.loads((out/'full/surrogate/summary.json').read_text())
    losses=json.loads((out/'full/integration/losses.json').read_text())
    baseline=json.loads(Path('artifacts/evaluation/results.json').read_text())
    summary=[]
    for kind,shift in [('reproduction',0),('score',0),('score',12)]:
        rows=[r for r in baseline if r['model']=='physics' and r['condition']==kind and r['semitones']==shift]
        summary.append(dict(variant='baseline',condition=kind,semitones=shift,
            wavlm_cosine=float(np.mean([r['wavlm_cosine'] for r in rows])),
            ecapa_cosine=float(np.mean([r['ecapa_cosine'] for r in rows]))))
    for variant in ('correction_only','one_epoch'):
        p=out/f'full/integration/{variant}/summary.json'
        if p.exists():summary.extend(dict(r,variant=variant) for r in json.loads(p.read_text()))
    chosen=None
    # Song choice uses mean WavLM over all three requested conditions; ECAPA and
    # the independent short validation loss remain visible, no per-song cherry-pick.
    scores={v:np.mean([r['wavlm_cosine'] for r in summary if r['variant']==v])
            for v in ('correction_only','one_epoch') if any(r['variant']==v for r in summary)}
    if len(scores)==2 and 'one_epoch_loss' in losses:chosen=max(scores,key=scores.get)
    before=json.loads((out/'protected_before.json').read_text())
    integrity={p:Path(p).exists() and hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in before.items()}
    save=dict(data=data,surrogate=surrogate,losses=losses,voice_comparison=summary,
              selected_variant=chosen,selection='mean WavLM over three requested conditions, four held-out songs each',
              protected_files=len(integrity),all_protected_unchanged=all(integrity.values()))
    memory=out/'full/integration/gpu_memory_summary.json'
    if memory.exists():save['gpu_memory']=json.loads(memory.read_text())
    song=out/'song.json'
    if song.exists():save['song']=json.loads(song.read_text())
    (out/'report.json').write_text(json.dumps(save,ensure_ascii=False,indent=2)+'\n')
    (out/'integrity.json').write_text(json.dumps(integrity,ensure_ascii=False,indent=2)+'\n')
    lines=['# C: FEM比の代理モデルを喉モデルへ組み込み','',
        f"学習形状 {data['train_shapes']}、独立未学習 {data['heldout_shapes']}、50–8000 Hz 等間隔80点。",
        'FEM教師のみのMLP。場のネットワークとPDE残差は使っていない。',
        f"R絶対値 min/5%/median/95%/99%/max: {data['ratio_abs_quantiles']}",
        f"代理モデル MAE {surrogate['heldout_magnitude_mae_db']:.4f} dB / {surrogate['heldout_phase_mae_rad']:.4f} rad。",
        f"代理モデル最大 |R|={surrogate['predicted_abs_max']:.4f}。補正単体の非増幅条件は未達、因果性・受動性の証明なし。",
        f"一様管低域 max |log R|={data['uniform_low_log_max']:.6g}。",
        f"検証損失: 元 {losses['baseline_loss']:.6f} / 補正のみ {losses['correction_only_loss']:.6f} / 1巡 {format(losses['one_epoch_loss'],'.6f') if 'one_epoch_loss' in losses else '計算中'}",
        f"1巡は {losses.get('epoch_samples','未完')} 完全1秒区間、新部品凍結、話者・調音部学習。lr={losses.get('learning_rate','未完')}。",
        '', '|版|条件|半音|WavLM|ECAPA|','|---|---|---:|---:|---:|']
    for r in summary:
        lines.append(f"|{r['variant']}|{r['condition']}|{r['semitones']}|{r['wavlm_cosine']:.6f}|{r['ecapa_cosine']:.6f}|")
    lines+=['',f"歌に選ぶ版: {chosen if chosen else '全評価後に決定'}（3条件のWavLM平均）。",
        f"保護対象{len(integrity)}ファイルのSHA256一致: {all(integrity.values())}。",'',
        '## FEMの粗密メッシュ比較（10形）','',
        '|形|振幅MAE dB|最大差 dB|位相MAE rad|H相対L2|','|---:|---:|---:|---:|---:|']
    for r in data['mesh_checks']:
        lines.append(f"|{r['shape']}|{r['magnitude_mae_db']:.5f}|{r['magnitude_max_db']:.5f}|{r['phase_mae_rad']:.5f}|{r['relative_l2']:.5f}|")
    lines+=['','境界条件・鼻腔への近似・教師の幾何条件変化は DESIGN_ISSUES_pinn.md を参照。',
        '声の近さは既存と同じ3参照音声、非無音の最大20秒窓、窓平均→曲平均。元モデルの声の数値は既存評価を読み取り使用。',
        '短い損失検証は固定seedで各検証曲から連続30秒を選び、完全1秒区間の非無音を等重みに集計。',
        '歌のsynth.pyは凍結補正を含むチェックポイントを自動認識。B1/B2は使用せず元LFを使用。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    d=np.load(out/'full/dataset.npz');f=d['freq'];db=20*np.log10(abs(d['ratio']))
    fig,axes=plt.subplots(2,1,figsize=(9,7),sharex=True)
    lo,median,hi=np.quantile(db,[.05,.5,.95],axis=0)
    axes[0].fill_between(f,lo,hi,alpha=.25,label='5–95% shapes');axes[0].plot(f,median,label='median')
    axes[0].axhline(0,color='k',lw=.7);axes[0].set_ylabel('20 log10 |H3D/H1D| (dB)');axes[0].legend()
    phase=np.unwrap(np.angle(d['ratio']),axis=1);lo,median,hi=np.quantile(phase,[.05,.5,.95],axis=0)
    axes[1].fill_between(f,lo,hi,alpha=.25);axes[1].plot(f,median)
    axes[1].set_ylabel('phase (rad)');axes[1].set_xlabel('Frequency (Hz)')
    fig.tight_layout();fig.savefig(out/'ratio_distribution.png',dpi=150);plt.close(fig)
    print(json.dumps(dict(selected_variant=chosen,losses=losses,voice_comparison=summary,
        surrogate=surrogate,protected_unchanged=all(integrity.values())),ensure_ascii=False))


if __name__=='__main__':main()
