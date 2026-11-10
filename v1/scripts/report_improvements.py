"""B spectral diagnosis, learned parameters and comparison tables (isolated)."""
import json
import argparse
import csv
import numpy as np
import soundfile as sf
import torch
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from diffart.data import ROOT, SR, HOP, sha
from diffart.runtime import save_json
from diffart.dataset import Corpus
from diffart.improved import ImprovedSynthesizer
from diffart.physics import transfer, radiation, C


def read(path): return json.loads(path.read_text())
def mean(rows,k): return float(np.mean([r[k] for r in rows if r.get(k) is not None]))
def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])+'\n'
def summary_rows(rows):
    groups=[('reproduction',0,None)]+[('score',s,None) for s in [-12,-5,0,5,7,12]]+[('rd',0,x) for x in [.5,1.,1.5,2.,2.5]]
    out=[]
    for kind,shift,rd in groups:
        rs=[r for r in rows if r['model']!='recording' and r['condition']==kind and r['semitones']==shift and r['rd']==rd]
        out.append(dict(condition=kind,semitones=shift,rd=rd,songs=len(rs),**{k:mean(rs,k) for k in ['wavlm_cosine','ecapa_cosine','pitch_median_abs_cents','pitch_voiced_recall','pitch_within50_given_voiced','rmvpe_range_coverage']},**({k:mean(rs,k) for k in ['kana_cer','mel_cepstral_distortion']} if kind=='reproduction' else {})))
    return out


def diagnose(variant):
    # Reuse A1 exactly: 2048 FFT/240 hop, reference activity, floor, RMS gain,
    # voiced/unvoiced masks, equal song aggregation. This module is CPU only.
    from diagnose_bands import spectrum_diagnosis, write_csv
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=ROOT/'artifacts/improvements'/variant
    dest=out/'diagnosis';dest.mkdir(parents=True,exist_ok=True)
    c=Corpus();bands=[]
    for song in [r for r in c.manifest['songs'] if r['split']=='test']:
        d=c.load(song);y,sr=sf.read(out/'audio'/song['id']/'reproduction_+0.wav',dtype='float32');assert sr==SR
        rows,curves,active,gain=spectrum_diagnosis(dict(song,pause_id=c.manifest['vocab']['pau']),d,y,dest)
        bands.extend(rows)
        f=np.fft.rfftfreq(2048,1/SR)
        dips=[]
        for voicing in curves:
            real,synth=curves[voicing]['rms_matched']
            smoothed=gaussian_filter1d(synth,2)
            ix,_=find_peaks(-smoothed)
            ix=ix[(f[ix]>=4000)&(f[ix]<=6000)]
            dips.append(dict(song=song['id'],voicing=voicing,ltas_minima_4_6khz_hz=f[ix].tolist()))
        save_json(dest/f"{song['id']}__notches.json",dips)
    write_csv(dest/'bands.csv',bands)
    aggregate=[]
    for key in sorted({(r['voicing'],r['normalization'],r['band']) for r in bands}):
        rs=[r for r in bands if (r['voicing'],r['normalization'],r['band'])==key]
        aggregate.append(dict(zip(['voicing','normalization','band'],key),songs=len(rs),**{k:mean(rs,k) for k in ['lsd_db','ltas_bias_db','ltas_distance_db']}))
    save_json(dest/'bands_summary.json',aggregate)
    state=torch.load(out/'best.pt',map_location='cpu',weights_only=True)['model']
    m=ImprovedSynthesizer(len(c.manifest['vocab']),b1=variant in ('b1','both'),b2=variant in ('b2','both'));m.load_state_dict(state);m.eval()
    params={}
    with torch.no_grad():
        if m.b1:
            p=m.piriform
            params['piriform']=dict(length_cm=(p.length*100).tolist(),area_cm2=p.area.tolist(),position_cm=(p.position*100).tolist(),coupling=p.coupling.tolist(),quarter_wave_hz=(C/(4*p.length)).tolist(),attachment='nonnegative Gaussian distribution over section junctions, width 0.65 sections')
        if m.b2:
            g=m.glottis
            params['glottis']=dict(shape=g.shape.tolist(),shape_rd=g.shape_rd.tolist(),open_quotient=float(1-g.closed),jitter_std=float(g.jitter),shimmer_std=float(g.shimmer),pulsation=float(g.pulsation),aspiration_min_pre_gate=float(.05*g.noise_warmup),aspiration_min_post_gate=float(.05*g.noise_warmup*(1-.85*g.pulsation)))
            phase=torch.linspace(0,1,1001)
            fig,ax=plt.subplots()
            for rd in [.5,1.,1.5,2.,2.5]:
                q=g.flow(phase,torch.full_like(phase,rd));ax.plot(phase,q,label=f'Rd={rd}')
            ax.set(xlabel='Cycle phase',ylabel='Nonnegative volume flow');ax.legend();fig.tight_layout();fig.savefig(dest/'source.png',dpi=140);plt.close(fig)
        if m.b1:
            anatomy=read(ROOT/'artifacts/anatomy/parameters.json')
            m.freq=torch.linspace(0,12000,4097)
            fig,ax=plt.subplots();rs=[]
            for vowel,a in anatomy['vowel_means'].items():
                a=torch.tensor(a)[None];v=torch.zeros(1)
                original=transfer(a,m.speaker.length,m.speaker.wall,m.freq,m.speaker.nose,v)
                improved=m.transfer(a,v)
                db=20*torch.log10((improved.abs()/original.abs()).clamp_min(1e-12))[0]
                ix=(m.freq>=4000)&(m.freq<=6000)
                notch=float(m.freq[ix][db[ix].argmin()])
                rs.append(dict(vowel=vowel,branch_ratio_notch_4_6khz_hz=notch,depth_db=float(db[ix].min()),velum=0.))
                ax.plot(m.freq,db,label=vowel)
            params['branch_notches']=rs
            ax.set(xlim=(3000,7000),xlabel='Frequency (Hz)',ylabel='Added branch transfer / original (dB)');ax.legend();fig.tight_layout();fig.savefig(dest/'branch_transfer.png',dpi=140);plt.close(fig)
    save_json(dest/'learned_parameters.json',params)
    save_json(dest/'provenance.json',dict(checkpoint_sha256=sha(out/'best.pt'),diagnose_bands_sha256=sha(ROOT/'scripts/diagnose_bands.py'),method='A1 identical full-duration analysis, no oracle intervention'))


def report():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dest=ROOT/'artifacts/improvements'
    old=summary_rows(read(ROOT/'artifacts/evaluation/results.json'))
    models={'元モデル':old};train={};bands={};params={}
    for variant,label in [('b1','B1'),('b2','B2'),('both','B1+B2')]:
        out=dest/variant
        train[label]=read(out/'result.json')
        models[label]=summary_rows(read(out/'evaluation/results.json'))
        bands[label]=read(out/'diagnosis/bands_summary.json')
        params[label]=read(out/'diagnosis/learned_parameters.json')
    bands['元モデル']=read(ROOT/'artifacts/diagnosis/summary.json')['bands']
    save_json(dest/'summary.json',dict(evaluation=models,training=train,bands=bands,parameters=params))
    rows=[]
    for i,r in enumerate(old):
        label='再現' if r['condition']=='reproduction' else f"楽譜 {r['semitones']:+d}" if r['condition']=='score' else f"Rd {r['rd']:.1f}"
        rows.append([label]+[f"{values[i]['wavlm_cosine']:.4f} / {values[i]['ecapa_cosine']:.4f}" for values in models.values()])
    text='## B. 梨状窩・学習できる声門音源（3条件の学習・評価完了）\n\n'
    text+='`diffart/improved.py` の別クラス `ImprovedSynthesizer` で実装。元の `physics.py`・`artifacts/main/best.pt`・既存の評価音声は変更していない。B1のみ・B2のみ・両方・両方無効の全設定で、元の best.pt と初期出力がビット単位で一致する単体テストが通過。受動分岐、波形制約、勾配、F0の声道形状からの分離も検査した。`diffart/pinn/` への書き込みなし。\n\n'
    text+='B1 は左右2本の損失付き閉管（長さ1.5〜2.5 cm、面積0.2〜2 cm²）を非負のシャントアドミタンスとして接続。結合位置2〜3 cmを区間の非負連続重みで表す分布近似で、各シャントは受動。B2はLF包絡で窓掛けした8個のBernstein基底に、話者固定＋Rd依存の正の重みを学習する。開放部のみ正で閉鎖部はゼロ。閉鎖長、脈動量、ジッター・シマー標準偏差も話者固定パラメーター。初期一致の後、最初の1巡で息雑音の下限を0から0.05へ上げる（ゲート前、元の0.03係数を掛ける前）。\n\n'
    text+='今回は追加部品の効果を比較するため既存話者・調音部を固定し、新部品だけを学習した。元の best.pt を共通の出発点とし、同じ抽選1時間の3,594個の完全な1秒区間、同じ順序・seed、Adam lr=0.002、batch=4を使用。検証は既存と同じ4検証曲の有音1秒区間全体。改善0.002未満が3巡続いたら停止し、最良検証重みを選択。テスト曲は重み選択に使っていない。3090は他ジョブ使用中・共有UUIDロック保持のためCPUを使用。RTX PRO 4000・2番PC・17番機は使っていない。資源確認は `artifacts/improvements/before.json`。\n\n'
    text+=table(['条件','初期検証損失','最良検証損失','選択した巡','終了した巡','停止理由'],[[label,f"{t['initial']:.6f}",f"{t['best']:.6f}",f"{t['best_step']/t['epoch_blocks']:.1f}",f"{t['completed_epochs']:.1f}",t['stopping_reason']] for label,t in train.items()])
    text+='\n評価は既存 `evaluate.py` の `Assessment.assess` と `render` をそのまま使用。test 4曲全長、同じ参照3本・20秒区間・RMS活動判定・F0・6移調・5 Rd条件。各曲内の非無音20秒区間の等重み平均を計算し、表は4曲等重み。数値は **WavLM / ECAPA**。前回の丸め値（再現0.693・楽譜±0 0.667・+12 0.567）に対応する保存済み評価値を左列に再掲。\n\n'
    text+=table(['条件',*models.keys()],rows)
    text+='\n'+table(['再現の指標',*models.keys()],[[k]+[f"{v[0][k]:.4f}" for v in models.values()] for k in ['kana_cer','mel_cepstral_distortion','pitch_median_abs_cents','pitch_voiced_recall','pitch_within50_given_voiced']])
    text+='\n帯域診断はA1の同じ関数を全4曲に再適用。曲別RMS一致・同じ有声／無声マスク・FFT2048／hop240・同じ床値。LTAS差は合成−録音（dB）。\n\n'
    def bandrow(model,v,b):return next(r for r in bands[model] if r['voicing']==v and r['normalization']=='rms_matched' and r['band']==b)
    text+=table(['区間・帯域','指標',*models.keys()],[[f'{v} {b}',k]+[f'{bandrow(model,v,b)[key]:+.2f}' for model in models] for v,b in [('voiced','4-6k'),('unvoiced','4-6k'),('unvoiced','6-8k'),('unvoiced','8-12k')] for k,key in [('LTAS差','ltas_bias_db'),('LSD','lsd_db')]])
    for label,p in params.items():
        if 'piriform' in p:
            a=p['piriform'];text+=f"\n{label} 梨状窩（左 / 右）: 長さ "+' / '.join(f'{x:.4f}' for x in a['length_cm'])+' cm、面積 '+ ' / '.join(f'{x:.4f}' for x in a['area_cm2'])+' cm²、位置 '+ ' / '.join(f'{x:.4f}' for x in a['position_cm'])+' cm、結合 '+ ' / '.join(f'{x:.4f}' for x in a['coupling'])+'。四分の一波長共鳴 '+ ' / '.join(f'{x:.1f}' for x in a['quarter_wave_hz'])+' Hz。母音ごとの枝追加による4〜6 kHzの伝達比の谷は '+ '、'.join(f"{r['vowel']} {r['branch_ratio_notch_4_6khz_hz']:.1f} Hz ({r['depth_db']:+.2f} dB)" for r in p['branch_notches'])+'。\n'
        if 'glottis' in p:
            a=p['glottis'];text+=f"\n{label} 音源: 開放率 {a['open_quotient']:.4f}、ジッター標準偏差 {100*a['jitter_std']:.4f}%、シマー {100*a['shimmer_std']:.4f}%、脈動量 {a['pulsation']:.4f}、息雑音のゲート後下限 {a['aspiration_min_post_gate']:.5f}（元の0.03係数前）。\n"
    deltas={label:models[label][0]['wavlm_cosine']-old[0]['wavlm_cosine'] for label in train}
    text+='\n再現WavLMの変化: '+ '、'.join(f'{label} {value:+.4f}' for label,value in deltas.items())+'。これは追加部品のみを学習した1 seedの比較で、全体再学習の効果や聴感の改善を保証するものではない。\n\n'
    text+='出力: [比較集計](artifacts/improvements/summary.json)、各条件の `result.json`・`training.jsonl`・`evaluation/results.json`・`evaluation/slopes.json`・`diagnosis/bands.csv`・`diagnosis/learned_parameters.json`。全曲音声は各条件の `audio/`。合成LTASの4〜6 kHzの局所的な谷は `diagnosis/*__notches.json` に保存（声道共鳴・調波・音素分布も含み、枝の共鳴と同一視しない）。\n\n'
    text+='再実行: `PYTHONPATH=. CUDA_VISIBLE_DEVICES=\'\' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/train_improvements.py b1 --device cpu --threads 2 --batch-size 4`（b1をb2/bothへ変更）。完了後に `scripts/evaluate_improvements.py b1 --device cpu`、`scripts/report_improvements.py --diagnose b1`、最後に `scripts/report_improvements.py`。\n'
    # Read the shared status at the last moment and only replace our B section.
    path=ROOT/'STATUS_pinn.md';current=path.read_text();marker='## B. 梨状窩・学習できる声門音源'
    if marker in current:
        begin=current.index(marker);end=current.find('\n## ',begin+1)
        current=current[:begin]+text+(current[end:] if end>=0 else '')
    else: current=current.rstrip()+'\n\n'+text
    path.write_text(current)
    (dest/'REPORT.md').write_text(text)
    fig,ax=plt.subplots()
    for label,t in train.items():
        history=[json.loads(line) for line in (dest/('both' if label=='B1+B2' else label.lower())/'training.jsonl').read_text().splitlines() if 'val_loss' in line]
        ax.plot([r['step']/t['epoch_blocks'] for r in history],[r['val_loss'] for r in history],marker='o',label=label)
    ax.set(xlabel='Passes through sampled hour',ylabel='Validation spectral loss');ax.legend();fig.tight_layout();fig.savefig(dest/'validation.png',dpi=140)
    before=read(dest/'before.json');after={k:sha(ROOT/p) for k,p in [('best_sha256','artifacts/main/best.pt'),('physics_sha256','diffart/physics.py'),('manifest_sha256','data/manifest.json')]}
    assert all(before[k]==after[k] for k in after)
    save_json(dest/'preservation.json',dict(before=before,after=after,unchanged=True))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--diagnose',choices=['b1','b2','both']);args=ap.parse_args()
    torch.set_num_threads(2)
    if args.diagnose: diagnose(args.diagnose)
    else: report()
if __name__=='__main__':main()
