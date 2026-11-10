"""Summarize measured results and write indexed listening reels."""
import json
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from math import gcd
from diffart.data import ROOT,SR,HOP,sha
from diffart.runtime import save_json

def read(path):return json.loads((ROOT/path).read_text())
def mean(rows,key):
    vals=[r[key] for r in rows if r.get(key) is not None]
    return float(np.mean(vals)) if vals else None

def main():
    train=read('artifacts/main/result.json');results=read('artifacts/evaluation/results.json');original=read('artifacts/comparison/results.json')
    assert len(results)==52 and len(original)==12
    assert len({r['song'] for r in results})==4
    cksha=sha(ROOT/'artifacts/main/best.pt')
    assert all(r['checkpoint_sha256']==cksha for r in results)
    assert all(r['physics_checkpoint_sha256']==cksha for r in original)
    summary={'training':train,'db_reproduction':{},'db_extrapolation':[],'db_rd':[],'original_comparison':[],
        'db_slopes':read('artifacts/evaluation/slopes.json'),'original_slopes':read('artifacts/comparison/slopes.json'),
        'noise':read('artifacts/evaluation/noise_paths.json'),'anatomy':read('artifacts/anatomy/parameters.json')}
    keys=['wavlm_cosine','ecapa_cosine','kana_cer','mel_cepstral_distortion','pitch_median_abs_cents','pitch_voiced_recall','pitch_within50_given_voiced','rmvpe_range_coverage']
    for name in ['recording','physics']:
        rows=[r for r in results if r['model']==name and r['condition']=='reproduction']
        summary['db_reproduction'][name]={k:mean(rows,k) for k in keys}
    for shift in [-12,-5,0,5,7,12]:
        rows=[r for r in results if r['condition']=='score' and r['semitones']==shift]
        summary['db_extrapolation'].append(dict(semitones=shift,**{k:mean(rows,k) for k in keys}))
        for name in ['physics','irodori']:
            row=next(r for r in original if r['model']==name and r['semitones']==shift)
            summary['original_comparison'].append({k:row[k] for k in ['model','semitones',*keys[:3],*keys[4:]]})
    for rd in [.5,1.,1.5,2.,2.5]:
        rows=[r for r in results if r['condition']=='rd' and r['rd']==rd]
        summary['db_rd'].append(dict(rd=rd,**{k:mean(rows,k) for k in keys}))
    logs=[json.loads(s) for s in (ROOT/'artifacts/main/training.jsonl').read_text().splitlines()]
    summary['training']['all_validation_history']=[r for r in logs if 'val_loss' in r]
    import torch
    selected=torch.load(ROOT/'artifacts/main/best.pt',map_location='cpu',weights_only=True)
    summary['training']['best_step']=selected['step']
    summary['training']['selected_optimizer_updates']=int(next(iter(selected['optimizer']['state'].values()))['step'].item())
    for k in ['source_noise_gain','constriction_noise_gain']:
        v=[r[k] for r in logs if k in r];summary['training'][k+'_quantiles']=np.quantile(v,[0,.5,1]).tolist()
    p=summary['anatomy'];a=np.array([p['neutral_area_cm2'],*p['vowel_means'].values()]);nose=np.array(p['nasal_area_cm2'])
    summary['anatomy']['bounded_parameters_verified']=bool(13<=p['length_cm']<=19 and .05<=a.min() and a.max()<=8 and .05<=nose.min() and nose.max()<=8)
    save_json(ROOT/'artifacts/evaluation/summary.json',summary)
    save_json(ROOT/'artifacts/STAGE4_PASS.json',dict(passed=True,selection='lowest held-out validation loss (minimum improvement 0.002)',
        checkpoint_sha256=cksha,best_step=summary['training']['best_step'],dataset_seconds=3600,
        initial=train['initial'],best=train['best'],ratio=train['ratio'],optional_neural_finishing='not used'))
    # Short audition reels: constant per-clip gain, separate from raw metric audio.
    from diffart.dataset import Corpus
    corpus=Corpus();test=[r for r in corpus.manifest['songs'] if r['split']=='test'];dest=ROOT/'listen'
    reels=[]
    def clip(path,a,b,label):
        y,sr=sf.read(path,dtype='float32',always_2d=True);y=y.mean(1);g=gcd(sr,SR);y=resample_poly(y,SR//g,sr//g)
        y=np.pad(y,(0,max(0,round(b*SR)-len(y))))[round(a*SR):round(b*SR)]
        assert len(y)>0 and np.isfinite(y).all()
        rms=float(np.sqrt(np.mean(y*y)));peak=float(np.max(np.abs(y)))
        gain=min(.08/max(rms,1e-9),.95/max(peak,1e-9),100.)
        return y*gain,dict(label=label,source=str(path),source_seconds=[a,b],constant_gain=gain,raw_rms=rms,raw_peak=peak)
    def reel(filename,items):
        ys=[];meta=[];cursor=0.
        for path,a,b,label in items:
            y,m=clip(path,a,b,label);m['reel_start_seconds']=cursor;cursor+=len(y)/SR+.5;ys.extend([y,np.zeros(SR//2,np.float32)]);meta.append(m)
        y=np.concatenate(ys);sf.write(dest/filename,y,SR,subtype='PCM_16')
        save_json(dest/(Path(filename).stem+'.json'),dict(clips=meta,separator_seconds=.5,loudness='each clip constant gain toward RMS 0.08, peak <=0.95; not used by metrics'))
        reels.append(dict(file=filename,seconds=len(y)/SR,clips=len(meta)))
    items=[]
    for song in test:
        phrase=corpus.phrase(song);a=phrase['start_frame']*.01;b=a+phrase['frames']*.01
        for name in ['recording','reproduction_+0']:
            items.append((dest/song['id']/f'{name}.wav',a,b,song['id']+' / '+name))
    reel('01_再現.wav',items)
    song=test[0];phrase=corpus.phrase(song);a=phrase['start_frame']*.01;b=a+phrase['frames']*.01
    reel('02_外挿.wav',[(dest/song['id']/f'score_{s:+d}.wav',a,b,f'{s:+d} semitones') for s in [-12,-5,0,5,7,12]])
    reel('03_発声の変化.wav',[(dest/song['id']/f'rd_{rd:.1f}.wav',a,b,f'Rd={rd}') for rd in [.5,1.,1.5,2.,2.5]])
    items=[]
    for shift in [-12,-5,0,5,7,12]:
        for name in ['physics','irodori']:
            path=dest/'original_verse1'/f'physics_{shift:+d}.wav' if name=='physics' else dest/'original_verse1'/f'irodori_{shift:+d}/vocal_ritsu.wav'
            items.append((path,30,36,f'{shift:+d} semitones / {name}'))
    reel('04_モデル間比較.wav',items)
    save_json(dest/'index.json',reels)
    (dest/'README.md').write_text('''# 試聴用（内部研究用・外部公開しない）

波音リツ（歌唱・提供：カノン）。条件と結果は ../STATUS.md、詳細は ../artifacts/evaluation/summary.json。

まず `01_再現.wav`、`02_外挿.wav`、`03_発声の変化.wav`、`04_モデル間比較.wav` を順に試聴してください。

- 01：評価4曲の最初の完全なフレーズを、各曲「元録音→物理モデル」の順。
- 02：最初の評価曲、−12、−5、0、+5、+7、+12半音。
- 03：同じフレーズ、Rd=0.5、1.0、1.5、2.0、2.5。
- 04：自作曲「まだない声で」1番のサビ冒頭30〜36秒。−12、−5、0、+5、+7、+12半音の順で、各移調「物理モデル→Irodori Zv2 15000」。DBの楽譜はIrodoriに渡していません。

各区間の間は0.5秒。上の4本は聴きやすさのため区間ごとに一定ゲインで音量を調整しています。調整量・元ファイル・切り出し時刻は同名JSON。評価値は音量調整前の音声で計算しました。

各評価曲のフォルダーには元録音と全曲の再現・外挿・Rd別のfloat WAVがあります。comparison.wav は元録音→再現→全6移調、rd_comparison.wav は再現→全5 Rd。これらは全区間に共通のゲインです。original_verse1/には両モデルの1番全体と各移調の比較があります。noise_diagnostic/ は雑音経路ごとの切り分けです。

ニューラル仕上げは任意段階として未学習・未評価。試聴による自然さ・本人らしさの判定は未実施です。推定断面積は人体の範囲へ制約した解の1つで、実際の解剖の同定を意味しません。
''')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    v=summary['training']['all_validation_history'];fig,ax=plt.subplots(figsize=(7,4));ax.plot([r['step']/train['epoch_blocks'] for r in v],[r['val_loss'] for r in v],marker='o');ax.set(xlabel='Passes through sampled training hour',ylabel='Held-out validation spectral loss');fig.tight_layout();fig.savefig(ROOT/'artifacts/main/validation.png',dpi=150)
    fig,axes=plt.subplots(2,2,figsize=(11,8),sharex=True)
    shifts=[-12,-5,0,5,7,12]
    for column,metric in enumerate(['wavlm_cosine','ecapa_cosine']):
        ax=axes[0,column]
        for song in test:
            rs=sorted([r for r in results if r['song']==song['id'] and r['condition']=='score'],key=lambda r:r['semitones'])
            ax.plot(shifts,[r[metric] for r in rs],alpha=.3,linewidth=1)
        ax.plot(shifts,[r[metric] for r in summary['db_extrapolation']],color='black',marker='o',label='Physics mean (4 held-out songs)')
        ax.axhline(summary['db_reproduction']['recording'][metric],color='gray',linestyle='--',label='Actual recordings, unshifted')
        ax.set(title='DB internal / '+metric,ylabel='Cosine similarity');ax.legend(fontsize=8)
        ax=axes[1,column]
        for name in ['physics','irodori']:
            rs=sorted([r for r in original if r['model']==name],key=lambda r:r['semitones'])
            ax.plot(shifts,[r[metric] for r in rs],marker='o',label=name)
        ax.set(title='Original composition, first verse (single score / one seed)',xlabel='Semitone shift',ylabel='Cosine similarity');ax.legend()
    fig.tight_layout();fig.savefig(ROOT/'artifacts/evaluation/similarity.png',dpi=150)
    print(json.dumps({'reels':reels,'reproduction':summary['db_reproduction'],'best_step':summary['training']['best_step']},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
