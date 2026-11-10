"""Full evaluate.py protocol, isolated B artifacts and read-only old results.

Uses its Assessment.assess and slope_rows unchanged: 4 songs, whole duration,
same 20 s windows, anchors, pitch/ASR/MCD, 12 conditions, deterministic render.
"""
from diffart.paths import input_path, required_path
import argparse
import json
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from diffart.data import ROOT, SR, HOP, sha, read_lab
from diffart.runtime import device_lease, save_json


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('variant',choices=['b1','b2','both'])
    ap.add_argument('--device',default='cpu')
    ap.add_argument('--threads',type=int,default=2)
    args=ap.parse_args()
    with device_lease(args.device) as device:
        import torch
        from diffart.dataset import Corpus
        from diffart.improved import ImprovedSynthesizer
        from diffart.render import render
        from diffart.metrics import (SpeakerMetrics, EcapaMetrics, label_kana,
                                     mel_cepstral_distortion)
        from evaluation_support import Assessment, slope_rows
        torch.set_num_threads(args.threads)
        out=ROOT/'artifacts/improvements'/args.variant
        evaluation=out/'evaluation';evaluation.mkdir(parents=True,exist_ok=True)
        assert (out/'result.json').exists(), 'Training must complete before evaluation'
        c=Corpus();ck=out/'best.pt';cksha=sha(ck)
        m=ImprovedSynthesizer(len(c.manifest['vocab']),b1=args.variant in ('b1','both'),b2=args.variant in ('b2','both')).to(device)
        m.load_state_dict(torch.load(ck,map_location=device,weights_only=True)['model']);m.eval()
        # Build the same assessment without its shared-directory writes.
        a=Assessment.__new__(Assessment)
        a.sv=SpeakerMetrics(device);a.ht=None;a.device=device
        import torchaudio
        if not hasattr(torchaudio,'list_audio_backends'):torchaudio.list_audio_backends=lambda:['soundfile']
        from speechbrain.inference.speaker import EncoderClassifier
        source=ROOT/'artifacts/models/ecapa'
        a.ecapa=EcapaMetrics.__new__(EcapaMetrics);a.ecapa.device=device
        a.ecapa.model=EncoderClassifier.from_hparams(source=str(source),savedir=str(evaluation/'ecapa_cache'),overrides={'pretrained_path':str(source)},run_opts={'device':device})
        import sys
        sys.path.insert(0,str(required_path('RMVPE_ROOT')))
        from preprocess.algorithms.rmvpe import RMVPEPitchAlgorithm
        a.pitch=RMVPEPitchAlgorithm(model_path=str(required_path('RMVPE_WEIGHTS')),device='cuda' if device.startswith('cuda') else 'cpu',sample_rate=SR,hop_size=HOP,fmin=31.7,fmax=2005.5)
        refpath=ROOT/'artifacts/evaluation/references.json'
        refs=json.loads(refpath.read_text());waves=[]
        for r in refs:
            y,sr=sf.read(r['path'],dtype='float32');assert sr==SR
            waves.append(resample_poly(y,2,3))
        a.anchors={name:np.stack([metric.embedding(y) for y in waves]) for name,metric in [('wavlm',a.sv),('ecapa',a.ecapa)]}
        save_json(evaluation/'provenance.json',dict(checkpoint_sha256=cksha,manifest_sha256=sha(ROOT/'data/manifest.json'),references=[dict(path=r['path'],sha256=sha(r['path'])) for r in refs],reference_selection_sha256=sha(refpath),evaluation_support_sha256=sha(ROOT/'scripts/evaluation_support.py'),render_sha256=sha(ROOT/'diffart/render.py'),device=device,protocol='scripts/evaluate.py full 4 songs, same Assessment.assess, render and conditions'))
        results=[]
        conditions=[('reproduction',0,None)]+[('score',s,None) for s in [-12,-5,0,5,7,12]]+[('rd',0,x) for x in [.5,1.,1.5,2.,2.5]]
        for song in [r for r in c.manifest['songs'] if r['split']=='test']:
            d=c.load(song);n=len(d['f0']);target=d['y'][:n*HOP]
            dest=out/'audio'/song['id'];dest.mkdir(parents=True,exist_ok=True)
            real=json.loads((ROOT/'artifacts/evaluation'/f"{song['id']}__recording.json").read_text())
            results.append(real)
            text=label_kana([p for _,_,p in read_lab(song['lab'])])
            for kind,shift,rd in conditions:
                f0=(d['f0'] if kind!='score' else d['score_f0'])*2**(shift/12)
                key=f'{kind}_{shift:+d}' if rd is None else f'rd_{rd:.1f}'
                cache=evaluation/f"{song['id']}__{key}.json";path=dest/f'{key}.wav'
                if cache.exists():
                    row=json.loads(cache.read_text());assert row['checkpoint_sha256']==cksha and path.exists()
                else:
                    if path.exists():
                        y,sr=sf.read(path,dtype='float32');assert sr==SR and len(y)==n*HOP
                    else:
                        y=render(m,d['context'],d['pos'],f0,device=device,rd=rd)
                        assert np.isfinite(y).all();sf.write(path,y,SR,subtype='FLOAT')
                    row=a.assess(y,f0,active_wave=target,expected_text=text if kind=='reproduction' else None)
                    row.update(song=song['id'],model=args.variant,condition=kind,semitones=shift,rd=rd,mel_cepstral_distortion=mel_cepstral_distortion(target,y) if kind=='reproduction' else None,path=str(path),checkpoint_sha256=cksha,peak=float(np.max(np.abs(y))),rms=float(np.sqrt(np.mean(y*y))))
                    save_json(cache,row)
                results.append(row);save_json(evaluation/'results_in_progress.json',results)
                print(song['id'],key,row['wavlm_cosine'],row['ecapa_cosine'],flush=True)
        save_json(evaluation/'results.json',results);save_json(evaluation/'slopes.json',slope_rows(results))

if __name__=='__main__':main()
