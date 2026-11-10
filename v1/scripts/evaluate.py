"""Evaluate all four held-out DB songs, physical model only (DB-internal)."""
import argparse,json
import numpy as np
import soundfile as sf
from diffart.data import ROOT,SR,HOP,read_lab,sha
from diffart.runtime import device_lease,save_json

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--device',default='cpu');ap.add_argument('--threads',type=int,default=3);args=ap.parse_args()
    assert (ROOT/'artifacts/main/result.json').exists(),'No completed main training'
    assert json.loads((ROOT/'artifacts/OVERFIT_GATE.json').read_text())['passed'] is True
    with device_lease(args.device) as device:
        import torch
        from diffart.dataset import Corpus
        from diffart.physics import Synthesizer
        from diffart.render import render
        from diffart.metrics import label_kana,mel_cepstral_distortion
        from evaluation_support import Assessment,slope_rows
        torch.set_num_threads(args.threads)
        corpus=Corpus();model=Synthesizer(len(corpus.manifest['vocab'])).to(device)
        ck=ROOT/'artifacts/main/best.pt';cksha=sha(ck)
        model.load_state_dict(torch.load(ck,map_location=device,weights_only=True)['model']);model.eval()
        assessment=Assessment(corpus,device)
        save_json(ROOT/'artifacts/evaluation/provenance.json',dict(checkpoint_sha256=cksha,
            manifest_sha256=sha(ROOT/'data/manifest.json'),evaluation_device=device,recording_metrics='CPU precomputed when available; see metrics_device in results',references_file='references.json',
            wavlm_path=str(assessment.sv.model.config._name_or_path),ecapa_revision='0f99f2d0ebe89ac095bcc5903c4dd8f72b367286',
            db_scores_used_by='physical model only; no DB score exported to another model',
            comparison='separate original composition in artifacts/comparison',
            neural_finishing='optional; not trained or evaluated',
            metrics='equal weight per non-silent-reference <=20s clip for speaker metrics; frame-weighted pitch; ASR on reproduction only',
            mcd_definition='Aligned log-mel cepstra DCT-II 1:25; 80 mel; magnitude; active reference frames; 10/ln10 * sqrt(2 sum delta²)'))
        results=[]
        conditions=[('reproduction',0,None)]+[('score',s,None) for s in [-12,-5,0,5,7,12]]+[('rd',0,x) for x in [.5,1.,1.5,2.,2.5]]
        for song in [r for r in corpus.manifest['songs'] if r['split']=='test']:
            d=corpus.load(song);n=len(d['f0']);target=d['y'][:n*HOP]
            dest=ROOT/'listen'/song['id'];dest.mkdir(parents=True,exist_ok=True)
            sf.write(dest/'recording.wav',target,SR,subtype='FLOAT')
            text=label_kana([p for a,b,p in read_lab(song['lab'])])
            cache=ROOT/'artifacts/evaluation'/f"{song['id']}__recording.json"
            if cache.exists():real=json.loads(cache.read_text());assert real['checkpoint_sha256']==cksha
            else:
                precache=ROOT/'artifacts/evaluation/recordings_precomputed.json'
                precached=json.loads(precache.read_text()) if precache.exists() else None
                if precached:
                    assert precached['manifest_sha256']==sha(ROOT/'data/manifest.json')
                    assert all(sha(r['path'])==r['sha256'] for r in precached['references'])
                candidates=[r for r in precached['results'] if r['song']==song['id']] if precached else []
                if candidates:
                    real=dict(candidates[0])
                    assert real['features_sha256']==sha(ROOT/'data/features'/f"{song['id']}.npz")
                    assert real['f0_sha256']==sha(ROOT/'data/f0'/f"{song['id']}.npz")
                else:real=assessment.assess(target,d['f0'],active_wave=target,expected_text=text)
                real.update(song=song['id'],model='recording',condition='reproduction',semitones=0,rd=None,checkpoint_sha256=cksha)
                save_json(cache,real)
            results.append(real)
            for kind,shift,rd in conditions:
                f0=(d['f0'] if kind!='score' else d['score_f0'])*2**(shift/12)
                key=f'{kind}_{shift:+d}' if rd is None else f'rd_{rd:.1f}'
                cache=ROOT/'artifacts/evaluation'/f"{song['id']}__{key}.json";path=dest/f'{key}.wav'
                if cache.exists():
                    row=json.loads(cache.read_text());assert row['checkpoint_sha256']==cksha and path.exists()
                else:
                    y=render(model,d['context'],d['pos'],f0,device=device,rd=rd)
                    assert np.isfinite(y).all();sf.write(path,y,SR,subtype='FLOAT')
                    row=assessment.assess(y,f0,active_wave=target,expected_text=text if kind=='reproduction' else None)
                    row.update(song=song['id'],model='physics',condition=kind,semitones=shift,rd=rd,
                        mel_cepstral_distortion=mel_cepstral_distortion(target,y) if kind=='reproduction' else None,
                        path=str(path),checkpoint_sha256=cksha,peak=float(np.max(np.abs(y))),rms=float(np.sqrt(np.mean(y*y))))
                    save_json(cache,row)
                results.append(row);save_json(ROOT/'artifacts/evaluation/results_in_progress.json',results)
                print(song['id'],key,row['wavlm_cosine'],row['ecapa_cosine'],flush=True)
            phrase=corpus.phrase(song);a=phrase['start_frame']*HOP;b=a+phrase['frames']*HOP
            for filename,keys in [('comparison',['recording','reproduction_+0','score_-12','score_-5','score_+0','score_+5','score_+7','score_+12']),
                                  ('rd_comparison',['reproduction_+0','rd_0.5','rd_1.0','rd_1.5','rd_2.0','rd_2.5'])]:
                clips=[]
                for key in keys:
                    y,_=sf.read(dest/f'{key}.wav',dtype='float32');clips.extend([y[a:b],np.zeros(SR//3,np.float32)])
                compare=np.concatenate(clips);gain=min(1.,.95/max(np.max(np.abs(compare)),1e-8))
                sf.write(dest/f'{filename}.wav',compare*gain,SR,subtype='PCM_16')
                save_json(dest/f'{filename}_order.json',dict(order=keys,common_gain=gain,phrase=phrase,separator_seconds=1/3))
        save_json(ROOT/'artifacts/evaluation/results.json',results)
        save_json(ROOT/'artifacts/evaluation/slopes.json',slope_rows(results))
        save_json(ROOT/'artifacts/EVALUATION_PARTIAL.json',dict(physics_complete=True,
            original_comparison_complete=False,optional_neural_finishing='not trained',count=len(results)))
if __name__=='__main__':main()
