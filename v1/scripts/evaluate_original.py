"""Matched original-composition comparison, never imports a DB score."""
import argparse,json
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from math import gcd
from diffart.data import ROOT,SR,sha
from diffart.runtime import device_lease,save_json

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--device',default='cpu');args=ap.parse_args()
    with device_lease(args.device) as device:
        import torch
        from diffart.dataset import Corpus
        from diffart.physics import Synthesizer
        from diffart.render import render
        from diffart.metrics import kana
        from evaluation_support import Assessment,slope_rows
        torch.set_num_threads(3);corpus=Corpus();assessment=Assessment(corpus,device)
        ck=ROOT/'artifacts/main/best.pt';cksha=sha(ck)
        model=Synthesizer(len(corpus.manifest['vocab'])).to(device)
        model.load_state_dict(torch.load(ck,map_location=device,weights_only=True)['model']);model.eval()
        folder=ROOT/'artifacts/comparison';front=dict(np.load(folder/'physics_front.npz'))
        provenance=json.loads((folder/'frontend_provenance.json').read_text())
        assert provenance['db_score_used'] is False and provenance['physics_front_sha256']==sha(folder/'physics_front.npz')
        dest=ROOT/'listen/original_verse1';dest.mkdir(exist_ok=True)
        results=[]
        for shift in [-12,-5,0,5,7,12]:
            f0=front['f0']*2**(shift/12)
            for name in ['physics','irodori']:
                cache=folder/f'{name}_{shift:+d}.json'
                if cache.exists():
                    row=json.loads(cache.read_text());assert row['physics_checkpoint_sha256']==cksha
                else:
                    if name=='physics':
                        y=render(model,front['context'],front['pos'],f0,device=device)
                        path=dest/f'physics_{shift:+d}.wav';sf.write(path,y,SR,subtype='FLOAT')
                    else:
                        path=dest/f'irodori_{shift:+d}/vocal_ritsu.wav'
                        assert (path.parent/'completed.json').exists()
                        y,sr=sf.read(path,dtype='float32');g=gcd(sr,SR);y=resample_poly(y,SR//g,sr//g)
                    row=assessment.assess(y,f0,expected_text=kana(provenance['lyrics']) if shift==0 else None)
                    row.update(song='original_verse1',model=name,condition='score',semitones=shift,
                        path=str(path),physics_checkpoint_sha256=cksha,original_source_sha256=provenance['source_sha256'])
                    save_json(cache,row)
                results.append(row);print(name,shift,row['wavlm_cosine'],row['ecapa_cosine'],flush=True)
            # Same score excerpt [30,36)s at every shift; model order fixed.
            clips=[]
            for name in ['physics','irodori']:
                p=dest/f'physics_{shift:+d}.wav' if name=='physics' else dest/f'irodori_{shift:+d}/vocal_ritsu.wav'
                y,sr=sf.read(p,dtype='float32');g=gcd(sr,SR);y=resample_poly(y,SR//g,sr//g)
                y=np.pad(y,(0,max(0,36*SR-len(y))));clips.extend([y[30*SR:36*SR],np.zeros(SR//2,np.float32)])
            audio=np.concatenate(clips);gain=min(1.,.95/max(float(np.max(np.abs(audio))),1e-8))
            sf.write(dest/f'comparison_{shift:+d}.wav',audio*gain,SR,subtype='PCM_16')
            save_json(dest/f'comparison_{shift:+d}_order.json',dict(order=['physics','irodori'],score_seconds=[30,36],common_gain=gain,separator_seconds=.5,semitones=shift))
        save_json(folder/'results.json',results);save_json(folder/'slopes.json',slope_rows(results))
        save_json(ROOT/'artifacts/EVALUATION_PARTIAL.json',dict(physics_complete=True,original_comparison_complete=True,
            all_required_stage5_conditions_measured=True,optional_neural_finishing='not trained',human_listening='not performed by agent'))
if __name__=='__main__':main()
