"""RMVPE, chunked with .2s context, resumable, no pitch-estimator substitution."""
from diffart.paths import input_path, required_path
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import soundfile as sf
from diffart.runtime import device_lease, save_json
from diffart.data import ROOT, SR, HOP, sha

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--device',default='cpu');ap.add_argument('--threads',type=int,default=3)
    args=ap.parse_args()
    with device_lease(args.device) as device:
        import torch
        torch.set_num_threads(args.threads)
        upstream=input_path('RMVPE_ROOT','external/rmvpe')
        sys.path.insert(0,str(upstream))
        from preprocess.algorithms.rmvpe import RMVPEPitchAlgorithm
        weights=input_path('RMVPE_WEIGHTS','external/rmvpe.pt')
        model=RMVPEPitchAlgorithm(model_path=str(weights),device='cuda' if device.startswith('cuda') else 'cpu',
             sample_rate=SR,hop_size=HOP,fmin=31.7,fmax=2005.5)
        m=json.loads((ROOT/'data/manifest.json').read_text())
        songs=[r for r in m['songs'] if not r['excluded']]
        # Stage 1 fully prepares ALL eligible songs, not just the training subset.
        start=time.monotonic()
        for k,r in enumerate(songs):
            dst=ROOT/'data/f0'/f"{r['id']}.npz"; dst.parent.mkdir(parents=True,exist_ok=True)
            if dst.exists(): continue
            y,_=sf.read(ROOT/'data/audio24'/f"{r['id']}.wav",dtype='float32')
            n=len(y)//HOP; f0=np.zeros(n,np.float32);conf=np.zeros(n,np.float32)
            for first in range(0,n,1000):
                last=min(first+1000,n);a=max(0,first-20);b=min(n,last+20)
                pitch,period=model.extract_continuous_periodicity(y[a*HOP:b*HOP])
                off=first-a;cnt=last-first
                f0[first:last]=pitch[off:off+cnt];conf[first:last]=period[off:off+cnt]
            f0[conf<.03]=0
            assert np.isfinite(f0).all()
            np.savez_compressed(dst,f0=f0,confidence=conf)
            progress=dict(done=k+1,total=len(songs),song=r['id'],elapsed=time.monotonic()-start,
                 device=device,weights=str(weights),weights_sha256=sha(weights),voiced_fraction=float((f0>0).mean()))
            save_json(ROOT/'artifacts/f0_progress.json',progress);print(json.dumps(progress),flush=True)
        save_json(ROOT/'artifacts/F0_COMPLETE.json',dict(songs=len(songs),device=device,weights_sha256=sha(weights)))
if __name__=='__main__':main()
