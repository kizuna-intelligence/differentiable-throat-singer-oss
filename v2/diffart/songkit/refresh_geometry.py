"""Upgrade the currently rendered CPU parts' observation precision, without rerendering audio.

Replays the identical prepare() inputs, checkpoint and model for coefficient observations only.
The 96 kHz wave, noise and solver equations are unchanged. Kept as an explicit audit utility.
"""
from pathlib import Path
import json
import numpy as np
import torch
from . import cpu
from .pipeline import REPO,sha,fingerprint,write
from diffart.chimera.constants import RHO_C2

def refresh(spec,out,checkpoint):
    torch.set_num_threads(4)
    ck=torch.load(checkpoint,map_location='cpu',weights_only=True)
    source_hash=fingerprint({str(p.relative_to(REPO)):sha(p) for p in [REPO/'diffart/songkit/cpu.py',REPO/'diffart/songkit/frontend.py']+list((REPO/'diffart/chimera').glob('*.py'))})
    for i,part in enumerate(spec['parts']):
        folder=out/'parts'/part['id'];rpath=folder/'synthesis.json'
        if not rpath.exists():raise RuntimeError('Synthesis must complete before observation refresh')
        inputs=np.load(folder/'inputs.npz');model=cpu.make_model(ck['config'],ck['config']['vocab'],ck['model'],part['body_scale'])
        with torch.inference_mode():
            _,geo,c,ga,ng,n=model.prepare(torch.as_tensor(inputs['context'][None],dtype=torch.long),torch.as_tensor(inputs['pos'][None]),torch.as_tensor(inputs['f0'][None]),seed=1234+i)
        count=len(model.airway.areas);c=c[0].numpy().astype(np.float64)
        with np.load(folder/'controls.npz') as z:values={k:z[k] for k in z.files}
        assert np.array_equal(values['states'],geo['states'][0].numpy().astype(np.float32))
        original_sha=sha(folder/(part['id']+'.wav'))
        dx=c[:,15,:count];volume=c[:,5,:count]*RHO_C2
        values['solver_volumes']=volume;values['solver_lengths']=dx
        values['solver_areas']=np.divide(volume,dx,out=np.zeros_like(volume),where=dx>0)
        np.savez_compressed(folder/'controls.npz',**values)
        record=json.loads(rpath.read_text());record['fingerprint']=fingerprint(dict(part=part,bpm=spec['bpm'],seconds=spec['seconds'],checkpoint=sha(checkpoint),source=source_hash))
        record['observation_precision_refresh']=dict(states_bit_equal=True,wave_sha256_unchanged=sha(folder/(part['id']+'.wav'))==original_sha,method='Identical prepare(), same checkpoint/inputs/seed; compliance and dx observed as float64 exactly as CPU graph receives them.')
        write(rpath,record);print('precision refreshed '+part['id'],flush=True)
        del model,c,geo,ga,ng,values
