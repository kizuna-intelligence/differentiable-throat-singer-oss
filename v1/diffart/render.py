"""Bounded-memory synthesis with phase offsets and overlap-add boundaries."""
import numpy as np
import torch
from .data import SR,HOP

@torch.inference_mode()
def render(model,context,pos,f0,device='cpu',rd=None,chunk_frames=400,overlap=40):
    n=len(f0);total=n*HOP;y=np.zeros(total,np.float32);weight=np.zeros(total,np.float32)
    # Absolute integrated F0 keeps the large-scale oscillator phase through chunks.
    offsets=np.concatenate([[0],np.cumsum(f0.astype(np.float64)*HOP/SR)])
    stride=chunk_frames-overlap
    for a in range(0,n,stride):
        b=min(a+chunk_frames,n)
        inp=[torch.as_tensor(v[a:b],device=device) for v in [context,pos,f0]]
        out=model(*inp,seed=1234,rd_override=rd,phase_offset=float(offsets[a])).cpu().numpy()
        w=np.ones(len(out),np.float32);fade=min(overlap*HOP,len(out))
        if a:w[:fade]=np.linspace(0,1,fade)
        if b<n:w[-fade:]=np.linspace(1,0,fade)
        start=a*HOP;y[start:start+len(out)]+=out*w;weight[start:start+len(out)]+=w
        if b==n:break
    return y/np.maximum(weight,1e-8)
