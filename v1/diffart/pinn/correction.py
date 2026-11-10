"""Frozen, FEM-supervised log-magnitude/phase MLP and grid interpolation.

This is a data-only surrogate, not a PDE-trained PINN. The name identifies
design C; field/PDE training is intentionally omitted in this short experiment.
"""
from pathlib import Path
import torch
from torch import nn


class RatioMLP(nn.Module):
    def __init__(self,width=128):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(9,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU(),
                               nn.Linear(width,width),nn.SiLU(),nn.Linear(width,2))

    def forward(self,x): return self.net(x)


class FEMCorrection(nn.Module):
    def __init__(self,metadata):
        super().__init__()
        self.mlp=RatioMLP(metadata.get('width',128))
        for key in ('mean','basis','coeff_mean','coeff_std','target_mean','target_std','frequencies'):
            self.register_buffer(key,torch.as_tensor(metadata[key],dtype=torch.float32))

    @classmethod
    def load(cls,path):
        ck=torch.load(Path(path),map_location='cpu',weights_only=True)
        model=cls(ck['metadata'])
        model.load_state_dict(ck['model'])
        return model.eval().requires_grad_(False)

    def grid(self,areas):
        coeff=(torch.log(areas.clamp_min(1e-5)*1e-4)-self.mean)@self.basis.T
        c=(coeff-self.coeff_mean)/self.coeff_std
        c=c.unsqueeze(-2).expand(*c.shape[:-1],len(self.frequencies),8)
        freq=(self.frequencies/4000-1).expand(*c.shape[:-1]).unsqueeze(-1)
        x=torch.cat((c,freq),-1)
        y=self.mlp(x)*self.target_std+self.target_mean
        return y

    def forward(self,areas,frequencies):
        y=self.grid(areas)
        # Interpolate log magnitude and unwrapped phase, never complex Cartesian
        # components near a phase wrap. Constant extrapolation below 50 Hz.
        ix=torch.searchsorted(self.frequencies,frequencies).clamp(1,len(self.frequencies)-1)
        fraction=((frequencies-self.frequencies[ix-1])/(self.frequencies[ix]-self.frequencies[ix-1])).clamp(0,1)
        interp=y[...,ix-1,:]+(y[...,ix,:]-y[...,ix-1,:])*fraction[...,None]
        # Raised-cosine fade of log R from 8 to 10 kHz; exactly unity thereafter.
        fade=(.5+.5*torch.cos(torch.pi*((frequencies-8000)/2000).clamp(0,1)))
        interp=interp*fade[...,None]
        return torch.polar(interp[...,0].exp(),interp[...,1])


def load_synthesizer(checkpoint,vocab):
    from diffart.improved import ImprovedSynthesizer
    ck=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if 'correction_metadata' in ck:
        correction=FEMCorrection(ck['correction_metadata'])
        model=ImprovedSynthesizer(vocab,pinn_correction=correction)
    else:
        model=ImprovedSynthesizer(vocab,b1=any(k.startswith('piriform.') for k in ck['model']),
                                  b2=any(k.startswith('glottis.') for k in ck['model']))
    model.load_state_dict(ck['model'])
    return model
