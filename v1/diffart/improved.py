"""Optional B1/B2 physics; the legacy physics module and checkpoint stay intact.

B1: positive-real closed-tube shunts, distributed across adjacent junctions
with nonnegative weights (a passive continuous relaxation of attachment index).
B2: eight positive LF-windowed Bernstein bases with speaker/Rd coefficients,
finite closure, pulsed aspiration, and speaker-specific jitter/shimmer.
"""
import math
import torch
from torch import nn
from .physics import (Synthesizer, bounded, lf_flow, lf_parameters, product,
                      tube_matrices, transfer, radiation, C)
import torch.nn.functional as F


class Piriform(nn.Module):
    def __init__(self):
        super().__init__()
        self.length_raw = nn.Parameter(torch.zeros(2))
        self.area_raw = nn.Parameter(torch.full((2,), math.log(.8 / 1.)))
        self.position_raw = nn.Parameter(torch.zeros(2))
        # Direct constrained coordinates permit exactly zero coupling with gradient.
        self.coupling = nn.Parameter(torch.zeros(2))

    @property
    def length(self): return bounded(self.length_raw, .015, .025)
    @property
    def area(self): return bounded(self.area_raw, .2, 2.)
    @property
    def position(self): return bounded(self.position_raw, .02, .03)

    def admittance(self, freq, loss):
        # Rigid termination U(L)=0: Y_in=C/A = area*tanh(gamma*L).
        return self.coupling.clamp(0, 1)[:, None] * self.area[:, None] * torch.tanh(
            (loss + 2j * math.pi * freq[None] / C) * self.length[:, None])

    def junction_admittance(self, freq, length, loss, n=44):
        x = torch.arange(n, device=freq.device, dtype=freq.dtype) * length / n
        logits = -.5 * ((x[None] - self.position[:, None]) / (length / n * .65)).square()
        weights = logits.masked_fill(~((x >= .02) & (x <= .03))[None], -torch.inf).softmax(-1)
        return torch.einsum('sn,sf->nf', weights.to(self.admittance(freq, loss).dtype),
                            self.admittance(freq, loss))

    @torch.no_grad()
    def project(self): self.coupling.clamp_(0, 1)


class LearnableGlottis(nn.Module):
    def __init__(self):
        super().__init__()
        self.shape = nn.Parameter(torch.zeros(8))
        self.shape_rd = nn.Parameter(torch.zeros(8))
        self.closed = nn.Parameter(torch.tensor(0.))
        self.pulsation = nn.Parameter(torch.tensor(0.))
        self.jitter_raw = nn.Parameter(torch.tensor(0.))
        self.shimmer_raw = nn.Parameter(torch.tensor(0.))
        self.register_buffer('noise_warmup', torch.tensor(0.))

    @property
    def jitter(self): return .002 * self.jitter_raw.clamp(-4, 3).exp()
    @property
    def shimmer(self): return .01 * self.shimmer_raw.clamp(-4, 3).exp()

    def flow(self, phase, rd, parameters=None):
        oq = 1 - self.closed.clamp(0, .45)
        u = (phase / oq).clamp(0, 1)
        base = lf_flow(u, rd, parameters)
        coeff = self.shape + ((rd - 1.5) / 1.2)[..., None] * self.shape_rd
        # Sum Bernstein bases=1. The residual form gives bitwise LF at zero.
        scale = torch.ones_like(u)
        for k in range(8):
            basis = math.comb(7, k) * u.pow(k) * (1-u).pow(7-k)
            scale = scale + torch.expm1(coeff[..., k].clamp(-3, 3)) * basis
        return torch.where(phase < oq, base * scale, torch.zeros_like(base))

    def aspiration(self, breath, flow, voiced):
        # Warm-up preserves exact checkpoint initialization, then a 5% minimum.
        amount = torch.maximum(breath, .05 * self.noise_warmup)
        gate = torch.where(voiced, .15 + .85 * flow.clamp(0, 1), torch.ones_like(flow))
        return amount * (1 + self.pulsation.clamp(0, 1) * (gate-1))

    @torch.no_grad()
    def project(self):
        self.closed.clamp_(0, .45)
        self.pulsation.clamp_(0, 1)
        self.shape.clamp_(-3, 3); self.shape_rd.clamp_(-3, 3)
        self.jitter_raw.clamp_(-4, 3); self.shimmer_raw.clamp_(-4, 3)


class ImprovedSynthesizer(Synthesizer):
    def __init__(self, vocab, b1=False, b2=False, pinn_correction=None, **kwargs):
        super().__init__(vocab, **kwargs)
        self.b1, self.b2 = b1, b2
        if b1: self.piriform = Piriform()
        if b2: self.glottis = LearnableGlottis()
        self.pinn_correction = pinn_correction
        if pinn_correction is not None:
            # The FEM surrogate is frozen; gradients through its area input remain.
            self.pinn_correction.requires_grad_(False)

    def load_legacy(self, state):
        result = self.load_state_dict(state, strict=False)
        assert not result.unexpected_keys, result
        assert all(k.startswith(('piriform.', 'glottis.', 'pinn_correction.')) for k in result.missing_keys), result
        return result

    @torch.no_grad()
    def project(self, warmup=1.):
        if self.b1: self.piriform.project()
        if self.b2:
            self.glottis.project()
            self.glottis.noise_warmup.fill_(min(1., warmup))

    def transfer(self, areas, velum):
        if not self.b1:
            h = transfer(areas, self.speaker.length, self.speaker.wall, self.freq,
                            self.speaker.nose, velum)
            return h if self.pinn_correction is None else h*self.pinn_correction(areas, self.freq)
        mats = tube_matrices(areas, self.speaker.length, self.speaker.wall, self.freq)
        adm = self.piriform.junction_admittance(self.freq, self.speaker.length, self.speaker.wall)
        # Shunt at input of each section. Identity at zero keeps legacy grouping.
        top, bottom = mats[..., 0, :], mats[..., 1, :]
        if not self.speaker.length_raw.requires_grad and not areas.requires_grad:
            # Fixed legacy anatomy: only 2--3 cm junctions need an autograd
            # graph. Avoid backpropagating a 44-section concatenation on CPU.
            x=torch.arange(44,device=areas.device)*self.speaker.length/44
            indices=torch.where((x>=.02)&(x<=.03))[0]
            first,last=int(indices[0]),int(indices[-1])
            mid=mats[...,first:last+1,:,:,:]
            top_mid,bottom_mid=mid[...,0,:],mid[...,1,:]
            mid=torch.stack([top_mid,bottom_mid+adm[first:last+1,:,None]*top_mid],-2)
            prefix=product(mats[...,:first,:,:,:])
            # Nasal junction 13 is always downstream of the 2--3 cm region.
            nasal=product(tube_matrices(self.speaker.nose,torch.as_tensor(.12,device=areas.device),self.speaker.wall,self.freq))
            yn=nasal[...,1,1]/(nasal[...,0,1]+1e-9)*velum[...,None].square()
            z,one=torch.zeros_like(yn),torch.ones_like(yn)
            shunt=torch.stack([torch.stack([one,z],-1),torch.stack([yn,one],-1)],-2)
            tail=product(torch.cat([mats[...,last+1:13,:,:,:],shunt.unsqueeze(-4),mats[...,13:,:,:,:]],-4))
            h=1/((prefix@product(mid)@tail)[...,1,1]+1e-9)
            if bool((self.piriform.coupling==0).all()):
                # Different matrix grouping has roundoff at zero: retain the
                # exact legacy value and the physical right derivative.
                legacy=transfer(areas,self.speaker.length,self.speaker.wall,self.freq,self.speaker.nose,velum)
                return legacy+(h-h.detach())
            return h
        mats = torch.stack([top, bottom + adm[..., None] * top], -2)
        nasal = product(tube_matrices(self.speaker.nose, torch.as_tensor(.12, device=areas.device), self.speaker.wall, self.freq))
        yn = nasal[..., 1, 1] / (nasal[..., 0, 1] + 1e-9) * velum[..., None].square()
        z, one = torch.zeros_like(yn), torch.ones_like(yn)
        shunt = torch.stack([torch.stack([one, z], -1), torch.stack([yn, one], -1)], -2)
        mats = torch.cat([mats[..., :13, :, :, :], shunt.unsqueeze(-4), mats[..., 13:, :, :, :]], -4)
        return 1 / (product(mats)[..., 1, 1] + 1e-9)

    def forward(self,phones,pos,f0,seed=0,rd_override=None,return_controls=False,phase_offset=0.):
        if not self.b1 and not self.b2 and self.pinn_correction is None:
            return super().forward(phones,pos,f0,seed,rd_override,return_controls,phase_offset)
        single=f0.ndim==1
        if single:phones,pos,f0=phones[None],pos[None],f0[None]
        p=self.controls(phones,pos,f0,self.speaker)
        if rd_override is not None:p['rd']=p['rd']*0+rd_override
        b,t=f0.shape;ns=t*self.hop
        gen=torch.Generator(device=f0.device);gen.manual_seed(seed)
        def up(x):return F.interpolate(x[:,None],size=ns,mode='linear',align_corners=False)[:,0]
        # Random perturbations are fixed per approximate cycle, never learned.
        freq=up(f0);base_phase=torch.cumsum(freq/self.sr,-1)+phase_offset
        cycles=base_phase.floor().long();count=int(cycles.max().item())+2
        jitter=torch.randn(b,count,device=f0.device,generator=gen)*(self.glottis.jitter if self.b2 else .002)
        shimmer=1+torch.randn(b,count,device=f0.device,generator=gen)*(self.glottis.shimmer if self.b2 else .01)
        phase=(torch.cumsum(freq*(1+jitter.gather(1,cycles))/self.sr,-1)+phase_offset).remainder(1)
        params=tuple(up(x) for x in lf_parameters(p['rd']))
        flow=(self.glottis.flow(phase,up(p['rd']),params) if self.b2 else lf_flow(phase,up(p['rd']),params))*(freq>0)*shimmer.gather(1,cycles)
        noise=torch.randn(b,ns,device=f0.device,generator=gen)
        aspiration=self.glottis.aspiration(up(p['breath']),flow,freq>0) if self.b2 else up(p['breath'])
        source=up(p['amplitude'])*(flow+.03*aspiration*noise)
        spec=torch.stft(source,self.nfft,self.hop,window=self.window,return_complex=True,pad_mode='constant').transpose(-1,-2)
        # Controls at the STFT center time: input starts at frame 0.
        areas=self.speaker.area(p['art']); areas_stft=torch.cat([areas,areas[:,-1:]],1)
        velum=torch.cat([p['velum'],p['velum'][:,-1:]],1)
        h=self.transfer(areas_stft,velum)
        rad=radiation(self.freq,self.sr,self.speaker.rad)
        # Each possible constriction injects downstream noise. Soft localization
        # remains differentiable; each path is a distinct suffix ABCD product.
        # Accumulate suffix D via a reverse scan, including the nasal shunt when upstream.
        mats=tube_matrices(areas_stft,self.speaker.length,self.speaker.wall,self.freq)
        identity=torch.eye(2,dtype=mats.dtype,device=mats.device).expand(b,t+1,len(self.freq),2,2)
        suffix=identity
        paths=[]
        nasal=product(tube_matrices(self.speaker.nose,torch.as_tensor(.12,device=f0.device),self.speaker.wall,self.freq))
        adm=nasal[...,1,1]/(nasal[...,0,1]+1e-9)*velum[...,None].square()
        zero=torch.zeros_like(adm);one=torch.ones_like(adm)
        shunt=torch.stack([torch.stack([one,zero],-1),torch.stack([adm,one],-1)],-2)
        branch_adm=self.piriform.junction_admittance(self.freq,self.speaker.length,self.speaker.wall) if self.b1 else None
        branch_indices=set(torch.where((torch.arange(44,device=f0.device)*self.speaker.length/44>=.02)&(torch.arange(44,device=f0.device)*self.speaker.length/44<=.03))[0].tolist()) if self.b1 else set()
        for j in range(43,-1,-1):
            if j==12:suffix=shunt@suffix
            suffix=mats[:,:,j]@suffix
            if j in branch_indices:
                top,bottom=suffix[...,0,:],suffix[...,1,:]
                suffix=torch.stack([top,bottom+branch_adm[j,:,None]*top],-2)
            paths.append(1/(suffix[...,1,1]+1e-9))
        paths=torch.stack(paths[::-1],dim=-2)
        location=torch.cat([p['location'],p['location'][:,-1:]],1)
        ix=torch.linspace(0,1,44,device=f0.device)
        narrow=-areas_stft.log()*3-((ix-location[...,None])/.12).square()
        weights=narrow.softmax(-1)
        turbulent_h=(paths*weights[...,None]).sum(-2)
        constriction=(weights/areas_stft).sum(-1)
        # Closure/release: flow is suppressed at a constriction; a positive area
        # change releases a short pulse at that same location (not a neural residual).
        opening=(areas_stft[:,1:]-areas_stft[:,:-1]).relu()
        release=F.pad((opening*weights[:,1:]).sum(-1),(1,0))
        gain=torch.cat([p['amplitude']*p['turbulence'],(p['amplitude']*p['turbulence'])[:,-1:]],1)
        n_spec=torch.stft(noise,self.nfft,self.hop,window=self.window,return_complex=True,pad_mode='constant').transpose(-1,-2)
        filtered=(spec*h/(1+.03*constriction[...,None])+.03*n_spec*turbulent_h*(gain*(constriction+release))[...,None])*rad
        y=torch.istft(filtered.transpose(-1,-2),self.nfft,self.hop,window=self.window,length=ns)
        # Fixed physical-to-digital gain, no per-utterance normalization in loss.
        y=y*.2
        p['areas']=areas
        if single:y=y[0];p={k:v[0] for k,v in p.items()}
        return (y,p) if return_controls else y
