"""LF glottal flow and frequency-domain lossy Kelly–Lochbaum tubes.

Pressure/volume-velocity ABCD convention, normalized rho*c=1 (area in cm²).
Rigid glottal volume-flow drive; pressure-release mouth with first-order radiation.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F

C = 350.0

def bounded(x, lo, hi):return lo+(hi-lo)*x.sigmoid()

def lf_parameters(rd):
    """Fant Rd -> Ra,Rk,Rg, then solve LF continuity and zero cycle integral.

    Roots found without a graph; implicit-root correction supplies d(root)/d(Rd).
    Returned flow has zero at cycle boundaries and peak flow 1 at Tp.
    """
    ra=-.01+.048*rd;rk=.224+.118*rd
    rg=(rk/4)*(.5+1.2*rk)/(.11*rd-ra*(.5+1.2*rk))
    tp=1/(2*rg);te=tp*(1+rk);tb=1-te;w=math.pi/tp
    with torch.no_grad():
        lo=torch.full_like(rd,1e-3);hi=torch.full_like(rd,1000.)
        for _ in range(30):
            e=(lo+hi)/2
            val=-torch.expm1(-e*tb)/e-ra
            lo=torch.where(val>0,e,lo);hi=torch.where(val>0,hi,e)
        eps=(lo+hi)/2
    g=-torch.expm1(-eps*tb)/eps-ra
    dg=((eps*tb+1)*torch.exp(-eps*tb)-1)/eps.square()
    eps=eps-g/dg.detach()
    shift=torch.exp(-eps*tb)
    ret=-(1-shift)/(eps.square()*ra)+tb*shift/(eps*ra)
    def integral(alpha,t):
        # E0 exp(alpha*t) sin(w*t), E(Te)=-1.
        return -(torch.exp(alpha*(t-te))*(alpha*torch.sin(w*t)-w*torch.cos(w*t))
            +w*torch.exp(-alpha*te))/((alpha.square()+w.square())*torch.sin(w*te))
    with torch.no_grad():
        lo=torch.full_like(rd,-80.);hi=torch.full_like(rd,100.)
        for _ in range(30):
            a=(lo+hi)/2;g=integral(a,te)+ret
            # Open integral decreases with alpha in this parameter range.
            lo=torch.where(g>0,a,lo);hi=torch.where(g>0,hi,a)
        alpha=(lo+hi)/2
    g=integral(alpha,te)+ret
    da=(integral(alpha+.001,te)-integral(alpha-.001,te))/.002
    alpha=alpha-g/da.detach()
    peak=integral(alpha,tp)
    return tp,te,eps,alpha,w,shift,peak

def lf_flow(phase,rd,parameters=None):
    tp,te,eps,alpha,w,shift,peak=lf_parameters(rd) if parameters is None else parameters
    opening=-(torch.exp(alpha*(phase-te))*(alpha*torch.sin(w*phase)-w*torch.cos(w*phase))
        +w*torch.exp(-alpha*te))/((alpha.square()+w.square())*torch.sin(w*te))
    ue=-(alpha*torch.sin(w*te)-w*torch.cos(w*te)+w*torch.exp(-alpha*te))/((alpha.square()+w.square())*torch.sin(w*te))
    dt=(phase-te).clamp_min(0)
    ra=-.01+.048*rd
    closing=ue-(-torch.expm1(-eps*dt)/eps-dt*shift)/(eps*ra)
    return torch.where(phase<=te,opening,closing)/peak

def product(m):
    """Balanced ABCD matrix product, section axis -4? input [..., N, F,2,2]."""
    while m.shape[-4]>1:
        n=m.shape[-4];pairs=n//2
        v=m[...,0:2*pairs:2,:,:,:] @ m[...,1:2*pairs:2,:,:,:]
        m=torch.cat([v,m[...,-1:,:,:,:]],dim=-4) if n%2 else v
    return m.squeeze(-4)

def tube_matrices(areas,length,loss,freq):
    # areas [..., N], scalar/tensor length in meters, loss in nepers / meter.
    n=areas.shape[-1]
    gamma=(loss+2j*math.pi*freq/C)*(length/n)
    co=torch.cosh(gamma);si=torch.sinh(gamma)
    z=1/areas[...,None]
    co=co+torch.zeros_like(z)
    return torch.stack([torch.stack([co,z*si],-1),torch.stack([si/z,co],-1)],-2)

def transfer(areas,length,loss,freq,nose=None,velum=None,nose_length=.12,branch=13):
    matrices=tube_matrices(areas,length,loss,freq)
    if nose is not None:
        nasal=product(tube_matrices(nose,torch.as_tensor(nose_length,device=areas.device),loss,freq))
        # Pressure release nostril: input admittance D/B. Velum gate is a shunt.
        adm=nasal[...,1,1]/(nasal[...,0,1]+1e-9)
        adm=adm*velum[...,None].square()
        z=torch.zeros_like(adm);one=torch.ones_like(adm)
        shunt=torch.stack([torch.stack([one,z],-1),torch.stack([adm,one],-1)],-2)
        matrices=torch.cat([matrices[...,:branch,:,:,:],shunt.unsqueeze(-4),matrices[...,branch:,:,:,:]],-4)
    full=product(matrices)
    return 1/(full[...,1,1]+1e-9)

def radiation(freq,sr,coefficient):
    z=torch.exp(-2j*math.pi*freq/sr)
    return (1-z)/(1-coefficient*z)

class Speaker(nn.Module):
    def __init__(self,n=44,rank=9):
        super().__init__()
        self.length_raw=nn.Parameter(torch.tensor(0.))
        self.neutral_raw=nn.Parameter(torch.full((n,),-1.))
        # Smooth DCT initial directions; all basis vectors are subsequently learned.
        x=(torch.arange(n)+.5)/n
        self.basis=nn.Parameter(.25*torch.cos(math.pi*torch.arange(1,rank+1)[:,None]*x))
        self.nose_raw=nn.Parameter(torch.full((20,),-1.))
        self.wall_raw=nn.Parameter(torch.tensor(-.5))
        self.radiation_raw=nn.Parameter(torch.tensor(0.))
        self.rd_raw=nn.Parameter(torch.tensor(-.4))
    @property
    def length(self):return bounded(self.length_raw,.13,.19)
    @property
    def wall(self):return bounded(self.wall_raw,.2,8.)
    @property
    def rad(self):return bounded(self.radiation_raw,.5,.98)
    @property
    def nose(self):return bounded(self.nose_raw,.05,8.)
    def area(self,art):return bounded(self.neutral_raw+art@self.basis,.05,8.)

class Articulation(nn.Module):
    def __init__(self,vocab,rank=9):
        super().__init__()
        self.embedding=nn.Embedding(vocab,16)
        self.context=nn.Sequential(nn.Linear(16*3+3,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh())
        # Only these two source controls receive F0.
        self.shape=nn.Linear(64,rank+4) # art, velum, breath, turbulence, source location
        self.source=nn.Sequential(nn.Linear(65,32),nn.Tanh(),nn.Linear(32,2))
    def forward(self,phones,pos,f0,speaker):
        feat=torch.cat([self.embedding(phones).flatten(-2),pos[...,None],
                        torch.sin(math.pi*pos)[...,None],torch.cos(math.pi*pos)[...,None]],-1)
        hidden=self.context(feat);p=self.shape(hidden)
        src=self.source(torch.cat([hidden,torch.log2(f0.clamp_min(30)/220)[...,None]],-1))
        return dict(art=2*p[...,:9].tanh(),velum=p[...,9].sigmoid(),
            breath=p[...,10].sigmoid(),turbulence=p[...,11].sigmoid(),location=p[...,12].sigmoid(),
            rd=bounded(src[...,0]+speaker.rd_raw,.3,2.7),amplitude=F.softplus(src[...,1]-2))

class Synthesizer(nn.Module):
    def __init__(self,vocab,sr=24000,hop=240,nfft=1024):
        super().__init__();self.speaker=Speaker();self.controls=Articulation(vocab)
        self.sr=sr;self.hop=hop;self.nfft=nfft
        self.register_buffer('window',torch.hann_window(nfft))
        self.register_buffer('freq',torch.fft.rfftfreq(nfft,1/sr))
    def forward(self,phones,pos,f0,seed=0,rd_override=None,return_controls=False,phase_offset=0.):
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
        jitter=torch.randn(b,count,device=f0.device,generator=gen)*.002
        shimmer=1+torch.randn(b,count,device=f0.device,generator=gen)*.01
        phase=(torch.cumsum(freq*(1+jitter.gather(1,cycles))/self.sr,-1)+phase_offset).remainder(1)
        params=tuple(up(x) for x in lf_parameters(p['rd']))
        flow=lf_flow(phase,up(p['rd']),params)*(freq>0)*shimmer.gather(1,cycles)
        noise=torch.randn(b,ns,device=f0.device,generator=gen)
        source=up(p['amplitude'])*(flow+.03*up(p['breath'])*noise)
        spec=torch.stft(source,self.nfft,self.hop,window=self.window,return_complex=True,pad_mode='constant').transpose(-1,-2)
        # Controls at the STFT center time: input starts at frame 0.
        areas=self.speaker.area(p['art']); areas_stft=torch.cat([areas,areas[:,-1:]],1)
        velum=torch.cat([p['velum'],p['velum'][:,-1:]],1)
        h=transfer(areas_stft,self.speaker.length,self.speaker.wall,self.freq,self.speaker.nose,velum)
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
        for j in range(43,-1,-1):
            if j==12:suffix=shunt@suffix
            suffix=mats[:,:,j]@suffix
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
    def regularization(self,p):
        a=p['areas'];spatial=(a[...,1:]-a[...,:-1]).square().mean()
        temporal=(a[...,1:,:]-a[...,:-1,:]).square().mean()
        return 1e-3*spatial+1e-3*temporal+1e-4*self.speaker.basis.square().mean()
