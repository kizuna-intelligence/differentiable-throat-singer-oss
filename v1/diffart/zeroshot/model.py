"""Frozen tone-colour -> anatomy and FiLM; original ImprovedSynthesizer equations.

Per-reference parameters are supplied by functional_call, never written into shared
modules. Body-only adaptation cannot update FiLM, controls, or shared glottis.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.func import functional_call
from diffart.physics import Articulation, bounded
from diffart.improved import ImprovedSynthesizer

BODY_NAMES = (['length_logit'] + [f'area_dct_{i}' for i in range(1,9)] +
              ['rd_center','rd_width','opening_gain','nasal_area','wall_loss','radiation'] +
              [f'glottis_bernstein_{i}' for i in range(8)] + ['glottal_closure','aspiration_pulsation'])

class FilmControl(Articulation):
    def __init__(self, vocab):
        super().__init__(vocab)
        self.register_buffer('film_gain',torch.zeros(64))
        self.register_buffer('film_bias',torch.zeros(64))
        self.register_buffer('rd_width',torch.ones(()))
        self.register_buffer('opening_gain',torch.ones(()))
    def forward(self, phones, pos, f0, speaker):
        feat=torch.cat([self.embedding(phones).flatten(-2),pos[...,None],
                        torch.sin(math.pi*pos)[...,None],torch.cos(math.pi*pos)[...,None]],-1)
        hidden=self.context(feat)*(1+self.film_gain)+self.film_bias
        p=self.shape(hidden)
        src=self.source(torch.cat([hidden,torch.log2(f0.clamp_min(30)/220)[...,None]],-1))
        return dict(art=2*p[...,:9].tanh(),velum=p[...,9].sigmoid(),breath=p[...,10].sigmoid(),
                    turbulence=p[...,11].sigmoid(),location=p[...,12].sigmoid(),
                    rd=bounded(src[...,0]*self.rd_width+speaker.rd_raw,.3,2.7),
                    amplitude=F.softplus(src[...,1]-2)*self.opening_gain)

class ZeroShotSynthesizer(nn.Module):
    def __init__(self,vocab,embedding_dim=256):
        super().__init__()
        self.embedding_dim=embedding_dim
        self.physics=ImprovedSynthesizer(vocab,b1=True,b2=True)
        self.physics.controls=FilmControl(vocab)
        self.voice=nn.Sequential(nn.Linear(embedding_dim,128),nn.Tanh(),nn.Linear(128,64),nn.Tanh())
        self.body_head=nn.Linear(64,len(BODY_NAMES))
        self.film_head=nn.Linear(64,128)
        nn.init.normal_(self.body_head.weight,std=.005);nn.init.zeros_(self.body_head.bias)
        nn.init.normal_(self.film_head.weight,std=.005);nn.init.zeros_(self.film_head.bias)
        x=(torch.arange(44)+.5)/44
        self.register_buffer('body_dct',torch.cos(math.pi*torch.arange(1,9)[:,None]*x))
        self.physics.glottis.noise_warmup.fill_(1.)
    def encode(self,embedding):
        return self.voice(F.normalize(embedding,dim=-1))
    def body(self,embedding):return self.body_head(self.encode(embedding))
    def parameters_for(self,body,film):
        p=self.physics;d=body.tanh();g,b=film.chunk(2,-1)
        return {'speaker.length_raw':p.speaker.length_raw+2*d[0],
                'speaker.neutral_raw':p.speaker.neutral_raw+.7*(d[1:9]@self.body_dct),
                'speaker.rd_raw':p.speaker.rd_raw+1.5*d[9],
                'controls.rd_width':torch.exp(.7*d[10]),
                'controls.opening_gain':torch.exp(d[11]),
                'speaker.nose_raw':p.speaker.nose_raw+d[12],
                'speaker.wall_raw':p.speaker.wall_raw+d[13],
                'speaker.radiation_raw':p.speaker.radiation_raw+d[14],
                'glottis.shape':p.glottis.shape+d[15:23],
                'glottis.closed':p.glottis.closed+.2*d[23],
                'glottis.pulsation':p.glottis.pulsation+.5*d[24],
                'controls.film_gain':.5*g.tanh(),'controls.film_bias':.5*b.tanh()}
    def forward(self,phones,pos,f0,embedding,seed=0,body_override=None,average=False,return_controls=False):
        z=self.encode(embedding)
        if average:z=torch.zeros_like(z)
        body=self.body_head(z) if body_override is None else body_override
        film=self.film_head(z)
        if average:body=torch.zeros_like(body);film=torch.zeros_like(film)
        if f0.ndim==1:
            return functional_call(self.physics,self.parameters_for(body,film),(phones,pos,f0),
                                   dict(seed=seed,return_controls=return_controls))
        ys=[];ps=[]
        for i in range(len(f0)):
            result=functional_call(self.physics,self.parameters_for(body[i],film[i]),(phones[i],pos[i],f0[i]),
                                   dict(seed=seed+i,return_controls=return_controls))
            if return_controls:y,p=result;ps.append(p)
            else:y=result
            ys.append(y)
        return (torch.stack(ys),{k:torch.stack([p[k] for p in ps]) for k in ps[0]}) if return_controls else torch.stack(ys)
    def regularization(self,p):return self.physics.regularization(p)
    def project(self):self.physics.project()

def contrastive(model,a,b,speakers,temperature=.1):
    """Contrast learned anatomy representation of two disjoint real references.

    OpenVoice itself is frozen and is never a differentiable loss on synthesis.
    Multiple rows of one speaker are positives, never false negatives.
    """
    za=F.normalize(model.encode(a),dim=-1);zb=F.normalize(model.encode(b),dim=-1)
    logits=za@zb.T/temperature
    positive=torch.tensor([[x==y for y in speakers] for x in speakers],device=a.device)
    def term(x,m):return (torch.logsumexp(x,1)-torch.logsumexp(x.masked_fill(~m,-torch.inf),1)).mean()
    return .5*(term(logits,positive)+term(logits.T,positive.T))

def fit_body(model,phones,pos,f0,embedding,target,criterion,steps=200):
    """Exactly steps anatomy updates on REFERENCE audio; never on evaluation target."""
    flags=[p.requires_grad for p in model.parameters()]
    model.requires_grad_(False)
    with torch.no_grad():initial=model.body(embedding).detach()
    body=nn.Parameter(initial.clone());opt=torch.optim.Adam([body],lr=.015)
    history=[]
    try:
        for step in range(steps):
            opt.zero_grad(set_to_none=True)
            y=model(phones,pos,f0,embedding,body_override=body,seed=4321)
            loss,_=criterion(y,target)
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite body adaptation')
            loss.backward();opt.step();history.append(float(loss.detach()))
    finally:
        for p,flag in zip(model.parameters(),flags):p.requires_grad_(flag)
    return body.detach(),history
