"""Vowel substrate with consonant overrides on the corpus's exact LAB grid.
Reference: CHIMERA control/score.rs and phonetics/planner.rs, MIT,
Copyright (c) 2026 Jameson Hodge. See THIRD_PARTY_NOTICES.md.
"""
import torch
from torch import nn
from .japanese_inventory import targets,VOWELS,VOICELESS
from .articulators import RANGES


class GesturalScore(nn.Module):
    def __init__(self,vocab):
        super().__init__();self.vocab=dict(vocab)
        values,mask=targets(vocab)
        self.targets=nn.Parameter(values);self.strength_raw=nn.Parameter(torch.zeros_like(values))
        self.register_buffer('dominant',mask)
        self.register_buffer('vowels',torch.tensor([p in VOWELS for p,i in sorted(vocab.items(),key=lambda kv:kv[1])]))
        self.register_buffer('voiced',torch.tensor([p not in VOICELESS for p,i in sorted(vocab.items(),key=lambda kv:kv[1])]))
        self.register_buffer('ranges',torch.tensor(RANGES,dtype=torch.float64))
        self.embedding=nn.Embedding(len(vocab),16,dtype=torch.float64)
        self.correction=nn.Sequential(nn.Linear(51,48,dtype=torch.float64),nn.Tanh(),
                                      nn.Linear(48,34,dtype=torch.float64))
        nn.init.zeros_(self.correction[-1].weight);nn.init.zeros_(self.correction[-1].bias)

    def forward(self,phones,pos):
        left,current,right=phones.unbind(-1)
        # Immediate phone context is the same model input as the baseline.
        lt,rt=self.targets[left],self.targets[right]
        lv,rv=self.vowels[left],self.vowels[right]
        blend=pos[...,None]
        base=torch.where((lv&rv)[...,None],lt*(1-blend)+rt*blend,
                         torch.where(lv[...,None],lt,torch.where(rv[...,None],rt,self.targets[current])))
        selected=current
        if 'cl' in self.vocab:
            selected=torch.where(current==self.vocab['cl'],right,current)
        dominant=self.dominant[selected]
        target=torch.where(self.vowels[current][...,None],self.targets[current],
                           base*(1-dominant)+self.targets[selected]*dominant)
        feat=torch.cat((self.embedding(phones).flatten(-2),pos[...,None],
                        torch.sin(torch.pi*pos)[...,None],torch.cos(torch.pi*pos)[...,None]),-1)
        delta,strength=self.correction(feat).chunk(2,-1)
        # Zero corrections initially. Range-scaled, bounded physical controls.
        target=target+.05*(self.ranges[:,1]-self.ranges[:,0])*delta.tanh()
        gain=1+.2*torch.tanh(self.strength_raw[current]+strength)
        return target,gain,self.voiced[current]
