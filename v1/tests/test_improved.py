"""B1/B2 invariants and regression against the real, unchanged best.pt."""
import unittest
import json
import torch
from diffart.data import ROOT, sha
from diffart.physics import Synthesizer, lf_flow, C
from diffart.improved import ImprovedSynthesizer, Piriform, LearnableGlottis

torch.set_num_threads(2)

class ImprovedPhysics(unittest.TestCase):
    def test_real_checkpoint_exact_initial_output(self):
        vocab=json.loads((ROOT/'models/vocab.json').read_text())
        state=torch.load(ROOT/'models/throat_ritsu_v1.pt',map_location='cpu',weights_only=True)['model']
        legacy=Synthesizer(len(vocab));legacy.load_state_dict(state)
        data=(torch.full((1,35,3),vocab['a'],dtype=torch.long),torch.linspace(0,1,35)[None],torch.full((1,35),220.))
        for b1,b2 in [(False,False),(True,False),(False,True),(True,True)]:
            m = ImprovedSynthesizer(len(vocab), b1=b1, b2=b2); m.load_legacy(state)
            for rd in [None,.5,2.5]:
                with torch.no_grad():
                    a=legacy(*data[:3],seed=975,rd_override=rd,phase_offset=12.37)
                    b=m(*data[:3],seed=975,rd_override=rd,phase_offset=12.37)
                self.assertTrue(torch.equal(a,b), (b1,b2,rd,float((a-b).abs().max())))

    def test_frozen_fast_path_matches_physical_path(self):
        torch.manual_seed(71)
        full=ImprovedSynthesizer(12,b1=True,b2=True)
        fast=ImprovedSynthesizer(12,b1=True,b2=True)
        with torch.no_grad():full.piriform.coupling.fill_(.27)
        fast.load_state_dict(full.state_dict())
        for name,p in fast.named_parameters():p.requires_grad_(name.startswith(('piriform.','glottis.')))
        phones=torch.randint(0,12,(2,17,3));pos=torch.rand(2,17);f0=torch.full((2,17),190.)
        a=full(phones,pos,f0,seed=4);b=fast(phones,pos,f0,seed=4)
        torch.testing.assert_close(a,b,atol=2e-7,rtol=2e-4)
        a.square().mean().backward();b.square().mean().backward()
        for name,p in fast.named_parameters():
            if p.requires_grad:
                torch.testing.assert_close(p.grad,dict(full.named_parameters())[name].grad,atol=1e-8,rtol=3e-3)

    def test_frozen_checkpoint_exact(self):
        vocab=json.loads((ROOT/'models/vocab.json').read_text())
        state=torch.load(ROOT/'models/throat_ritsu_v1.pt',map_location='cpu',weights_only=True)['model']
        legacy=Synthesizer(len(vocab));legacy.load_state_dict(state)
        m=ImprovedSynthesizer(len(vocab),b1=True,b2=True);m.load_legacy(state)
        for n,p in m.named_parameters():p.requires_grad_(n.startswith(('piriform.','glottis.')))
        d=(torch.full((1,30,3),vocab['a'],dtype=torch.long),torch.linspace(0,1,30)[None],torch.full((1,30),220.))
        self.assertTrue(torch.equal(legacy(*d[:3]),m(*d[:3])))

    def test_branch_passivity_and_quarter_wave(self):
        p=Piriform().double()
        with torch.no_grad(): p.coupling.fill_(1)
        f=torch.linspace(0,12000,48001,dtype=torch.float64)
        y=p.admittance(f,torch.tensor(.8,dtype=torch.float64))
        self.assertGreaterEqual(float(y.real.min().detach()),0)
        expected=C/(4*float(p.length[0].detach()))
        actual=float(f[y[0].abs().argmax()])
        self.assertLess(abs(actual-expected),.3)
        self.assertTrue(torch.all((p.position>=.02)&(p.position<=.03)))

    def test_flow_constraints_and_gradients(self):
        g=LearnableGlottis().double()
        with torch.no_grad():
            g.closed.fill_(.15);g.shape.copy_(torch.linspace(-.7,.7,8));g.shape_rd.fill_(.1)
            g.pulsation.fill_(.5);g.noise_warmup.fill_(1)
        phase=torch.linspace(0,1,2001,dtype=torch.float64)[None]
        rd=torch.linspace(.3,2.7,9,dtype=torch.float64)[:,None]
        q=g.flow(phase,rd)
        self.assertGreater(float(q.min().detach()),-1e-6)
        self.assertTrue(torch.equal(q[:,phase[0]>=.85],torch.zeros_like(q[:,phase[0]>=.85])))
        self.assertLess(float(torch.diff(q,dim=-1).abs().max().detach())*2000,100)
        q.square().mean().backward()
        for p in [g.shape,g.shape_rd,g.closed]:
            self.assertTrue(torch.isfinite(p.grad).all());self.assertGreater(float(p.grad.norm()),0)
        noise=g.aspiration(torch.zeros_like(q),q,torch.ones_like(q,dtype=torch.bool))
        self.assertGreaterEqual(float(noise.min().detach()),.05*.15)

    def test_connected_model_gradients_and_f0_isolation(self):
        torch.manual_seed(17)
        m=ImprovedSynthesizer(12,b1=True,b2=True)
        with torch.no_grad():
            m.piriform.coupling.fill_(.15);m.glottis.closed.fill_(.1);m.glottis.pulsation.fill_(.3)
        phones=torch.randint(0,12,(1,15,3));pos=torch.rand(1,15);f0=torch.full((1,15),180.)
        p=m.controls(phones,pos,f0,m.speaker);q=m.controls(phones,pos,f0*2,m.speaker)
        for k in ['art','velum','breath','turbulence','location']: self.assertTrue(torch.equal(p[k],q[k]))
        y,p=m(phones,pos,f0,return_controls=True)
        (y.square().mean()+m.regularization(p)).backward()
        for name,p in m.named_parameters():
            self.assertIsNotNone(p.grad,name);self.assertTrue(torch.isfinite(p.grad).all(),name)
            self.assertGreater(float(p.grad.norm()),0,name)

if __name__=='__main__': unittest.main(verbosity=2)
