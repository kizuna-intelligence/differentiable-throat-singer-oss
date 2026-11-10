"""Numerical invariants for the C data and synthesis path."""
import unittest
import numpy as np
import torch
from diffart.physics import Synthesizer,transfer
from diffart.improved import ImprovedSynthesizer
from .correction_data import reference,plane
from .correction import FEMCorrection


class CorrectionTests(unittest.TestCase):
    def test_uniform_and_abcd(self):
        freq=np.array([50.,300.,900.]);areas=np.ones(44)
        ratio=reference(areas,.154,3.6,freq)/plane(areas,.154,3.6,freq)
        self.assertLess(float(abs(np.log(ratio)).max()),.001)
        area=np.linspace(.3,3,44)
        truth=transfer(torch.tensor(area),torch.tensor(.154,dtype=torch.float64),
                       torch.tensor(3.6,dtype=torch.float64),torch.tensor(freq)).numpy()
        np.testing.assert_allclose(plane(area,.154,3.6,freq),truth,rtol=1e-8)

    def test_refined_step(self):
        freq=np.array([500.,1000.,4000.]);area=np.r_[np.ones(22),np.full(22,2.)]
        coarse=reference(area,.154,3.6,freq,3,6)
        fine=reference(area,.154,3.6,freq,6,12)
        self.assertLess(float(np.linalg.norm(coarse-fine)/np.linalg.norm(fine)),.02)

    def test_grid_fade_gradient(self):
        meta=dict(mean=np.zeros(44),basis=np.eye(8,44),coeff_mean=np.zeros(8),coeff_std=np.ones(8),
                  target_mean=[.2,.3],target_std=[1,1],frequencies=np.linspace(50,8000,80))
        model=FEMCorrection(meta).requires_grad_(False)
        for p in model.mlp.parameters():p.data.zero_()
        area=torch.ones(2,44,requires_grad=True)
        result=model(area,torch.tensor([0.,8000.,9000.,10000.,12000.]))
        self.assertTrue(torch.equal(result[:,-2:],torch.ones_like(result[:,-2:])))
        self.assertAlmostEqual(float(abs(result[0,0]).detach()),np.exp(.2),places=6)
        result.abs().sum().backward();self.assertTrue(torch.isfinite(area.grad).all())

    def test_unity_matches_legacy(self):
        class Unity(torch.nn.Module):
            def forward(self,area,freq):return torch.ones(*area.shape[:-1],len(freq),device=area.device)
        torch.set_num_threads(2);torch.manual_seed(12)
        original=Synthesizer(38)
        improved=ImprovedSynthesizer(38,pinn_correction=Unity())
        improved.load_legacy(original.state_dict())
        phone=torch.randint(0,38,(12,3));pos=torch.linspace(0,1,12);f0=torch.full((12,),220.)
        with torch.no_grad():a=original(phone,pos,f0);b=improved(phone,pos,f0)
        torch.testing.assert_close(a,b,rtol=0,atol=0)


if __name__=='__main__':unittest.main()
