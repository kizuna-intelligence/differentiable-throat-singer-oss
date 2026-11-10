"""Finite differences of the physical tube and articulated geometry."""
import unittest
import torch
from diffart.chimera.frequency import transfer
from diffart.chimera.anatomy import HumanAnatomy
from diffart.chimera.articulators import NEUTRAL
class Gradients(unittest.TestCase):
    def test_tube_area_gradient(self):
        areas=torch.tensor([2.,4.,3.],dtype=torch.float64,requires_grad=True)
        dx=torch.tensor([5.,7.,5.],dtype=torch.float64)
        def objective(a):return transfer(a,dx,[350.,1050.],attenuation=.002).abs().square().sum()
        value=objective(areas);value.backward();delta=1e-5
        finite=[]
        for i in range(3):
            plus=areas.detach().clone();minus=plus.clone();plus[i]+=delta;minus[i]-=delta
            finite.append((objective(plus)-objective(minus))/(2*delta))
        torch.testing.assert_close(areas.grad,torch.stack(finite),rtol=1e-5,atol=1e-7)
    def test_anatomy_gradient(self):
        model=HumanAnatomy(sections=30,conservative=True).double().eval()
        states=torch.tensor(NEUTRAL,dtype=torch.float64)[None,None].requires_grad_()
        def objective(x):return model(x)['areas'].square().sum()
        value=objective(states);value.backward();j=1;delta=1e-5
        plus=states.detach().clone();minus=plus.clone();plus[0,0,j]+=delta;minus[0,0,j]-=delta
        expected=(objective(plus)-objective(minus))/(2*delta)
        torch.testing.assert_close(states.grad[0,0,j],expected,rtol=1e-4,atol=1e-5)
