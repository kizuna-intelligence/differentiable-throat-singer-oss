"""Independent convergence and physical-coordinate checks for extended C."""
import unittest

import numpy as np

from .finite_volume import solve
from .multidomain import Piece,TubePINN
from .run_extended import relative_field,slab_reference,modal_ratio
from .smooth_reference import solve as fem_solve
from .uniform import plane_transfer


class ExtendedChecks(unittest.TestCase):
    def test_fv_second_order_and_conservative_flux(self):
        pieces=[Piece(0,.17,0,.015)]
        errors=[]
        for spacing in (.001,.0005,.00025):
            ref=solve(pieces,4500,spacing=spacing,axisymmetric=True)
            errors.append(abs(ref.transfer/plane_transfer(4500,.17,.15)-1))
            mass_transfer=1-.17/(.015**2/2)*(2*np.pi*4500/350)**2*np.dot(ref.volumes,ref.pressure)
            np.testing.assert_allclose(mass_transfer,ref.transfer,atol=1e-9)
        self.assertLess(errors[1],errors[0]/3.5)
        self.assertLess(errors[2],errors[1]/3.5)

    def test_step_weak_convergence_and_independent_reference(self):
        pieces=[Piece(0,.08,0,.01),Piece(.08,.17,0,.015)]
        ref=solve(pieces,4500,spacing=.00025,axisymmetric=True)
        errors=[]
        for nx,ny in ((16,6),(32,14)):
            model=TubePINN(pieces,axisymmetric=True,nx=nx,ny=ny).train_variational(4500)
            errors.append(relative_field(model,ref))
            r=np.linspace(0,.01,101)
            np.testing.assert_allclose(model.pressure(0,.08,r),model.pressure(1,.08,r),atol=1e-9)
        self.assertLess(errors[-1],.005)
        self.assertLess(errors[-1],errors[0]*.6)

    def test_cartesian_odd_mode_and_uniform_control(self):
        p=Piece(0,.17,0,.03)
        drive=lambda y:1+.5*np.cos(np.pi*y/p.width)
        model=TubePINN([p],nx=48,ny=10).train(7000,drive=drive)
        rng=np.random.default_rng(334)
        x,y=rng.random((2,300));x*=.17;y*=.03
        truth=slab_reference(x,y,7000,p,.15)
        self.assertLess(np.linalg.norm(model.pressure(0,x,y)-truth)/np.linalg.norm(truth),1e-6)
        # rho² features would exclude this mode: opposite pressures at two walls.
        self.assertGreater(abs(model.pressure(0,.03,0)-model.pressure(0,.03,.03)),.001)
        control=TubePINN([p],nx=40,ny=8).train(7000)
        self.assertLess(modal_ratio(control),1e-8)

    def test_curved_metric_against_independent_fv(self):
        p=Piece(0,.17,-.015,.015)
        model=TubePINN([p],curvature=6,nx=40,ny=10).train(4500)
        ref=solve([p],4500,spacing=.00025,curvature=6)
        self.assertLess(relative_field(model,ref),.004)
        self.assertLess(abs(model.transfer()/ref.transfer-1),.004)
        # Midline reflection reverses the bend without changing mean transfer.
        reverse=TubePINN([p],curvature=-6,nx=40,ny=10).train(4500)
        np.testing.assert_allclose(model.transfer(),reverse.transfer(),rtol=1e-8,atol=1e-8)

    def test_smooth_fem_independent_uniform_convergence(self):
        errors=[]
        for nx,nr in ((80,8),(160,16),(320,24)):
            ref=fem_solve(lambda x:np.full_like(x,.015),.17,4500,.15,nx,nr)
            errors.append(abs(ref.transfer/plane_transfer(4500,.17,.15)-1))
        self.assertLess(errors[1],errors[0]/3.5)
        self.assertLess(errors[2],errors[1]/3.5)

    def test_non_aligned_step_is_rejected(self):
        with self.assertRaises(ValueError):
            solve([Piece(0,.0812,0,.01),Piece(.0812,.17,0,.015)],4500,spacing=.0005)


if __name__=='__main__':
    unittest.main()
