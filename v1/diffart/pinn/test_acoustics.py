"""Physical invariants and independent reference checks for experiment C.

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest diffart.pinn.test_acoustics
"""
import unittest

import numpy as np
from scipy import special
import torch

from .field import helmholtz_residual, radiation_residual, wall_residual
from .uniform import UniformPINN, modal_reference, piston_impedance, plane_transfer


class PhysicsChecks(unittest.TestCase):
    def test_axis_is_three_dimensional_laplacian(self):
        # Laplacian(x²+rho²)=6 in 3D, including rho=0. Cartesian 2D gives 4.
        x = torch.tensor([[.01], [.12]], dtype=torch.float64, requires_grad=True)
        r = torch.tensor([[0.], [.005]], dtype=torch.float64, requires_grad=True)
        v = x**2 + r**2
        p = torch.cat((v, 2*v), -1)
        actual = helmholtz_residual(p, x, r, torch.zeros_like(x))
        torch.testing.assert_close(actual, torch.tensor([[6., 12.], [6., 12.]], dtype=torch.float64))

    def test_rigid_sloping_wall_normal(self):
        class Geometry:
            def radius(self, x, coefficients):
                return .01 + .1*x
        x = torch.tensor([[.01], [.12]], dtype=torch.float64, requires_grad=True)
        r = torch.tensor([[.011], [.022]], dtype=torch.float64, requires_grad=True)
        v = x + .1*r
        p = torch.cat((v, -v), -1)
        torch.testing.assert_close(wall_residual(p, x, r, None, Geometry()), torch.zeros_like(p))

    def test_radiation_sign_for_outgoing_wave(self):
        self.assertEqual(piston_impedance(0., .01), 0j)
        impedance = piston_impedance(np.linspace(0, 8000, 101), .01)
        self.assertTrue(np.all(impedance.real >= 0))
        x = torch.tensor([[.17], [.17]], dtype=torch.float64, requires_grad=True)
        f = torch.tensor([[500.], [7000.]], dtype=torch.float64)
        phase = 2*torch.pi*f*x/350
        p = torch.cat((torch.cos(phase), -torch.sin(phase)), -1)
        result = radiation_residual(p, x, f, torch.ones(2, dtype=torch.complex128))
        torch.testing.assert_close(result, torch.zeros_like(p), atol=1e-13, rtol=0)

    def test_volume_velocity_normalization_and_dc(self):
        for frequency in (0., 500., 8000.):
            with self.subTest(frequency=frequency):
                m = UniformPINN(axial_degree=40, radial_degree=3, interior_points=700).train(
                    frequency, z=.03 if frequency > 0 else 0.)
                np.testing.assert_allclose(m.axial_derivative(0., .25), 1., atol=1e-8)
                np.testing.assert_allclose(m.axial_derivative(1., .25),
                                           plane_transfer(frequency, .17, .03), atol=1e-8)

    def test_unseen_random_points_transverse_and_resolution(self):
        alpha = special.jn_zeros(1, 1)[0]
        drive = lambda s: 1 + .5*special.j0(alpha*np.sqrt(s))
        rng = np.random.default_rng(38)
        t, s = rng.random((2, 300))
        errors = []
        for nx, nr, n in ((24, 4, 800), (48, 8, 1800)):
            m = UniformPINN(radius=.04, axial_degree=nx, radial_degree=nr,
                            interior_points=n).train(7000., .15, drive)
            reference = modal_reference(t, s, 7000, .17, .04, .15, (1., .5))
            errors.append(np.linalg.norm(m.pressure(t, s)-reference)/np.linalg.norm(reference))
        self.assertLess(errors[-1], 1e-6)
        self.assertLess(errors[-1], errors[0]/1000)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
