"""Physical checks for the short stage-(a) experiment, independent of training."""
import unittest

import numpy as np
import torch

from .finite_volume import solve as fv_solve
from .multidomain import Piece
from .quick_a import Field, LENGTH, derivative, fem
from .uniform import plane_transfer


class QuickChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        torch.set_default_dtype(torch.float64)

    def test_sharp_fem_against_independent_finite_volume(self):
        pieces = [Piece(0, .08, 0, .01), Piece(.08, .17, 0, .015)]
        for frequency in (1000., 4000.):
            ref = fem(frequency, .0005)
            independent = fv_solve(pieces, frequency, spacing=.0005, axisymmetric=True)
            self.assertLess(abs(ref['transfer']/independent.transfer-1), .005)
            self.assertLess(ref['residual'], 1e-9)
            # The shoulder is an actual vertical boundary, not a smooth taper.
            self.assertEqual(np.count_nonzero(np.isclose(ref['nodes'][:, 0], .08)), 31)
            coarse = fem(frequency, .001)
            self.assertLess(abs(coarse['transfer']/ref['transfer']-1), .01)

    def test_fem_sign_and_source_normalization_against_uniform_theory(self):
        for frequency in (1000., 4000.):
            ref = fem(frequency, .0005, uniform=True)
            self.assertLess(abs(ref['transfer']/plane_transfer(frequency, LENGTH, .15)-1), .0003)

    def test_exact_inlet_for_all_hard_arms_and_even_radial_field(self):
        for frequency in (1000., 4000.):
            for arm in (2, 3, 4):
                torch.manual_seed(19)
                model = Field(frequency, arm).double()
                t = torch.zeros(27, 1, requires_grad=True)
                rho = torch.linspace(0, .01, 27)[:, None].requires_grad_()
                p = model(0, t, (rho/.01)**2)
                px = derivative(p, t)*LENGTH/.08
                torch.testing.assert_close(px, torch.tensor([1., 0.]).expand_as(px), rtol=0, atol=1e-12)
                pr = derivative(p, rho)
                torch.testing.assert_close(pr[0], torch.zeros(2), rtol=0, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
