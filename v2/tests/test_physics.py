"""Independent quarter-wave tube reference, CGS units."""
import unittest
import numpy as np
from diffart.chimera.frequency import resonances
from diffart.chimera.constants import C_SOUND
class Physics(unittest.TestCase):
    def test_closed_open_uniform_tube(self):
        length=17.;areas=np.full(8,3.);dx=np.full(8,length/8)
        actual=resonances(areas,dx,maximum=5000)
        expected=np.arange(1,10,2)*C_SOUND/(4*length)
        np.testing.assert_allclose(actual,expected[expected<5000],atol=1e-5,rtol=1e-8)
