"""Passive 3D junction correction: extra series acoustic inertance.

Uses the existing independent axisymmetric FEM assembler in diffart.pinn,
not the noncausal whole-tract MLP ratio. A nonnegative inertance stores
E=0.5*L*U^2 and cannot create energy. Calibration uses only one low-frequency
resonance; higher resonances and a second area ratio are held out.
This initial sharp-junction correction does not model all transverse modes.
"""
import math
import numpy as np
import torch
from .constants import RHO, C_SOUND

def add_correction(net, coefficient=1.0):
    """Static graph edges only; differentiable dynamic-geometry version must
    recompute this term from current areas rather than detaching their values.
    """
    a = net.areas[net.edge_a]
    b = net.areas[net.edge_b]
    small = torch.minimum(a, b).clamp_min(net.closure_area)
    large = torch.maximum(a, b).clamp_min(net.closure_area)
    radius = (small / math.pi).sqrt()
    correction = coefficient * RHO * radius / small * (1 - (small / large).sqrt()).square()
    nc = getattr(net, 'nasal_correction_coefficient', None)
    if nc is not None and 'nasal' in net.tubes:
        ix = net.tubes['nasal']
        lo, hi = (min(ix), max(ix))
        nasal = (net.edge_a >= lo) & (net.edge_a <= hi) & (net.edge_b >= lo) & (net.edge_b <= hi)
        correction = torch.where(nasal, nc * RHO * radius / small * (1 - (small / large).sqrt()).square(), correction)
    correction = correction * ~net.is_port * ~net.fixed_mask[net.edge_a] * ~net.fixed_mask[net.edge_b]
    net.extra_inertance.copy_(correction)
    return net
