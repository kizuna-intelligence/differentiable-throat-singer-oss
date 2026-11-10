"""Inference coefficient packing from the CHIMERA port; MIT, Jameson Hodge (2026)."""
import torch
import numpy as np
from .constants import RHO, MU, RHO_C2

def pack_coefficients(net, areas=None, dx=None, port_area=None, wall_scale=1.0, minimum_areas=None):
    """Control-rate [B,F,32,D] coefficients; gradients stay in Torch."""
    areas = net.areas[None, None] if areas is None else areas
    if areas.ndim == 1:
        areas = areas[None, None]
    elif areas.ndim == 2:
        areas = areas[:, None]
    dx = net.dx if dx is None else dx
    el = net.elements(areas, dx, port_area, minimum_areas=minimum_areas)
    bg, ng = net.relaxation_elements(el, dx=dx if getattr(net, 'consistent_loss_geometry', True) else None)
    l, r, kf, kr, opened, comp, wa, eff = el
    if hasattr(net, 'correction_coefficient'):
        a = areas[..., net.edge_a].clamp_min(net.closure_area)
        z = areas[..., net.edge_b].clamp_min(net.closure_area)
        small = torch.minimum(a, z)
        large = torch.maximum(a, z)
        correction = net.correction_coefficient * RHO * (small / np.pi).sqrt() / small * (1 - (small / large).sqrt()).square()
        nc = getattr(net, 'nasal_correction_coefficient', None)
        if nc is not None and 'nasal' in net.tubes:
            ix = net.tubes['nasal']
            lo, hi = (min(ix), max(ix))
            nasal = (net.edge_a >= lo) & (net.edge_a <= hi) & (net.edge_b >= lo) & (net.edge_b <= hi)
            local = nc * RHO * (small / np.pi).sqrt() / small * (1 - (small / large).sqrt()).square()
            correction = torch.where(nasal, local, correction)
        mask = ~net.is_port * ~net.fixed_mask[net.edge_a] * ~net.fixed_mask[net.edge_b]
        l = l - net.extra_inertance + correction * mask
    b, f = areas.shape[:2]
    n = len(net.areas)
    e = len(net.edge_a)
    width = 1 << (max(n, e, len(net.noise_sources)) - 1).bit_length()

    def pad(x, size, default=0.0):
        x = torch.as_tensor(x, device=areas.device, dtype=areas.dtype)
        x = x.expand(b, f, size)
        return torch.nn.functional.pad(x, (0, width - size), value=default)
    cols = [pad(l, e, 1.0), pad(r, e), pad(kf, e), pad(kr, e), pad(opened, e), pad(comp, n, 1.0), pad(wa, n), pad(eff, e, 1.0)]
    m, d, k = net.walls.unbind(-1)
    cols += [pad(m * wall_scale, n), pad(d, n), pad(k, n), pad(net.rad_mask, n), pad(net.fixed_mask, n, 1.0), pad(net.fixed_pressure, n), pad(net.wall_radiation, n), pad(dx, n, 1.0)]
    cols += [pad(bg[..., j], e) for j in range(8)] + [pad(ng[..., j], n) for j in range(8)]
    return torch.stack(cols, 2).contiguous()

def fixed_parameters(net, width, coeff):
    device, dtype = (coeff.device, coeff.dtype)
    h = torch.zeros(10, width, device=device, dtype=dtype)
    if hasattr(net, 'glottis_edge'):
        j = net.glottis_edge
        a = int(net.edge_a[j])
        b = int(net.edge_b[j])
        h[0, j] = 1.0
        h[1, j] = RHO * net.port_len[j]
        h[2, j] = 12 * MU * net.port_len[j] * net.port_slit[j] ** 2 if net.viscous else 0.0
        if net.nonlinear:
            h[3, j] = 0.5 * RHO * net.port_entrance[j]
            h[4, j] = 0.5 * RHO * net.port_exit[j]
        h[5, j] = 1 / net.areas[a]
        h[6, j] = 1 / net.areas[b]
    h[9, :len(net.edge_a)] = 1.0
    for i, v in enumerate(net.noise_sources):
        h[7, i] = v[3]
        h[8, i] = 1.0
    return h

def initial_state(net, batch, width, coefficients=None):
    nst = 25
    out = net.areas.new_zeros(batch, nst, width)
    n = len(net.areas)
    out[:, 0, :n] = net.fixed_pressure
    out[:, 2] = 1.0
    if coefficients is not None:
        out[:, 0] = coefficients[:, 0, 13]
        out[:, 2] = coefficients[:, 0, 0]
        out[:, -1] = coefficients[:, 0, 5] * RHO_C2
    else:
        out[:, -1, :n] = net.areas * net.dx
    return out
