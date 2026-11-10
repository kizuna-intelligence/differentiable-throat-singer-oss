"""Piecewise smooth ray/polyline intersections for midsagittal geometry.
Reference: CHIMERA anatomy/geom.rs, MIT, Copyright (c) 2026 Jameson Hodge.
"""
import torch

def cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

def ray_polyline(origin, direction, contour):
    """Nearest forward hit, with derivatives through the selected segment.

    origin,direction [...,R,2]; contour [...,P,2]. Returns [...,R].
    Invalid rays are reported, never silently assigned an arbitrary width.
    """
    a = contour[..., None, :-1, :]
    edge = contour[..., None, 1:, :] - a
    d = direction[..., :, None, :]
    offset = a - origin[..., :, None, :]
    denominator = cross(d, edge)
    safe = torch.where(denominator.abs() > 1e-10, denominator, torch.ones_like(denominator))
    distance = cross(offset, edge) / safe
    fraction = cross(offset, d) / safe
    valid = (denominator.abs() > 1e-10) & (distance >= 0) & (fraction >= -1e-08) & (fraction <= 1 + 1e-08)
    hits = torch.where(valid, distance, torch.full_like(distance, torch.inf)).amin(-1)
    if not torch.isfinite(hits).all():
        raise ValueError('A semipolar ray missed its anatomical contour')
    return hits

def contact_width(distance, training, width=0.01):
    exact = distance.clamp_min(0)
    return exact

def contact_area(distance, alpha, training, beta=1.4, width=0.01):
    """Exact contact in forward; area-level smooth surrogate in backward.

    Smoothing distance alone still gives zero gradient after d**beta at d=0.
    Apply the surrogate to the complete area law so a contacted tongue can move.
    """
    exact = alpha * distance.clamp_min(0).pow(beta)
    return exact
_RESAMPLE_INDEX = {}

def conservative_resample(areas, sections, training, sub=8, closure=1e-05):
    """Harmonic section areas plus point minima (CHIMERA anatomy/human.rs).

    The original dense grid points participate in closure detection, including
    the final lip slit. A narrow contact cannot disappear between sample centres.
    Exact evaluation/forward closure, smooth area-level training surrogate.
    """
    from torch.nn import functional as F
    dense = areas.shape[-1]
    shape = areas.shape[:-1]
    sampled = F.interpolate(areas.reshape(-1, 1, dense), size=sections * sub, mode='linear', align_corners=False).reshape(*shape, sections, sub)
    key = (dense, sections, str(areas.device))
    if key not in _RESAMPLE_INDEX:
        bins = [min(sections - 1, int((j + 0.5) * sections / dense)) for j in range(dense)]
        groups = [[j for j, b in enumerate(bins) if b == i] for i in range(sections)]
        width = max(map(len, groups))
        groups = [g + [g[-1]] * (width - len(g)) for g in groups]
        _RESAMPLE_INDEX[key] = torch.tensor(groups, device=areas.device, dtype=torch.long)
    minimum = areas[..., _RESAMPLE_INDEX[key]].amin(-1)
    minimum = torch.minimum(minimum, sampled.amin(-1))
    harmonic = sampled.clamp_min(1e-12).reciprocal().mean(-1).reciprocal()
    exact = torch.where(minimum > closure, harmonic, torch.zeros_like(harmonic))
    effective_min = torch.where(minimum > closure, torch.minimum(minimum, harmonic), torch.zeros_like(minimum))
    return (exact, effective_min)
