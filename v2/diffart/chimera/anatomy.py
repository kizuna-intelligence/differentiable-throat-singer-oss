"""Differentiable midsagittal prototype; not a validated anatomical reconstruction.

Landmarks, body scales and area-law coefficients reference CHIMERA anatomy/human.rs,
MIT, Copyright (c) 2026 Jameson Hodge. See THIRD_PARTY_NOTICES.md.
The single semipolar centre is an explicit approximation documented in design issues.
"""
import math
import torch
from torch import nn
from .articulators import NEUTRAL
from .geom import contact_width, contact_area, conservative_resample

class HumanAnatomy(nn.Module):

    def __init__(self, sections=36, conservative=False, shape_controls=False):
        super().__init__()
        self.sections = sections
        self.conservative = conservative
        self.shape_controls = shape_controls
        self.area_calibration = nn.Parameter(torch.zeros(6, dtype=torch.float64))
        self.body_logits = nn.Parameter(torch.tensor([2.0, 0.0, -2.0], dtype=torch.float64))
        self.log_scale = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        self.pharynx_scale = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        self.register_buffer('bodies', torch.tensor([[1.0, 1.0, 2.25, 4.6], [0.9, 0.88, 2.2, 4.4], [0.74, 0.8, 2.15, 4.0]], dtype=torch.float64))
        self.register_buffer('neutral', torch.tensor(NEUTRAL, dtype=torch.float64))

    def forward(self, articulators, tongue_states=None):
        a = articulators
        body = self.body_logits.softmax(0) @ self.bodies
        scale = body[0] * self.log_scale.exp()
        pharynx = body[1] * self.pharynx_scale.exp()
        dense = max(128, 4 * self.sections)
        nph = dense // 2
        deformation = None
        if tongue_states is not None:
            axis = torch.linspace(0, 1, dense, device=a.device, dtype=a.dtype)
            basis = torch.sin(math.pi * axis).square() * torch.cos(torch.arange(tongue_states.shape[-1], device=a.device, dtype=a.dtype)[:, None] * math.pi * axis)
            deformation = 0.25 * (tongue_states @ basis)
        y = torch.linspace(0.15, 7.4, nph, device=a.device, dtype=a.dtype)
        y = y + a[..., 12, None] * (1 - y / 7.4)
        posterior = -1.05 - 0.035 * y
        radius = body[2]
        shape = a.new_ones(3)
        dy = y - a[..., 2, None]
        inside = radius.square() - (dy * shape[0]).square()
        tongue_back = a[..., 1, None] - shape[0] * inside.clamp_min(1e-06).sqrt()
        epi = 0.25 + 0.25 * (y / 7.4) + a[..., 6, None] * 0.5
        blend = ((y - 4.0) / 1.8).clamp(0, 1)
        root = epi * (1 - blend) + tongue_back * blend
        dph = root - posterior
        dph = dph * (1 - 0.6 * a[..., 13, None] * torch.exp(-((y - 1.0) / 0.8).square()))
        dph = dph * (1 - 0.5 * a[..., 14, None] * torch.exp(-((y - 0.5) / 0.3).square()))
        if deformation is not None:
            dph = dph + deformation[..., :nph]
        theta = torch.linspace(math.radians(140), math.radians(5), dense - nph, device=a.device, dtype=a.dtype)
        direction = torch.stack((theta.cos(), theta.sin()), -1)
        center = a.new_tensor([2.6, 6.95])
        tongue_center = torch.stack((a[..., 1], a[..., 2] - 0.25 * a[..., 0]), -1)
        delta = tongue_center - center
        proj = (delta[..., None, :] * direction).sum(-1)
        disc = radius.square() - delta.square().sum(-1, keepdim=True) + proj.square()
        inner_radius = proj + disc.clamp_min(1e-06).sqrt()
        roof = 1 / torch.sqrt((theta.cos() / (4.6 * shape[1])).square() + (theta.sin() / (3.35 * shape[2])).square())
        anterior = ((theta.cos() - 0.5) / 0.5).clamp(0, 1).pow(4)
        tip_lift = (a[..., 4, None] - 7.3 + 0.4 * a[..., 5, None]) * torch.exp(-((theta - 0.38) / 0.2).square())
        inner_radius = inner_radius + tip_lift * (a[..., 3, None] / 6.45)
        dor = a[..., 16, None] * 0.6 * torch.exp(-((theta - 0.8) / 0.4).square())
        oral_width = roof - inner_radius + dor
        lip_width = a[..., 7, None] * (1 - 0.85 * a[..., 10, None])
        oral_width = oral_width * (1 - anterior) + lip_width * anterior
        oral_width = oral_width + 0.25 * a[..., 0, None] * (1 - anterior)
        if deformation is not None:
            oral_width = oral_width + deformation[..., nph:]
        d = torch.cat((dph, oral_width), -1) * scale
        alpha = torch.cat((a.new_full((nph,), 2.0), a.new_full((dense - nph,), 2.6)))
        areas = contact_area(d, alpha, self.training)
        lateral = a[..., 15, None] * 0.25 * torch.exp(-((torch.linspace(0, 1, dense, device=a.device) - 0.8) / 0.12).square())
        areas = areas + lateral * scale.square()
        lips = math.pi / 4 * contact_width(a[..., 7], self.training) * body[3] * (0.45 + 0.55 * a[..., 9]) * scale.square()
        areas = torch.cat((areas[..., :-1], lips[..., None]), -1)
        axis = (torch.arange(dense, device=a.device, dtype=a.dtype) + 0.5) / dense
        basis = torch.cos(torch.arange(6, device=a.device, dtype=a.dtype)[:, None] * math.pi * axis)
        log_width = 1.6 * torch.tanh(self.area_calibration @ basis / 1.6)
        areas = areas * log_width.exp()
        length = (8.7 * pharynx - a[..., 12] + (8.3 + a[..., 8])) * scale
        if self.conservative:
            areas, minimum_areas = conservative_resample(areas, self.sections, self.training)
        else:
            areas = torch.nn.functional.interpolate(areas.reshape(-1, 1, dense), size=self.sections, mode='linear', align_corners=False).reshape(*a.shape[:-1], self.sections)
        dx = (length / self.sections)[..., None].expand_as(areas)
        ph_outer = torch.stack((posterior.expand_as(y), y), -1)
        ph_inner = torch.stack((posterior + contact_width(dph, False), y), -1)
        roof_points = center + direction * roof[..., None]
        tongue_points = center + direction * (roof - contact_width(oral_width, False))[..., None]
        outer = torch.cat((ph_outer, roof_points.expand(*a.shape[:-1], -1, 2)), -2) * scale
        inner = torch.cat((ph_inner, tongue_points), -2) * scale
        result = dict(areas=areas, lengths=dx, outer=outer, inner=inner, velum=contact_width(a[..., 11], self.training) * scale.square(), nasal_area=2.0 * scale.square(), fossa_area=0.9 * scale.square(), sinus_volume=14.0 * scale.pow(3), scale=scale)
        if self.conservative:
            result['minimum_areas'] = minimum_areas
            result['distances'] = torch.cat((d[..., :-1], (a[..., 7] * scale)[..., None]), -1)
        return result
