"""Parametric neural field and physical-coordinate residuals for smooth tubes.

Convention exp(+i omega t). Complex pressure is stored as two real channels.
Inputs are x/L, (rho/r(x))², f/8000, and standardized log-area PCA coefficients.
Geometry derivatives enter through autograd in physical x,rho coordinates.
This general geometry model is experimental; see uniform.py for validated cases.
"""
import math

import torch
from torch import nn


C = 350.0  # fixed physical constant; not read from the trained throat


class AreaGeometry(nn.Module):
    """Smooth cosine fit to log-area PCA at the 44 section centers.

    Smoothing intentionally defines a new smooth geometry; sharp step validation
    needs interface conditions and is NOT supplied by this representation.
    """
    def __init__(self, mean, basis, length, modes=16):
        super().__init__()
        n = len(mean)
        t = (torch.arange(n, dtype=torch.float64) + .5) / n
        harmonics = torch.arange(min(modes, n), dtype=torch.float64)
        design = torch.cos(math.pi * t[:, None] * harmonics)
        profiles = torch.cat((torch.as_tensor(mean)[None], torch.as_tensor(basis)), 0).double()
        fit = torch.linalg.lstsq(design, profiles.T).solution.T
        self.register_buffer('log_area_cosine', fit)
        self.register_buffer('harmonics', harmonics)
        self.length = float(length)

    def radius(self, x, coefficients):
        weights = self.log_area_cosine[0] + coefficients @ self.log_area_cosine[1:]
        log_area = (weights * torch.cos(math.pi * x / self.length * self.harmonics)).sum(-1, keepdim=True)
        return torch.exp(log_area / 2) / math.sqrt(math.pi)


class ParametricField(nn.Module):
    def __init__(self, rank, width=64, depth=3):
        super().__init__()
        self.register_buffer('scales', 2.**torch.arange(5))
        self.register_buffer('coefficient_scale', torch.ones(rank))
        n = 3 + rank + 10
        layers = [nn.Linear(n, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers.extend((nn.Linear(width, width), nn.Tanh()))
        layers.append(nn.Linear(width, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x, rho, f, coefficients, geometry):
        t = x / geometry.length
        s = (rho / geometry.radius(x, coefficients)).square()
        phase = math.pi * t * self.scales
        features = torch.cat((t, s, f / 8000, coefficients / self.coefficient_scale,
                              torch.sin(phase), torch.cos(phase)), -1)
        return self.net(features)


def derivative(value, variable):
    return torch.autograd.grad(value.sum(), variable, create_graph=True)[0]


def helmholtz_residual(p, x, rho, f, c=C):
    result = []
    for channel in p.unbind(-1):
        channel = channel[:, None]
        px = derivative(channel, x)
        pr = derivative(channel, rho)
        pxx = derivative(px, x)
        prr = derivative(pr, rho)
        # Limit at the axis is 2 p_rr, NOT a Cartesian 2D Laplacian.
        radial = torch.where(rho.abs() > 1e-10,
                             prr + pr / rho.clamp_min(1e-10), 2 * prr)
        result.append(pxx + radial + (2 * math.pi * f / c).square() * channel)
    return torch.cat(result, -1)


def wall_residual(p, x, rho, coefficients, geometry):
    rprime = derivative(geometry.radius(x, coefficients), x)
    return torch.cat([derivative(v[:, None], rho) - rprime * derivative(v[:, None], x)
                      for v in p.unbind(-1)], -1)


def inlet_residual(p, x, target):
    """p_x prescribed by p_x=-i omega rho_air U/A; target uses chosen scaling."""
    return torch.cat([derivative(v[:, None], x) for v in p.unbind(-1)], -1) - target


def radiation_residual(p, x, f, impedance, c=C):
    """p + z/(i k) p_x=0, z=Z/(rho_air*c), for exp(+i omega t).

    Use f>0; exact DC is evaluated by its limiting boundary condition separately.
    """
    px = torch.cat([derivative(v[:, None], x) for v in p.unbind(-1)], -1)
    pc = torch.complex(p[:, 0], p[:, 1])
    dpc = torch.complex(px[:, 0], px[:, 1])
    r = pc + impedance.flatten() / (1j * 2 * math.pi * f.flatten() / c) * dpc
    return torch.stack((r.real, r.imag), -1)
