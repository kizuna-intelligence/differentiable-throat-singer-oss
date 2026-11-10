"""Conservative staggered pressure/flow graph, CGS units.

Equations and physiological constants reference CHIMERA airway/network.rs and
voice/human_airway.rs, MIT, Copyright (c) 2026 Jameson Hodge.
This CPU reference uses PyTorch autograd and block checkpointing, not a GPU kernel.
The independent Numba forward backend is for long CPU physics experiments only.
"""
from dataclasses import dataclass
import math
import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from .constants import RHO, C_SOUND, MU, RHO_C2, NU, RE_CRIT

@dataclass
class TubeSpec:
    name: str
    n: int
    length: float
    area: object = 3.0
    wall: tuple | None = None
    perimeter: float = 1.0
    wall_radiation: float = 0.0

@dataclass
class PortSpec:
    area: float
    length: float
    entrance: float = 0.37
    exit: float = 1.0
    slit_width: float = 0.0
    half_sections: bool = True
    resistance: float = 0.0

def quadratic_flow(rhs, linear, kinetic):
    """Stable exact root of b U + k |U| U = rhs, including k=0."""
    return 2 * rhs / (linear + (linear.square() + 4 * kinetic * rhs.abs()).sqrt())

class AirwayNetwork(nn.Module):

    def __init__(self, sample_rate=96000, *, nonlinear=True, viscous=True, volume_source=True, closure_area=1e-05, frequency_dependent_loss=False):
        super().__init__()
        self.sample_rate = int(sample_rate)
        self.nonlinear, self.viscous = (nonlinear, viscous)
        self.volume_source, self.closure_area = (volume_source, closure_area)
        self.frequency_dependent_loss = frequency_dependent_loss
        from .acoustics import relaxation_fit
        poles, gains, error = relaxation_fit()
        self.register_buffer('loss_poles', torch.tensor(poles, dtype=torch.float64))
        self.register_buffer('loss_gains', torch.tensor(gains, dtype=torch.float64))
        self.loss_fit_error = error
        self.tubes, self.names = ({}, [])
        self._areas, self._dx, self._walls, self._perim, self._wall_rad = ([], [], [], [], [])
        self._edges, self._ports, self._fixed, self._radiation = ([], [], {}, [])
        self.noise_sources = []
        self.finalized = False

    def add_tube(self, spec):
        if self.finalized or spec.n < 1 or spec.length <= 0:
            raise ValueError('Invalid or finalized tube')
        first = len(self._areas)
        area = np.broadcast_to(spec.area, (spec.n,)).tolist()
        self._areas.extend(area)
        self._dx.extend([spec.length / spec.n] * spec.n)
        self._walls.extend([spec.wall or (0.0, 0.0, 0.0)] * spec.n)
        self._perim.extend([spec.perimeter] * spec.n)
        self._wall_rad.extend([spec.wall_radiation] * spec.n)
        nodes = list(range(first, first + spec.n))
        self.tubes[spec.name] = nodes
        self.names.extend([spec.name] * spec.n)
        for a, b in zip(nodes[:-1], nodes[1:]):
            self.connect(a, b)
        return nodes

    def add_pressure_node(self, name, pressure=0.0):
        node = len(self._areas)
        self._areas.append(1.0)
        self._dx.append(0.0)
        self._walls.append((0.0, 0.0, 0.0))
        self._perim.append(1.0)
        self._wall_rad.append(0.0)
        self.names.append(name)
        self._fixed[node] = pressure
        return node

    def add_cavity(self, name, volume, wall=None):
        return self.add_tube(TubeSpec(name, 1, volume ** (1 / 3), volume ** (2 / 3), wall))[0]

    def connect(self, a, b, port=None):
        if self.finalized:
            raise RuntimeError('Graph is finalized')
        edge = len(self._edges)
        self._edges.append((a, b))
        self._ports.append(port)
        return edge

    def add_radiation(self, node):
        self._radiation.append(node)

    def finalize(self):
        if self.finalized:
            return self

        def buf(k, x, dtype=torch.float64):
            self.register_buffer(k, torch.as_tensor(x, dtype=dtype))
        buf('areas', self._areas)
        buf('dx', self._dx)
        buf('walls', self._walls)
        buf('perimeter', self._perim)
        buf('wall_radiation', self._wall_rad)
        buf('edge_a', [a for a, b in self._edges], torch.long)
        buf('edge_b', [b for a, b in self._edges], torch.long)
        n, e = (len(self._areas), len(self._edges))
        incidence = np.zeros((n, e))
        for j, (a, b) in enumerate(self._edges):
            incidence[a, j] = -1
            incidence[b, j] = 1
        buf('incidence', incidence)
        buf('extra_inertance', np.zeros(e))
        buf('is_port', [p is not None for p in self._ports], torch.bool)
        buf('port_area', [p.area if p else 1.0 for p in self._ports])
        buf('port_len', [p.length if p else 0.0 for p in self._ports])
        buf('port_half', [float(p.half_sections) if p else 1.0 for p in self._ports])
        buf('port_entrance', [p.entrance if p else 0.0 for p in self._ports])
        buf('port_exit', [p.exit if p else 0.0 for p in self._ports])
        buf('port_slit', [p.slit_width if p else 0.0 for p in self._ports])
        buf('port_r', [p.resistance if p else 0.0 for p in self._ports])
        buf('fixed_mask', [k in self._fixed for k in range(n)], torch.bool)
        buf('fixed_pressure', [self._fixed.get(k, 0.0) for k in range(n)])
        buf('rad_mask', [k in self._radiation for k in range(n)])
        self.finalized = True
        return self

    def elements(self, areas=None, dx=None, port_area=None, wall_scale=1.0, viscous_scale=1.0, minimum_areas=None):
        areas = self.areas if areas is None else areas
        if minimum_areas is None:
            minimum_areas = getattr(self, 'profile_minima', None)
        dx = self.dx if dx is None else dx
        pa = self.port_area if port_area is None else port_area
        aa = areas.clamp_min(1e-06)
        ap = pa.clamp_min(1e-06)
        a, b = (self.edge_a, self.edge_b)
        half = 0.5 * RHO * dx / aa
        inertance = (half[..., a] + half[..., b]) * self.port_half
        inertance = inertance + RHO * self.port_len / ap + self.extra_inertance
        perimeter = 2 * math.sqrt(math.pi) * aa.sqrt() * self.perimeter
        factor = math.sqrt(RHO * MU * 2 * math.pi * 1000 / 2)
        resistance = 0.5 * factor * (perimeter * dx / aa.square())
        resistance = (resistance[..., a] + resistance[..., b]) * self.port_half
        port_r = factor * 2 * math.sqrt(math.pi) * self.port_len / ap.pow(1.5)
        slit_r = 12 * MU * self.port_len * self.port_slit.square() / ap.pow(3)
        resistance = resistance + torch.where(self.port_slit > 0, slit_r, port_r) + self.port_r
        if self.frequency_dependent_loss:
            resistance = torch.where(self.port_slit > 0, slit_r, slit_r * 0) + self.port_r
        resistance = resistance * viscous_scale if self.viscous else resistance * 0
        fwd = 0.5 * RHO * ((1 / aa[..., a] - 1 / aa[..., b]).relu().square() + 0.1 * (1 / aa[..., b] - 1 / aa[..., a]).relu().square())
        rev = 0.5 * RHO * ((1 / aa[..., b] - 1 / aa[..., a]).relu().square() + 0.1 * (1 / aa[..., a] - 1 / aa[..., b]).relu().square())
        if minimum_areas is not None:
            throat = torch.minimum(minimum_areas[..., a], minimum_areas[..., b]).clamp_min(1e-06)
            fwd = 0.5 * RHO * ((1 / throat - 1 / aa[..., b]).relu().square() + 0.1 * (1 / throat - 1 / aa[..., a]).relu().square())
            rev = 0.5 * RHO * ((1 / throat - 1 / aa[..., a]).relu().square() + 0.1 * (1 / throat - 1 / aa[..., b]).relu().square())
        fp = 0.5 * RHO * (self.port_entrance / ap.square() + self.port_exit * (1 / ap - 1 / aa[..., b]).relu().square())
        rp = 0.5 * RHO * (self.port_entrance / ap.square() + self.port_exit * (1 / ap - 1 / aa[..., a]).relu().square())
        fwd, rev = (torch.where(self.is_port, fp, fwd), torch.where(self.is_port, rp, rev))
        if not self.nonlinear:
            fwd, rev = (fwd * 0, rev * 0)
        local = areas if minimum_areas is None else minimum_areas
        effective = torch.where(self.is_port, pa, torch.minimum(local[..., a], local[..., b]))
        opened = effective >= self.closure_area
        compliance = (areas.clamp_min(self.closure_area) * dx / RHO_C2).clamp_min(1e-15)
        wall_area = perimeter * dx
        internal = ~self.fixed_mask[self.edge_a] & ~self.fixed_mask[self.edge_b]
        conductance = opened.detach() / inertance.clamp_min(1e-15)
        cfl = conductance * (1 + internal.to(inertance.dtype)) @ self.incidence.abs().T / compliance
        active = ~self.fixed_mask
        root = compliance.detach().sqrt()
        diagonal = conductance.detach() @ self.incidence.abs().T / compliance.detach()
        off = conductance.detach() * internal / (root[..., self.edge_a] * root[..., self.edge_b])
        symmetric = diagonal + off @ self.incidence.abs().T
        bound = torch.minimum(cfl[..., active].detach().amax(-1), symmetric[..., active].amax(-1))
        rejected = bound / self.sample_rate ** 2 >= 3.9
        if torch.any(rejected):
            with torch.no_grad():
                flow = conductance.detach().reshape(-1, len(self.edge_a))[rejected.reshape(-1)]
                comp = compliance.detach().reshape(-1, len(self.areas))[rejected.reshape(-1)][:, active]
                inc = self.incidence[active].double()
                for first in range(0, len(flow), 128):
                    stiffness = inc[None] * flow[first:first + 128, None, :].double() @ inc.T
                    root = comp[first:first + 128].double().sqrt()
                    matrix = stiffness / (root[:, :, None] * root[:, None, :]) / self.sample_rate ** 2
                    maximum = torch.linalg.eigvalsh(matrix)[..., -1].max()
                    if maximum >= 3.9:
                        raise ValueError(f'Unstable graph CFL: eigenvalue {float(maximum):.6g}; reduce sections or increase sample rate')
        return (inertance, resistance, fwd, rev, opened, compliance, wall_area, effective)

    def initial_state(self, batch=1):
        z = self.areas.new_zeros(batch, len(self.areas))
        e = self.areas.new_zeros(batch, len(self.edge_a))
        bx = e[..., None].expand(-1, -1, len(self.loss_poles)).clone()
        nx = z[..., None].expand(-1, -1, len(self.loss_poles)).clone()
        return (z + self.fixed_pressure, e, e + 1.0, z, z, z, z, self.areas.expand(batch, -1) * self.dx, z[:, 0], bx, nx)

    def relaxation_elements(self, elements, dx=None):
        """Kirchhoff thin-boundary-layer viscosity + thermal admittance.

        Z'=rho/A*(s+2*sqrt(nu)/radius*sqrt(s));
        Y'=A/(rho*c²)*(s+2*(gamma-1)*sqrt(alpha)/radius*sqrt(s)).
        Each sqrt(s) is a sum of positive-real RC/RL relaxations.
        """
        l, r, kf, kr, opened, compliance, wall_area, eff = elements
        dx = self.dx if dx is None else dx
        area = (compliance * RHO_C2 / dx.clamp_min(1e-12)).clamp_min(1e-06)
        radius = (area / math.pi).sqrt()
        node_factor = RHO * dx / area * 2 * math.sqrt(NU) / radius * self.perimeter
        branch_factor = 0.5 * (node_factor[..., self.edge_a] + node_factor[..., self.edge_b]) * self.port_half
        ap = eff.clamp_min(1e-06)
        branch_factor = branch_factor + RHO * self.port_len / ap * 2 * math.sqrt(NU) / (ap / math.pi).sqrt() * (self.port_slit == 0)
        branch_gain = branch_factor[..., None] * self.loss_gains
        node_gain = (compliance * 0.4 * 2 * math.sqrt(NU / 0.71) / radius)[..., None] * self.loss_gains
        node_gain = node_gain * (~self.fixed_mask)[..., None]
        return (branch_gain, node_gain)

    def step(self, state, elements, injection, noise, noise_gain=1.0, wall_scale=1.0, volume_delta=None):
        p, momentum, old_l, wall_v, wall_y, rad_ul, old_rad_l, old_volume, old_radiated, bx, nx = state
        l, r, kf, kr, opened, compliance, wall_area, eff = elements
        dt, fs = (1 / self.sample_rate, self.sample_rate)
        rhs = p[..., self.edge_a] - p[..., self.edge_b]
        momentum = momentum * torch.minimum(torch.ones_like(l), l / old_l.clamp_min(1e-20))
        psrc = torch.zeros_like(rhs)
        for i, (src, fwd, rev, gain) in enumerate(self.noise_sources):
            u = momentum[..., src] / l[..., src]
            diameter = 2 * torch.sqrt(eff[..., src].clamp_min(1e-06) / math.pi)
            reynolds = u.abs() * diameter / (NU * eff[..., src].clamp_min(1e-06))
            excess = (reynolds.square() - RE_CRIT ** 2).relu()
            saturation = 12000.0 ** 2 - RE_CRIT ** 2
            amp = gain * noise_gain * excess / (1 + excess / saturation) * math.sqrt(96000 / self.sample_rate)
            signal = amp * noise[..., i]
            psrc = psrc + torch.nn.functional.one_hot(self.edge_a.new_tensor(fwd), len(self.edge_a)) * torch.where(u >= 0, signal, signal * 0)[..., None]
            psrc = psrc - torch.nn.functional.one_hot(self.edge_a.new_tensor(rev), len(self.edge_a)) * torch.where(u < 0, signal, signal * 0)[..., None]
        rhs = rhs + psrc + momentum * fs
        thermal_g = torch.zeros_like(p)
        thermal_src = torch.zeros_like(p)
        if self.frequency_dependent_loss and self.viscous:
            bg, ng = self.relaxation_elements(elements)
            lb = self.loss_poles * dt / (2 + self.loss_poles * dt)
            la = 1 - 2 * lb
            old_u = momentum / l
            rhs = rhs + (bg * (la * bx + lb * old_u[..., None])).sum(-1)
            r = r + (bg * (1 - lb)).sum(-1)
            thermal_g = (ng * (1 - lb) * 0.5).sum(-1)
            thermal_src = (ng * (1 - lb) * nx).sum(-1)
        k = torch.where(rhs >= 0, kf, kr)
        u = quadratic_flow(rhs, l * fs + r, k) * opened
        if self.frequency_dependent_loss and self.viscous:
            bx = la * bx + lb * (u + old_u)[..., None]
        flux = u @ self.incidence.T + injection
        volume = compliance * RHO_C2
        if self.volume_source and volume_delta is not None:
            flux = flux - volume_delta * fs
        m, damping, stiffness = self.walls.unbind(-1)
        has_wall = m > 0
        mass = torch.where(has_wall, m * wall_scale, torch.ones_like(m))
        wv = ((mass * fs - 0.5 * damping) * wall_v + p - stiffness * wall_y) / (mass * fs + 0.5 * damping)
        wv = wv * has_wall
        wy = wall_y + dt * wv
        wall_flow = wall_area * wv
        flux = flux - wall_flow
        new_p = (p * (compliance * fs - thermal_g) + flux + thermal_src) / (compliance * fs + thermal_g)
        ar = (volume / self.dx.clamp_min(1e-12)).clamp_min(0.0001)
        lr = 8 * RHO / (3 * math.pi * torch.sqrt(math.pi * ar))
        rr = 128 * RHO * C_SOUND / (9 * math.pi ** 2 * ar)
        prev_ul = rad_ul * torch.minimum(torch.ones_like(lr), old_rad_l / lr)
        h, g = (dt / (2 * lr), 0.5 / rr)
        rad_p = (p * (compliance * fs - 0.5 * h - g - thermal_g) + flux - prev_ul + thermal_src) / (compliance * fs + 0.5 * h + g + thermal_g)
        new_ul = (prev_ul + h * (p + rad_p)) * self.rad_mask
        rad_flow = ((prev_ul + new_ul) * 0.5 + g * (p + rad_p)) * self.rad_mask
        new_p = torch.where(self.rad_mask.bool(), rad_p, new_p)
        new_p = torch.where(self.fixed_mask, self.fixed_pressure, new_p)
        if self.frequency_dependent_loss and self.viscous:
            nx = la * nx + lb * (p + new_p)[..., None]
        radiated = rad_flow.sum(-1) + (wall_flow * self.wall_radiation).sum(-1)
        out = (radiated - old_radiated) * fs
        return ((new_p, l * u, l, wv, wy, new_ul, lr, volume, radiated, bx, nx), out)

    def forward(self, samples, *, areas=None, port_area=None, injection=None, wall_scale=1.0, noise_gain=1.0, seed=0, checkpoint_samples=128, return_pressure=False):
        """Reference differentiable static geometry, optional per-sample ports.

        injection [B,S,N], port_area [B,S,E] or [E]. No normalization or clipping
        is applied to the emitted waveform. Closure uses an exact zero flow.
        """
        if not self.finalized:
            self.finalize()
        if samples < 1:
            raise ValueError('samples must be positive')
        batch = injection.shape[0] if injection is not None else areas.shape[0] if areas is not None and areas.ndim > 1 else 1
        if injection is None:
            injection = self.areas.new_zeros(batch, samples, len(self.areas))
        generator = torch.Generator(device=self.areas.device).manual_seed(seed)
        noise = torch.randn(batch, samples, len(self.noise_sources), dtype=self.areas.dtype, device=self.areas.device, generator=generator)
        if noise.shape[-1]:
            from .acoustics import bandlimit_noise
            noise = bandlimit_noise(noise, self.sample_rate)
        static = self.elements(areas, port_area=port_area if port_area is None or port_area.ndim == 1 else None, wall_scale=wall_scale)
        state = self.initial_state(batch)
        ys, pressures = ([], [])
        for start in range(0, samples, checkpoint_samples):
            end = min(start + checkpoint_samples, samples)

            def block(*args, start=start, end=end):
                st = tuple(args)
                outputs, ps = ([], [])
                for t in range(start, end):
                    el = static if port_area is None or port_area.ndim == 1 else self.elements(areas, port_area=port_area[:, t])
                    st, y = self.step(st, el, injection[:, t], noise[:, t], noise_gain, wall_scale)
                    outputs.append(y)
                    if return_pressure:
                        ps.append(st[0])
                result = (*st, torch.stack(outputs, 1))
                return (*result, torch.stack(ps, 1)) if return_pressure else result
            result = checkpoint(block, *state, use_reentrant=False) if torch.is_grad_enabled() else block(*state)
            state = result[:11]
            ys.append(result[11])
            if return_pressure:
                pressures.append(result[12])
        wave = torch.cat(ys, 1)
        return (wave, torch.cat(pressures, 1)) if return_pressure else wave

    @classmethod
    def open_tube(cls, areas, length=17.0, sample_rate=96000, **kwargs):
        net = cls(sample_rate, **kwargs)
        nodes = net.add_tube(TubeSpec('oral', len(areas), length, areas))
        zero = net.add_pressure_node('atmosphere')
        net.connect(nodes[-1], zero)
        return net.finalize()
