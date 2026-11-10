"""Parametric area source and pressure-driven colliding two-mass folds.

Parametric pulse: CHIMERA sources/glottis.rs, MIT,
Copyright (c) 2026 Jameson Hodge. See THIRD_PARTY_NOTICES.md.
Two-mass: symmetric Steinecke-Herzel simplification, Bernoulli force on
inferior mass, positive contact springs. Equations independently implemented.
Primary reference: https://doi.org/10.1371/journal.pone.0187486
Limitations: quasi-steady separation at the minimum area, no resolved jet.
"""
import math
import numpy as np
from numba import njit
import torch
from torch import nn
from scipy.signal import find_peaks
from .constants import CM_H2O

class ParametricGlottis(nn.Module):

    def __init__(self, sample_rate=96000):
        super().__init__()
        self.sample_rate = sample_rate
        self.oq_raw = nn.Parameter(torch.tensor(0.5, dtype=torch.float64))
        self.area_raw = nn.Parameter(torch.tensor(-1.7, dtype=torch.float64))
        self.skew_raw = nn.Parameter(torch.tensor(0.5, dtype=torch.float64))
        self.chink_raw = nn.Parameter(torch.tensor(-5.1, dtype=torch.float64))
        self.sharpness_raw = nn.Parameter(torch.tensor(math.log(0.6 / 0.4), dtype=torch.float64))
        self.precise_phase = False

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        key = prefix + 'sharpness_raw'
        if key not in state_dict:
            state_dict[key] = self.sharpness_raw.detach().new_tensor(-30.0)
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, f0, pressure=None, seed=0, jitter=0.0, shimmer=0.0, vibrato_cents=0.0, drift_cents=0.0, phase_offset=0.0, oq_offset=0.0, area_offset=0.0, skew_offset=0.0, sharpness_offset=0.0):
        fs = self.sample_rate
        gen = torch.Generator(device=f0.device).manual_seed(seed)
        t = torch.arange(f0.shape[-1], device=f0.device, dtype=f0.dtype) / fs
        drift = torch.zeros_like(f0)
        if drift_cents:
            for h in range(1, 9):
                ph = torch.rand((*f0.shape[:-1], 1), generator=gen, device=f0.device, dtype=f0.dtype) * 2 * math.pi
                drift = drift + torch.sin(2 * math.pi * (0.25 * h) * t + ph) / math.sqrt(h)
        freq = f0 * torch.pow(2.0, (vibrato_cents * torch.sin(2 * math.pi * 6 * t) + drift_cents * drift / 3) / 1200)
        count = 0
        if jitter or shimmer:
            cycles = (torch.cumsum(freq / fs, -1) + phase_offset).floor().long()
            count = int(cycles.max().detach()) + 2
            shape = (*f0.shape[:-1], count)
        if jitter:
            period = torch.randn(shape, generator=gen, device=f0.device, dtype=f0.dtype)
            freq = freq * (1 + jitter * period.gather(-1, cycles))
        if self.precise_phase:
            phase = (torch.cumsum(freq.to(torch.float64) / fs, -1) + phase_offset % 1.0).remainder(1.0).to(f0.dtype)
        else:
            phase = (torch.cumsum(freq / fs, -1) + phase_offset).remainder(1.0)
        oq = 0.25 + 0.7 * (self.oq_raw + oq_offset).sigmoid()
        skew = 0.3 + 0.6 * (self.skew_raw + skew_offset).sigmoid()
        x = (phase / oq).clamp(0, 1)
        opening = 0.5 - 0.5 * torch.cos(math.pi * x / skew)
        u = ((x - skew) / (1 - skew)).clamp(0, 1)
        soft = 0.5 + 0.5 * torch.cos(math.pi * u)
        hard = torch.cos(0.5 * math.pi * u)
        closing = soft + (hard - soft) * (self.sharpness_raw + sharpness_offset).sigmoid()
        pulse = torch.where(x < skew, opening, closing) * (phase < oq)
        amp = 1.0 if not shimmer else 1 + shimmer * torch.randn(shape, generator=gen, device=f0.device, dtype=f0.dtype).gather(-1, cycles)
        gate = (f0 > 0).to(f0.dtype)
        if pressure is not None:
            gate = gate * ((pressure / CM_H2O - 2.0) / 2).clamp(0, 1)
        return torch.nn.functional.softplus(self.chink_raw) + torch.nn.functional.softplus(self.area_raw + area_offset) * pulse * amp * gate

def estimate_f0(area, sample_rate, low=50.0, high=1000.0, discard=0.3):
    value = np.asarray(area)[round(discard * sample_rate):]
    if len(value) < 100 or not np.isfinite(value).all() or np.ptp(value) < 1e-06:
        return (float('nan'), float('inf'))
    peaks, _ = find_peaks(value, distance=int(sample_rate / high * 0.8), prominence=np.ptp(value) * 0.2)
    if len(peaks) < 5:
        return (float('nan'), float('inf'))
    peaks = peaks[(peaks > 0) & (peaks < len(value) - 1)]
    denominator = value[peaks - 1] - 2 * value[peaks] + value[peaks + 1]
    pos = peaks + 0.5 * (value[peaks - 1] - value[peaks + 1]) / np.where(abs(denominator) > 1e-20, denominator, 1)
    periods = np.diff(pos) / sample_rate
    f = 1 / np.median(periods)
    return (float(f), float(np.std(periods) / np.mean(periods)))

@njit(cache=True)
def _two_mass_cpu(sample_rate, samples, tension, sub_pressure, supra_pressure, rest_area, damping):
    x1, x2, v1, v2 = (1e-05, 0.0, 0.0, 0.0)
    length = 1.4
    d1 = 0.25
    m1, m2 = (0.125 / tension, 0.025 / tension)
    k1, k2, kc = (80000 * tension, 8000 * tension, 25000 * tension)
    r1 = 2 * damping * np.sqrt(m1 * k1)
    r2 = 2 * damping * np.sqrt(m2 * k2)
    rest = rest_area / (2 * length)
    dt = 1 / sample_rate
    result = np.zeros(samples)
    lower = np.zeros(samples)
    upper = np.zeros(samples)
    for t in range(samples):
        a1 = max(0.0, rest_area + 2 * length * x1)
        a2 = max(0.0, rest_area + 2 * length * x2)
        dp = sub_pressure[t] - supra_pressure[t]
        amin = min(a1, a2)
        force = length * d1 * dp * (1 - (amin / max(a1, 1e-08)) ** 2)
        c1 = 3 * k1 * min(0.0, x1 + rest)
        c2 = 3 * k2 * min(0.0, x2 + rest)
        f1 = force - k1 * x1 * (1 + 100 * x1 * x1) - kc * (x1 - x2) - c1
        f2 = -k2 * x2 * (1 + 100 * x2 * x2) - kc * (x2 - x1) - c2
        v1 = (m1 * v1 + dt * f1) / (m1 + dt * r1)
        v2 = (m2 * v2 + dt * f2) / (m2 + dt * r2)
        x1 += dt * v1
        x2 += dt * v2
        lower[t] = a1
        upper[t] = a2
        result[t] = min(a1, a2)
    return (result, lower, upper)
