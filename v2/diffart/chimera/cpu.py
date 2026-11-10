"""Compiled CPU forward reference for independent long impulse experiments.
Equations reference CHIMERA airway/network.rs, MIT, Copyright (c) 2026 Jameson Hodge.
The learning path is network.py; this backend has no custom gradient claims.
"""
import numpy as np
from numba import njit
from .constants import RHO, C_SOUND, RHO_C2

@njit(cache=True)
def _simulate(fs, edges_a, edges_b, inertance, resistance, kf, kr, opened, compliance, wall_area, wall, fixed, fixed_p, rad_mask, area, injection, steps, bg, ng, poles):
    n = len(compliance)
    e = len(edges_a)
    p = fixed_p.copy()
    momentum = np.zeros(e)
    wv = np.zeros(n)
    wy = np.zeros(n)
    ul = np.zeros(n)
    pressure = np.zeros((steps, n))
    output = np.zeros(steps)
    end_flow = np.zeros(steps)
    old_rad = 0.0
    dt = 1 / fs
    bx = np.zeros_like(bg)
    nx = np.zeros_like(ng)
    lb = poles * dt / (2 + poles * dt)
    la = 1 - 2 * lb
    for t in range(steps):
        flux = injection[t].copy()
        for j in range(e):
            old_u = momentum[j] / inertance[j]
            rhs = p[edges_a[j]] - p[edges_b[j]] + momentum[j] * fs
            relaxation_r = 0.0
            for h in range(len(poles)):
                rhs += bg[j, h] * (la[h] * bx[j, h] + lb[h] * old_u)
                relaxation_r += bg[j, h] * (1 - lb[h])
            k = kf[j] if rhs >= 0 else kr[j]
            b = inertance[j] * fs + resistance[j] + relaxation_r
            u = 2 * rhs / (b + np.sqrt(b * b + 4 * k * abs(rhs))) if opened[j] else 0.0
            for h in range(len(poles)):
                bx[j, h] = la[h] * bx[j, h] + lb[h] * (u + old_u)
            momentum[j] = inertance[j] * u
            flux[edges_a[j]] -= u
            flux[edges_b[j]] += u
            if fixed[edges_b[j]] and fixed_p[edges_b[j]] == 0:
                end_flow[t] += u
        radiated = 0.0
        for i in range(n):
            if wall[i, 0] > 0:
                m, r, k = wall[i]
                wv[i] = ((m * fs - r / 2) * wv[i] + p[i] - k * wy[i]) / (m * fs + r / 2)
                wy[i] += dt * wv[i]
                flux[i] -= wall_area[i] * wv[i]
            if fixed[i]:
                continue
            thermal_g = 0.0
            thermal_src = 0.0
            old_p = p[i]
            for hidx in range(len(poles)):
                thermal_g += ng[i, hidx] * (1 - lb[hidx]) * 0.5
                thermal_src += ng[i, hidx] * (1 - lb[hidx]) * nx[i, hidx]
            if rad_mask[i]:
                ar = max(area[i], 0.0001)
                lr = 8 * RHO / (3 * np.pi * np.sqrt(np.pi * ar))
                rr = 128 * RHO * C_SOUND / (9 * np.pi * np.pi * ar)
                h = dt / (2 * lr)
                g = 0.5 / rr
                pn = (p[i] * (compliance[i] * fs - 0.5 * h - g - thermal_g) + flux[i] - ul[i] + thermal_src) / (compliance[i] * fs + 0.5 * h + g + thermal_g)
                un = ul[i] + h * (p[i] + pn)
                radiated += 0.5 * (ul[i] + un) + g * (p[i] + pn)
                ul[i] = un
                p[i] = pn
            else:
                p[i] = (p[i] * (compliance[i] * fs - thermal_g) + flux[i] + thermal_src) / (compliance[i] * fs + thermal_g)
            for hidx in range(len(poles)):
                nx[i, hidx] = la[hidx] * nx[i, hidx] + lb[hidx] * (old_p + p[i])
        output[t] = (radiated - old_rad) * fs
        old_rad = radiated
        pressure[t] = p
    return (output, pressure, end_flow)

def simulate(net, seconds=0.5, injection=None):
    """Returns radiation derivative, node pressures, flow to fixed exterior."""

    def array(v):
        return np.ascontiguousarray(v.detach().cpu().numpy())
    elements = net.elements()
    samples = round(seconds * net.sample_rate)
    if injection is None:
        injection = np.zeros((samples, len(net.areas)))
        injection[0, 0] = 1.0
    values = [array(x) for x in elements[:7]]
    return _simulate(net.sample_rate, array(net.edge_a), array(net.edge_b), *values, array(net.walls), array(net.fixed_mask), array(net.fixed_pressure), array(net.rad_mask), array(net.areas), np.asarray(injection, dtype=np.float64), samples, *relaxation_arrays(net, elements))

def relaxation_arrays(net, elements):
    if not net.frequency_dependent_loss or not net.viscous:
        return (np.empty((len(net.edge_a), 0)), np.empty((len(net.areas), 0)), np.empty(0))
    bg, ng = net.relaxation_elements(elements)
    return (*(np.ascontiguousarray(x.detach().cpu().numpy()) for x in (bg, ng)), net.loss_poles.numpy())
