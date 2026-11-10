"""Continuum pressure/volume-flow chain matrices, CGS; independent of FDTD.
Reference: CHIMERA airway/frequency.rs, MIT, Copyright (c) 2026 Jameson Hodge.
"""
import math
import numpy as np
import torch
from scipy.optimize import brentq
from .constants import RHO, C_SOUND


def chain(areas, lengths, frequency, attenuation=0.):
    f = torch.as_tensor(frequency, dtype=areas.dtype, device=areas.device)
    gamma = attenuation + 2j * math.pi * f / C_SOUND
    result = torch.eye(2, dtype=torch.complex128 if areas.dtype == torch.float64 else torch.complex64,
                       device=areas.device).expand(*areas.shape[:-1], len(f), 2, 2)
    lengths = torch.broadcast_to(torch.as_tensor(lengths, dtype=areas.dtype, device=areas.device), areas.shape)
    for i in range(areas.shape[-1]):
        x = lengths[..., i, None] * gamma
        co, si = torch.cosh(x), torch.sinh(x)
        z = RHO * C_SOUND / areas[..., i, None]
        m = torch.stack((torch.stack((co, z * si), -1), torch.stack((si / z, co), -1)), -2)
        result = result @ m
    return result


def transfer(areas, lengths, frequency, attenuation=0., load=0.):
    m = chain(areas, lengths, frequency, attenuation)
    # U_out/U_in with p_out = Z_load U_out, rigid volume-flow inlet.
    return 1 / (m[..., 1, 0] * load + m[..., 1, 1])


def resonances(areas, lengths, maximum=5000):
    a = torch.as_tensor(areas, dtype=torch.float64)
    dx = torch.as_tensor(lengths, dtype=torch.float64)
    def determinant(f):
        return float(chain(a, dx, [f])[0, 1, 1].real)
    grid = np.arange(1., maximum + 1, 10.)
    values = chain(a, dx, grid)[:, 1, 1].real.numpy()
    roots = [brentq(determinant, lo, hi, xtol=1e-8) for lo, hi, x, y in
             zip(grid[:-1], grid[1:], values[:-1], values[1:]) if x * y < 0]
    return np.asarray(roots)


def graph_impedance(net, frequencies, source_node=0, chunk=64):
    """Continuous-frequency reference for the full passive linear graph.

    Includes actual topology, soft-wall mobility, radiation R||L and the same
    causal viscous/thermal relaxation spectrum. Nonlinear/turbulent terms are
    zero in this small-signal test, not fitted to the measured time response.
    """
    import numpy as np
    from .constants import RHO_C2,RHO,C_SOUND
    def a(x):return x.detach().cpu().numpy()
    elements=net.elements();l,r,kf,kr,opened,comp,wall_area,eff=map(a,elements)
    bg,ng=map(a,net.relaxation_elements(elements));poles=a(net.loss_poles)
    n=len(comp);ea=a(net.edge_a);eb=a(net.edge_b)
    fixed=a(net.fixed_mask);free=np.flatnonzero(~fixed);source=int(np.flatnonzero(free==source_node)[0])
    mass,damping,stiffness=a(net.walls).T;haswall=mass>0
    area=(comp*RHO_C2/np.maximum(a(net.dx),1e-12)).clip(1e-4)
    lr=8*RHO/(3*np.pi*np.sqrt(np.pi*area));rr=128*RHO*C_SOUND/(9*np.pi**2*area)
    output=[]
    for start in range(0,len(frequencies),chunk):
        freq=np.asarray(frequencies[start:start+chunk]);s=2j*np.pi*freq
        relaxation=s[:,None]/(s[:,None]+poles)
        z=s[:,None]*l+r
        thermal=np.zeros((len(s),n),complex)
        if net.frequency_dependent_loss and net.viscous:
            z=z+relaxation@bg.T;thermal=relaxation@ng.T
        edge_y=opened/z
        diag=s[:,None]*comp+thermal
        wall_d=mass*s[:,None]**2+damping*s[:,None]+stiffness
        wall_d=np.where(haswall,wall_d,1.)
        diag+=wall_area*s[:,None]/wall_d*haswall
        diag+=(1/rr+1/(s[:,None]*lr))*a(net.rad_mask)
        matrix=np.zeros((len(s),n,n),complex)
        ix=np.arange(n);matrix[:,ix,ix]=diag
        for j,(u,v) in enumerate(zip(ea,eb)):
            y=edge_y[:,j]
            matrix[:,u,u]+=y;matrix[:,v,v]+=y;matrix[:,u,v]-=y;matrix[:,v,u]-=y
        reduced=matrix[:,free][:,:,free]
        rhs=np.zeros((len(s),len(free),1),complex);rhs[:,source,0]=1.
        solution=np.linalg.solve(reduced,rhs)
        output.extend(solution[:,source,0])
    return np.asarray(output)
