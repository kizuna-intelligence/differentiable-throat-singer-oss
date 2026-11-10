"""Independent cell-centred conservative finite-volume Helmholtz reference.

No collocation features, interface fitting or neural weights are shared with the
PINN. Uniform orthogonal grids align exactly with step faces; radial annular
volumes implement axisymmetry. Curved slabs use the exact circular-arc metric.
Robin conditions eliminate a half-cell to the boundary (second-order centred
flux); rigid shoulders are absent faces. exp(+i omega t), inlet p_n=-1/L.
"""
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from .uniform import C


@dataclass
class FVResult:
    x: np.ndarray
    y: np.ndarray
    piece: np.ndarray
    pressure: np.ndarray
    volumes: np.ndarray
    transfer: complex
    cells: int
    algebraic_relative_residual: float


def solve(pieces, frequency, z=.15, spacing=.0005, axisymmetric=False, curvature=0., drive=None):
    if frequency <= 0 or np.real(z) < 0 or spacing <= 0:
        raise ValueError('Positive frequency/spacing and passive termination required')
    if axisymmetric and (curvature or any(p.lower != 0 for p in pieces)):
        raise ValueError('Invalid axisymmetric geometry')
    if curvature and len(pieces) != 1:
        raise ValueError('Curved reference currently supports one piece')
    length = pieces[-1].end-pieces[0].start
    ymin,ymax = min(p.lower for p in pieces),max(p.upper for p in pieces)
    nx,ny = round(length/spacing),round((ymax-ymin)/spacing)
    dx,dy = length/nx,(ymax-ymin)/ny
    # Refuse hidden staircasing; all dimensions and interfaces must align.
    for value in [p.start-pieces[0].start for p in pieces]+[p.end-pieces[0].start for p in pieces]:
        if abs(value/dx-round(value/dx))>1e-8:
            raise ValueError('Axial step does not align with grid')
    for value in [p.lower-ymin for p in pieces]+[p.upper-ymin for p in pieces]:
        if abs(value/dy-round(value/dy))>1e-8:
            raise ValueError('Transverse step does not align with grid')
    ids = -np.ones((nx,ny),dtype=int)
    locations,labels = [],[]
    for i in range(nx):
        x = pieces[0].start+(i+.5)*dx
        for j,p in enumerate(pieces):
            if p.start < x < p.end:
                for r in range(round((p.lower-ymin)/dy),round((p.upper-ymin)/dy)):
                    ids[i,r] = len(locations)
                    locations.append((i,r))
                    labels.append(j)
                break
    locations = np.asarray(locations)
    x = pieces[0].start+(locations[:,0]+.5)*dx
    y = ymin+(locations[:,1]+.5)*dy
    h = 1-curvature*y
    if np.any(h <= 0):
        raise ValueError('Invalid curvature metric')
    axial_area = y*dy if axisymmetric else np.full_like(y,dy)
    volumes = axial_area*h*dx
    k = 2*np.pi*frequency/C
    diagonal = -(k*k)*volumes.astype(complex)
    rhs = np.zeros(len(x),complex)
    rows,cols,values = [],[],[]
    exit_ids,exit_conductance = [],[]
    for idx,(i,r) in enumerate(locations):
        for di,dr in ((1,0),(-1,0),(0,1),(0,-1)):
            ni,nr = i+di,r+dr
            if 0<=ni<nx and 0<=nr<ny and ids[ni,nr]>=0:
                other = ids[ni,nr]
                if di:
                    conductance = axial_area[idx]/(h[idx]*dx)
                else:
                    face_y = ymin+(r+(dr+1)/2)*dy
                    conductance = (face_y if axisymmetric else 1-curvature*face_y)*dx/dy
                diagonal[idx] += conductance
                rows.append(idx); cols.append(other); values.append(-conductance)
            elif di == -1 and i == 0:
                rhs[idx] -= axial_area[idx]/length*(1 if drive is None else drive(y[idx]))
            elif di == 1 and i == nx-1:
                conductance = axial_area[idx]/(h[idx]*dx/2+z/(1j*k))
                diagonal[idx] += conductance
                exit_ids.append(idx); exit_conductance.append(conductance)
    rows.extend(range(len(x))); cols.extend(range(len(x))); values.extend(diagonal)
    matrix = sparse.coo_matrix((values,(rows,cols)),shape=(len(x),len(x))).tocsc()
    pressure = spsolve(matrix,rhs)
    inlet = pieces[0]
    inlet_measure = (inlet.upper**2-inlet.lower**2)/2 if axisymmetric else inlet.width
    transfer = -np.dot(exit_conductance,pressure[exit_ids])*length/inlet_measure
    residual = np.linalg.norm(matrix@pressure-rhs)/max(np.linalg.norm(rhs),1e-30)
    return FVResult(x,y,np.asarray(labels),pressure,volumes,complex(transfer),len(x),float(residual))
