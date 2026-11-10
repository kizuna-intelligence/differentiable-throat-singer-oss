"""Independent P1 triangular axisymmetric FEM for smooth radius profiles.

Uses only numpy/scipy: natural rigid walls, specified inlet derivative 1/L,
local Robin radiation. Linear triangles follow a radius sampled on axial nodes.
No torch/autograd, neural features or neural residuals enter the assembly.
"""
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from .uniform import C


@dataclass
class SmoothResult:
    nodes: np.ndarray
    pressure: np.ndarray
    weights: np.ndarray
    transfer: complex
    algebraic_relative_residual: float


def solve(radius, length, frequency, impedance, nx=240, nr=20):
    if length <= 0 or frequency <= 0 or np.real(impedance) <= 0 or nx < 2 or nr < 2:
        raise ValueError('Positive geometry, frequency, resistance and mesh sizes required')
    x=np.linspace(0,length,nx+1)
    r=np.asarray(radius(x))
    if np.any(r <= 0) or r.shape != x.shape:
        raise ValueError('Radius must return a positive value per axial node')
    q=np.linspace(0,1,nr+1)
    xx,rr=np.meshgrid(x,q,indexing='ij')
    nodes=np.stack((xx.ravel(),(rr*r[:,None]).ravel()),axis=-1)
    base=(np.arange(nx)[:,None]*(nr+1)+np.arange(nr)[None,:]).ravel()
    triangles=np.concatenate((np.stack((base,base+nr+1,base+nr+2),-1),
                              np.stack((base,base+nr+2,base+1),-1)))
    tri=nodes[triangles]
    delta=tri[:,1:]-tri[:,0,None]
    determinant=delta[:,0,0]*delta[:,1,1]-delta[:,0,1]*delta[:,1,0]
    area=abs(determinant)/2
    # Gradients of each barycentric basis, constant within each triangle.
    grad=np.stack((tri[:,[1,2,0],1]-tri[:,[2,0,1],1],
                   tri[:,[2,0,1],0]-tri[:,[1,2,0],0]),-1)/determinant[:,None,None]
    stiffness=area[:,None,None]*tri[:,:,1].mean(-1)[:,None,None]*np.einsum('tia,tja->tij',grad,grad)
    bary=np.array([[1/3,1/3,1/3],[.6,.2,.2],[.2,.6,.2],[.2,.2,.6]])
    quadrature=np.array([-27,25,25,25])/48
    rq=tri[:,:,1]@bary.T
    mass=area[:,None,None]*np.einsum('tq,q,qi,qj->tij',rq,quadrature,bary,bary)
    block=stiffness-(2*np.pi*frequency/C)**2*mass
    rows=np.broadcast_to(triangles[:,:,None],block.shape).ravel()
    cols=np.broadcast_to(triangles[:,None,:],block.shape).ravel()
    values=block.astype(complex).ravel()
    rhs=np.zeros(len(nodes),complex)
    lump=np.zeros(len(nodes))
    np.add.at(lump,triangles.ravel(),mass.sum(-1).ravel())
    # Boundary integration uses degree-three exact two-point Gaussian quadrature.
    u,w=np.polynomial.legendre.leggauss(2);phi=np.stack(((1-u)/2,(1+u)/2),-1)
    boundary_rows=[];boundary_cols=[];boundary_values=[]
    end_pairs=None;end_integral=None
    for side in (0,nx):
        pairs=side*(nr+1)+np.stack((np.arange(nr),np.arange(1,nr+1)),-1)
        rad=nodes[pairs,1]
        ds=rad[:,1]-rad[:,0]
        rq=rad@phi.T
        integrals=ds[:,None]/2*np.einsum('eq,q,qi->ei',rq,w,phi)
        if side==0:
            np.add.at(rhs,pairs.ravel(),-integrals.ravel()/length)
        else:
            face=ds[:,None,None]/2*np.einsum('eq,q,qi,qj->eij',rq,w,phi,phi)
            boundary_rows=np.broadcast_to(pairs[:,:,None],face.shape).ravel()
            boundary_cols=np.broadcast_to(pairs[:,None,:],face.shape).ravel()
            boundary_values=(1j*2*np.pi*frequency/C/impedance*face).ravel()
            end_pairs,end_integral=pairs,integrals
    matrix=sparse.coo_matrix((np.r_[values,boundary_values],
        (np.r_[rows,boundary_rows],np.r_[cols,boundary_cols])),shape=(len(nodes),len(nodes))).tocsc()
    pressure=spsolve(matrix,rhs)
    transfer=-1j*2*np.pi*frequency/C/impedance*np.sum(end_integral*pressure[end_pairs])*length/(r[0]**2/2)
    residual=np.linalg.norm(matrix@pressure-rhs)/np.linalg.norm(rhs)
    return SmoothResult(nodes,pressure,lump,complex(transfer),float(residual))
