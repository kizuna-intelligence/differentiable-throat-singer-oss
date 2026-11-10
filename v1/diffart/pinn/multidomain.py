"""Physics-only Chebyshev collocation on stepped/curved tubes.

Each rectangle has a separate complex linear head. Interface pressure and axial
velocity are continuous; closed shoulders have zero axial velocity. No reference
solution or acoustic mode is used in fitting. This is a per-geometry/frequency
spectral baseline, not the amortized neural surrogate.
"""
from dataclasses import dataclass

import numpy as np
from scipy import linalg

from .uniform import C, chebyshev_features, tensor


@dataclass(frozen=True)
class Piece:
    start: float
    end: float
    lower: float
    upper: float

    @property
    def length(self):
        return self.end - self.start

    @property
    def width(self):
        return self.upper - self.lower


class TubePINN:
    def __init__(self, pieces, axisymmetric=False, curvature=0., nx=24, ny=10,
                 points=1400, boundary=140, seed=20261109):
        self.pieces = tuple(pieces)
        self.axisymmetric = axisymmetric
        self.curvature = float(curvature)
        self.nx, self.ny = nx, ny
        self.points, self.boundary, self.seed = points, boundary, seed
        self.length = self.pieces[-1].end - self.pieces[0].start
        if axisymmetric and (curvature or any(p.lower != 0 for p in pieces)):
            raise ValueError('Axisymmetry requires lower=0 and zero curvature')
        if curvature and len(pieces) != 1:
            raise ValueError('Constant-curvature validation uses one uniform piece')
        if any(p.width <= 0 or p.length <= 0 or
               min(1-curvature*p.lower, 1-curvature*p.upper) <= 0 for p in pieces):
            raise ValueError('Positive tube widths, lengths and metric required')
        for a, b in zip(pieces[:-1], pieces[1:]):
            if abs(a.end-b.start) > 1e-12 or min(a.upper,b.upper) <= max(a.lower,b.lower):
                raise ValueError('Pieces must touch and have overlapping apertures')
        self.local_size = (nx+1)*(ny+1)
        self.size = self.local_size * len(pieces)

    def local(self, j, x, y, dx=0, dy=0):
        p = self.pieces[j]
        x, y = np.broadcast_arrays(x, y)
        t = (x.ravel()-p.start)/p.length
        if self.axisymmetric:
            s = (y.ravel()/p.upper)**2
            a = chebyshev_features(t, self.nx, dx) / p.length**dx
            b = chebyshev_features(s, self.ny)
            if dy == 1:
                b = chebyshev_features(s, self.ny, 1)* (2*y.ravel()/p.upper**2)[:,None]
            elif dy == 2:
                b = (chebyshev_features(s, self.ny, 2)*(4*s/p.upper**2)[:,None] +
                     chebyshev_features(s, self.ny, 1)*2/p.upper**2)
        else:
            s = (y.ravel()-p.lower)/p.width
            a = chebyshev_features(t, self.nx, dx) / p.length**dx
            b = chebyshev_features(s, self.ny, dy) / p.width**dy
        return tensor(a, b)

    def embed(self, j, matrix):
        out = np.zeros((len(matrix), self.size), dtype=complex)
        out[:, j*self.local_size:(j+1)*self.local_size] = matrix
        return out

    def measure(self, lower, upper):
        return (upper**2-lower**2)/2 if self.axisymmetric else upper-lower

    def train(self, frequency, z=.15, drive=None):
        if frequency <= 0 or np.real(z) < 0:
            raise ValueError('Positive frequency and passive termination required')
        self.frequency, self.impedance = float(frequency), complex(z)
        k = 2*np.pi*frequency/C
        rng = np.random.default_rng(self.seed)
        matrices, targets = [], []

        def add(mat, target=0., weight=1.):
            scale = np.sqrt(weight/len(mat))
            matrices.append(mat*scale)
            targets.append(np.broadcast_to(target, (len(mat),))*scale)

        def sample(p, n):
            v = (1-np.cos(np.pi*(np.arange(n)+.5)/n))/2
            rng.shuffle(v)
            return np.sqrt(p.lower**2+v*(p.upper**2-p.lower**2)) if self.axisymmetric else p.lower+p.width*v

        for j, p in enumerate(self.pieces):
            # Tensor oversampling controls high-degree edge oscillations; random
            # interior-only points can miss shoulder singularities entirely.
            nt = max(2*(self.nx+1), int(np.sqrt(self.points*(self.nx+1)/(self.ny+1))))
            ns = max(2*(self.ny+1), int(np.ceil(self.points/nt)))
            tx = (1-np.cos(np.pi*(np.arange(nt)+.5)/nt))/2
            sy = (1-np.cos(np.pi*(np.arange(ns)+.5)/ns))/2
            tt,ss = np.meshgrid(tx,sy,indexing='ij')
            x = p.start+p.length*tt.ravel()
            y = p.upper*np.sqrt(ss.ravel()) if self.axisymmetric else p.lower+p.width*ss.ravel()
            val = self.local(j,x,y)
            if self.axisymmetric:
                s = (y/p.upper)**2
                t = (x-p.start)/p.length
                radial = tensor(chebyshev_features(t,self.nx),
                    4/p.upper**2*(s[:,None]*chebyshev_features(s,self.ny,2)+chebyshev_features(s,self.ny,1)))
                lap = self.local(j,x,y,dx=2)+radial
            else:
                h = 1-self.curvature*y
                lap = (self.local(j,x,y,dx=2)/h[:,None]**2 + self.local(j,x,y,dy=2)
                       - self.curvature/h[:,None]*self.local(j,x,y,dy=1))
            add(self.embed(j,(lap+k*k*val)*self.length**2/(1+k*self.length)))
            xb = p.start+p.length*(1-np.cos(np.pi*(np.arange(self.boundary)+.5)/self.boundary))/2
            for wall in ([p.upper] if self.axisymmetric else [p.lower,p.upper]):
                add(self.embed(j,self.local(j,xb,np.full_like(xb,wall),dy=1)*self.length))

        first, last = self.pieces[0], self.pieces[-1]
        y = sample(first,self.boundary)
        mat = self.local(0,np.full_like(y,first.start),y,dx=1)*self.length/(1-self.curvature*y)[:,None]
        add(self.embed(0,mat), np.ones_like(y) if drive is None else drive(y))
        y = sample(last,self.boundary)
        x = np.full_like(y,last.end)
        mat = self.local(len(self.pieces)-1,x,y)+z/(1j*k)*self.local(len(self.pieces)-1,x,y,dx=1)/(1-self.curvature*y)[:,None]
        add(self.embed(len(self.pieces)-1,mat)*(1+k*self.length))

        for j,(a,b) in enumerate(zip(self.pieces[:-1],self.pieces[1:])):
            lo,hi = max(a.lower,b.lower),min(a.upper,b.upper)
            v = (1-np.cos(np.pi*(np.arange(2*self.boundary)+.5)/(2*self.boundary)))/2
            y = np.sqrt(lo*lo+v*(hi*hi-lo*lo)) if self.axisymmetric else lo+(hi-lo)*v
            x = np.full_like(y,a.end)
            add(self.embed(j,self.local(j,x,y))-self.embed(j+1,self.local(j+1,x,y)),weight=2.*(1+k*self.length)**2)
            add(self.length*(self.embed(j,self.local(j,x,y,dx=1))-self.embed(j+1,self.local(j+1,x,y,dx=1))),weight=2.)
            for index,p in ((j,a),(j+1,b)):
                for low,high in ((p.lower,lo),(hi,p.upper)):
                    if high-low > 1e-12:
                        y = low+(high-low)*(1-np.cos(np.pi*(np.arange(self.boundary)+.5)/self.boundary))/2
                        add(self.embed(index,self.local(index,np.full_like(y,a.end),y,dx=1))*self.length)
        matrix, rhs = np.concatenate(matrices), np.concatenate(targets).astype(complex)
        scale = np.maximum(np.linalg.norm(matrix,axis=0),1e-15)
        head,_,rank,_ = linalg.lstsq(matrix/scale,rhs,cond=1e-12,lapack_driver='gelsy')
        self.weights = head/scale
        self.training = dict(loss=float(np.linalg.norm(matrix@self.weights-rhs)**2),rank=int(rank),parameters=self.size)
        return self

    def train_variational(self, frequency, z=.15, drive=None):
        """Weak physics solve with mortar pressure continuity.

        Natural Neumann shoulders admit corner gradient singularities. Quadrature
        integrates polynomial feature/test functions; no reference data enter.
        This Galerkin fixed-feature baseline is distinct from the strong PINN.
        """
        if frequency <= 0 or np.real(z) <= 0:
            raise ValueError('Positive frequency and dissipative z required')
        self.frequency, self.impedance = float(frequency), complex(z)
        k = 2*np.pi*frequency/C
        qx,wx = np.polynomial.legendre.leggauss(self.nx+3)
        qy,wy = np.polynomial.legendre.leggauss(2*self.ny+5)
        matrix = np.zeros((self.size,self.size),complex)
        rhs = np.zeros(self.size,complex)
        for j,p in enumerate(self.pieces):
            xx = p.start+(qx+1)*p.length/2
            if self.axisymmetric:
                yy = p.upper*np.sqrt((qy+1)/2)
                yw = wy*p.upper**2/4
            else:
                yy = p.lower+(qy+1)*p.width/2
                yw = wy*p.width/2
            x,y = np.meshgrid(xx,yy,indexing='ij')
            w = (wx[:,None]*p.length/2*yw[None,:]).ravel()
            x,y = x.ravel(),y.ravel()
            h = 1-self.curvature*y
            val = self.local(j,x,y)
            px,py = self.local(j,x,y,dx=1),self.local(j,x,y,dy=1)
            block = px.T@((w/h)[:,None]*px)+py.T@((w*h)[:,None]*py)-k*k*val.T@((w*h)[:,None]*val)
            sl = slice(j*self.local_size,(j+1)*self.local_size)
            if j == 0:
                face = self.local(j,np.full_like(yy,p.start),yy)
                g = np.ones_like(yy) if drive is None else drive(yy)
                rhs[sl] -= face.T@(yw*g)/self.length
            if j == len(self.pieces)-1:
                face = self.local(j,np.full_like(yy,p.end),yy)
                block = block+1j*k/z*face.T@(yw[:,None]*face)
            matrix[sl,sl] = block
        constraints = []
        for j,(a,b) in enumerate(zip(self.pieces[:-1],self.pieces[1:])):
            lo,hi = max(a.lower,b.lower),min(a.upper,b.upper)
            if self.axisymmetric:
                yy = np.sqrt(lo*lo+(qy+1)/2*(hi*hi-lo*lo))
                yw = wy*(hi*hi-lo*lo)/4
            else:
                yy = lo+(qy+1)*(hi-lo)/2
                yw = wy*(hi-lo)/2
            tests = chebyshev_features((qy+1)/2,self.ny)
            face = (self.embed(j,self.local(j,np.full_like(yy,a.end),yy))-
                    self.embed(j+1,self.local(j+1,np.full_like(yy,a.end),yy)))
            constraints.append(tests.T@(yw[:,None]*face))
        if constraints:
            constraint = np.concatenate(constraints)
            n = len(constraint)
            matrix = np.block([[matrix,constraint.T],[constraint,np.zeros((n,n))]])
            rhs = np.r_[rhs,np.zeros(n)]
        scale = np.sqrt(np.maximum(np.max(abs(matrix),axis=0),1e-25))
        balanced = matrix/scale[:,None]/scale[None,:]
        solution = linalg.solve(balanced,rhs/scale,assume_a='sym')/scale
        self.weights = solution[:self.size]
        self.training = dict(method='weak Galerkin fixed-feature physics baseline',
            parameters=self.size, constraints=len(solution)-self.size,
            algebraic_relative_residual=float(np.linalg.norm(matrix@solution-rhs)/np.linalg.norm(rhs)))
        return self

    def pressure(self,j,x,y,dx=0,dy=0):
        x,y = np.broadcast_arrays(x,y)
        w = self.weights[j*self.local_size:(j+1)*self.local_size]
        return (self.local(j,x,y,dx,dy)@w).reshape(x.shape)

    def transfer(self):
        # Physical velocity is proportional to normal pressure derivative.
        v,w = np.polynomial.legendre.leggauss(100)
        p = self.pieces[-1]
        y = p.lower+(v+1)*p.width/2
        density = y if self.axisymmetric else np.ones_like(y)
        flux = np.sum(w*density*self.pressure(len(self.pieces)-1,p.end,y,dx=1)/(1-self.curvature*y))*p.width/2
        a = self.pieces[0]
        return flux*self.length/self.measure(a.lower,a.upper)

    def save(self,path):
        np.savez_compressed(path,weights=self.weights,pieces=[[p.start,p.end,p.lower,p.upper] for p in self.pieces],
            frequency=self.frequency,impedance=self.impedance,axisymmetric=self.axisymmetric,
            curvature=self.curvature,nx=self.nx,ny=self.ny)
