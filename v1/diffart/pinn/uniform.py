"""CPU spectral PINN baseline: a fixed polynomial feature net + learned head.

The complex linear head is fitted to RANDOM Helmholtz/BC collocation points by
least squares, with no reference pressures in training. This is a spectral,
linear-output PINN baseline, not a trained amortized geometry surrogate.
Chebyshev features in t=x/L and s=(rho/R)^2 avoid the axial singularity and
spectral bias of a low-frequency tanh MLP. No acoustic eigenmodes/dispersion
relation or exact pressures are embedded in the feature map.
"""
from dataclasses import dataclass
import math

import numpy as np
from numpy.polynomial import chebyshev as cheb
from scipy import linalg, special


C = 350.0


def chebyshev_features(t, degree, order=0):
    z = 2 * np.asarray(t).reshape(-1) - 1
    values = []
    for j in range(degree + 1):
        coef = np.zeros(j + 1)
        coef[j] = 1
        d = cheb.chebder(coef, m=order)
        values.append(2.**order * cheb.chebval(z, d))
    return np.stack(values, -1)


def tensor(a, b):
    return np.einsum('ni,nj->nij', a, b).reshape(len(a), -1)


def piston_impedance(f, radius, c=C):
    """Normalized baffled uniform-piston radiation, exp(+i omega t).

    Applied locally as a Robin approximation, not full modal radiation coupling.
    """
    ka = 2 * np.pi * np.asarray(f) * radius / c
    safe = np.maximum(ka, 1e-12)
    resistance = 1 - special.j1(2 * ka) / safe
    reactance = special.struve(1, 2 * ka) / safe
    small = abs(ka) < 1e-4
    # Correct DC limit and avoid cancellation in the radiation resistance.
    resistance = np.where(small, ka**2 / 2 - ka**4 / 12, resistance)
    reactance = np.where(small, 8 * ka / (3 * np.pi) - 32 * ka**3 / (45 * np.pi), reactance)
    return resistance + 1j * reactance


@dataclass
class UniformPINN:
    length: float = .17
    radius: float = .01
    axial_degree: int = 40
    radial_degree: int = 4
    interior_points: int = 1000
    boundary_points: int = 96
    seed: int = 20261109

    def features(self, t, s, dt=0, ds=0):
        return tensor(chebyshev_features(t, self.axial_degree, dt),
                      chebyshev_features(s, self.radial_degree, ds))

    def operators(self, t, s, f):
        val = self.features(t, s)
        dt = self.features(t, s, dt=1)
        ds = self.features(t, s, ds=1)
        lap = (self.features(t, s, dt=2) + 4 * (self.length / self.radius)**2 *
               (np.asarray(s)[:, None] * self.features(t, s, ds=2) + ds))
        return val, dt, ds, lap + (2 * np.pi * f * self.length / C)**2 * val

    def train(self, f, z=.03, drive=None):
        """Minimize mean squared PDE + inlet + wall + radiation residuals.

        p_t(0,s)=drive(s). Default 1 is uniform volume velocity, with arbitrary
        complex pressure scaling. z is Z/(rho_air*c); z=0 is pressure release.
        Exactly f=0 requires z=0: a separate static pressure-gauge problem.
        U_out/U_in at DC is evaluated by the limit of plane_transfer(), not by
        identifying the arbitrarily scaled static pressure with loaded pressure.
        """
        if f < 0 or self.radius <= 0 or self.length <= 0 or np.real(z) < 0:
            raise ValueError('f >= 0, positive geometry and passive termination required')
        if f == 0 and z != 0:
            raise ValueError('At DC use z=0 for the static gauge; use plane_transfer for the loaded DC limit')
        if drive is None:
            drive = lambda s: np.ones_like(s)
        rng = np.random.default_rng(self.seed)
        t, s = rng.random((2, self.interior_points))
        _, _, _, pde = self.operators(t, s, f)
        # Use uniform s: correct axisymmetric area measure rho d(rho).
        b = rng.random(self.boundary_points)
        zero, one = np.zeros_like(b), np.ones_like(b)
        inlet = self.features(zero, b, dt=1)
        wall = self.features(b, one, ds=1)
        kL = 2 * np.pi * f * self.length / C
        h = z / (1j * kL) if f > 0 else 0j
        outlet = self.features(one, b) + h * self.features(one, b, dt=1)
        pde_scale = 1 + kL**2
        blocks = [pde / (pde_scale * np.sqrt(len(t))),
                  inlet / np.sqrt(len(b)), wall / np.sqrt(len(b)),
                  outlet / np.sqrt(len(b))]
        matrix = np.concatenate(blocks).astype(np.complex128)
        rhs = np.concatenate((np.zeros(len(t)), drive(b) / np.sqrt(len(b)),
                              np.zeros(2 * len(b)))).astype(np.complex128)
        scale = np.linalg.norm(matrix, axis=0)
        scale = np.maximum(scale, 1e-15)
        # QR with pivoting is faster than gradient descent for a linear head.
        head, _, rank, _ = linalg.lstsq(matrix / scale, rhs, cond=1e-12, lapack_driver='gelsy')
        self.weights = head / scale
        self.frequency, self.impedance = float(f), complex(z)
        self.training = dict(loss=float(np.linalg.norm(matrix @ self.weights - rhs)**2),
                             rank=int(rank), parameters=len(head),
                             pde_scale=pde_scale, seed=self.seed)
        return self

    def pressure(self, t, s):
        t, s = np.broadcast_arrays(t, s)
        return (self.features(t.ravel(), s.ravel()) @ self.weights).reshape(t.shape)

    def axial_derivative(self, t, s):
        t, s = np.broadcast_arrays(t, s)
        return (self.features(t.ravel(), s.ravel(), dt=1) @ self.weights).reshape(t.shape)

    def validate(self, drive=None, points=2048):
        if drive is None:
            drive = lambda s: np.ones_like(s)
        rng = np.random.default_rng(self.seed + 100003)
        t, s = rng.random((2, points))
        p, _, _, op = self.operators(t, s, self.frequency)
        kL = 2 * np.pi * self.frequency * self.length / C
        residual = op @ self.weights
        wave = max(1., kL**2) * max(np.linalg.norm(p @ self.weights), 1e-15)
        b = rng.random(256)
        inlet = self.axial_derivative(np.zeros_like(b), b) - drive(b)
        wall = self.features(b, np.ones_like(b), ds=1) @ self.weights
        h = self.impedance / (1j * kL) if self.frequency > 0 else 0j
        outlet = self.pressure(np.ones_like(b), b) + h * self.axial_derivative(np.ones_like(b), b)
        return dict(pde_relative_l2=float(np.linalg.norm(residual) / wave),
                    inlet_rms=float(np.sqrt(np.mean(abs(inlet)**2))),
                    wall_ds_rms=float(np.sqrt(np.mean(abs(wall)**2))),
                    outlet_rms=float(np.sqrt(np.mean(abs(outlet)**2))))

    def save(self, path):
        np.savez_compressed(path, weights=self.weights, length=self.length, radius=self.radius,
                            axial_degree=self.axial_degree, radial_degree=self.radial_degree,
                            frequency=self.frequency, impedance=self.impedance)


def modal_reference(t, s, f, length, radius, z, amplitudes=(1.,)):
    """INDEPENDENT separation-of-variables reference; never called by train().

    Neumann radial eigenvalues: alpha_0=0, J1(alpha_m)=0. p_t(0,s)=sum
    amplitudes_m*J0(alpha_m*sqrt(s)). Stable even below cutoff (no cosh overflow).
    """
    t, s = np.broadcast_arrays(t, s)
    roots = np.r_[0., special.jn_zeros(1, len(amplitudes) - 1)] if len(amplitudes) > 1 else [0.]
    kL = 2 * np.pi * f * length / C
    h = z / (1j * kL) if f > 0 else 0j
    result = np.zeros(t.shape, dtype=complex)
    for alpha, amplitude in zip(roots, amplitudes):
        beta2 = complex(kL**2 - (alpha * length / radius)**2)
        if abs(beta2) < 1e-12:
            axial = t - 1 - h
        else:
            beta = np.sqrt(beta2)
            # Exponentials referenced to inlet/outlet keep evanescent modes stable.
            if beta.imag > 1:
                eta = beta.imag
                reflection = (h * eta - 1) / (h * eta + 1)
                decay = np.exp(-eta)
                a = 1 / (eta * (reflection * decay**2 - 1))
                axial = a * (np.exp(-eta * t) + reflection * np.exp(-eta * (2 - t)))
            else:
                denominator = np.cos(beta) - h * beta * np.sin(beta)
                a = -(np.sin(beta) / beta + h * np.cos(beta)) / denominator
                axial = a * np.cos(beta * t) + np.sin(beta * t) / beta
        result += amplitude * axial * special.j0(alpha * np.sqrt(s))
    return result


def plane_transfer(f, length, z):
    """U_out/U_in, identical inlet drive and local Robin boundary as PINN."""
    kL = 2 * np.pi * np.asarray(f) * length / C
    return 1 / (np.cos(kL) + 1j * z * np.sin(kL))
