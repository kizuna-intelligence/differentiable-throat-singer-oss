"""Exact critically damped dynamics, rather than Euler stepping.
Ranges/speeds: CHIMERA control/articulators.rs, MIT.
Copyright (c) 2026 Jameson Hodge. See THIRD_PARTY_NOTICES.md.
"""
import math
import torch
from torch import nn

NAMES = ('jaw','tbX','tbY','ttX','ttY','ttCurl','root','lipAp','lipProt',
         'lipWidth','labiodental','velum','larynx','epilarynx','ventricular','lateral','dorsum')
NEUTRAL = (.6,2.9,6.85,6.45,7.3,0.,0.,.7,.2,.5,0.,-.15,0.,0.,0.,0.,0.)
RANGES = ((-.1,2.6),(1,4.8),(4.8,9.2),(3.5,8),(5,10.8),(0,1),(-1,1),(-.4,2.4),
          (-.2,1.5),(0,1),(0,1),(-.25,1.4),(-1.6,1.6),(0,1),(0,1),(0,1),(0,1))
SPEEDS = (10,12,12,13,13,9,9,11,11,11,10,12,8,6,6,10,9)


class ArticulatorDynamics(nn.Module):
    def __init__(self, sample_rate=100, speeds=None, bounds=None, neutral=None):
        super().__init__()
        self.sample_rate = sample_rate
        self.register_buffer('omega', 2 * math.pi * torch.tensor(SPEEDS if speeds is None else speeds, dtype=torch.float64))
        self.register_buffer('bounds', torch.tensor(RANGES if bounds is None else bounds, dtype=torch.float64))
        self.register_buffer('neutral', torch.tensor(NEUTRAL if neutral is None else neutral, dtype=torch.float64))

    def forward(self, targets, strength=None, initial=None):
        """[B,T,K] zero-order held targets; exact state transition per frame.

        Direct convolution of the analytic impulse response handles time in
        parallel, preserving the initial x and v. No Python loop over frames.
        """
        if strength is not None:
            targets = self.neutral + strength * (targets - self.neutral)
        targets = targets.clamp(self.bounds[:,0], self.bounds[:,1])
        initial = self.neutral.expand_as(targets[:,0]) if initial is None else initial
        t = targets.shape[1]; dt = 1 / self.sample_rate
        times = torch.arange(t + 1, device=targets.device, dtype=targets.dtype)[:,None] * dt
        step = 1 - (1 + times * self.omega) * torch.exp(-times * self.omega)
        impulse = step[1:] - step[:-1]
        nfft = 1 << (2 * t - 1).bit_length()
        input_spec = torch.fft.rfft((targets - initial[:,None]).transpose(1,2), n=nfft)
        kernel_spec = torch.fft.rfft(impulse.T, n=nfft)
        return initial[:,None] + torch.fft.irfft(input_spec * kernel_spec, n=nfft)[...,:t].transpose(1,2)
