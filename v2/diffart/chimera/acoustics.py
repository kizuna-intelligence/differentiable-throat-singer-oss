"""Causal positive-real relaxation losses and fixed bandlimited noise.

A sum of g*s/(s+p) branches is a passive diffusive approximation of sqrt(s).
Unlike multiplying a noncausal complex response ratio, each state is realizable
in the time-domain graph. Coefficients are fitted on the declared frequency band.
"""
import numpy as np
import torch
from scipy.optimize import nnls
from scipy.signal import lfilter


def relaxation_fit(low=50., high=10000., count=8):
    poles = 2 * np.pi * np.geomspace(low / 8, high * 8, count)
    s = 2j * np.pi * np.geomspace(low, high, 300)
    basis = s[:, None] / (s[:, None] + poles)
    target = np.sqrt(s)
    matrix = np.concatenate((basis.real, basis.imag))
    gain = nnls(matrix / np.r_[abs(target), abs(target)][:, None],
                np.r_[target.real, target.imag] / np.r_[abs(target), abs(target)])[0]
    prediction = basis @ gain
    return poles, gain, float(np.max(abs(prediction / target - 1)))


def bandlimit_noise(noise, sample_rate, low=400., high=7000.):
    if noise.is_cuda:
        # Exact causal impulse, truncated below 1e-40. Linear FFT convolution
        # keeps the zero initial conditions and avoids CPU transfers of audio.
        # Both the input noise and this filter are fixed, never learned.
        import math
        ah = np.exp(-2 * np.pi * low / sample_rate)
        al = np.exp(-2 * np.pi * high / sample_rate)
        taps = math.ceil(100 / min(-np.log(ah), -np.log(al)))
        impulse = np.zeros(taps);impulse[0] = 1.
        impulse = lfilter([ah,-ah],[1.,-ah],impulse)
        impulse = lfilter([1-al],[1.,-al],impulse)
        impulse = lfilter([1-al],[1.,-al],impulse)
        n = noise.shape[1];nfft = 1 << (n+taps-2).bit_length()
        kernel = torch.as_tensor(impulse,device=noise.device,dtype=noise.dtype)
        with torch.no_grad():
            x = noise.detach().transpose(1,2).contiguous()
            value = torch.fft.irfft(torch.fft.rfft(x,n=nfft) * torch.fft.rfft(kernel,n=nfft),n=nfft)
            return value[...,:n].transpose(1,2).contiguous()

    # Noise is exogenous, so generating/filtering it outside the gradient graph
    # is intentional. No learnable signal is detached here.
    white = noise.detach().cpu().numpy()
    ah = np.exp(-2 * np.pi * low / sample_rate)
    al = np.exp(-2 * np.pi * high / sample_rate)
    value = lfilter([ah, -ah], [1., -ah], white, axis=1)
    value = lfilter([1 - al], [1., -al], value, axis=1)
    value = lfilter([1 - al], [1., -al], value, axis=1)
    return torch.as_tensor(value, dtype=noise.dtype, device=noise.device)
