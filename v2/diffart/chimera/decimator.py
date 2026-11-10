"""Fixed linear-phase half-band polyphase decimation, differentiable in Torch."""
import torch
from torch import nn
from scipy.signal import firwin


class HalfBandDecimator(nn.Module):
    def __init__(self,ratio=4,taps=63):
        super().__init__()
        if ratio not in (1,2,4):raise ValueError('Expected ratio 1, 2 or 4')
        self.stages=ratio.bit_length()-1
        kernel=torch.tensor(firwin(taps,.5,window=('kaiser',8.)),dtype=torch.float64)
        self.register_buffer('kernel',kernel[None,None])
    def forward(self,wave):
        value=wave[:,None]
        for _ in range(self.stages):
            value=torch.nn.functional.conv1d(value,self.kernel,stride=2,padding=self.kernel.shape[-1]//2)
        return value[:,0]
