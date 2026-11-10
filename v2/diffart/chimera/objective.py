"""main_round2 multi-resolution spectral and mel objective."""
import torch
from torch import nn
import librosa

class SpectralLoss(nn.Module):
    def __init__(self,sr=24000):
        super().__init__()
        for size in (256,512,1024,2048):self.register_buffer(f'window{size}',torch.hann_window(size))
        self.register_buffer('mel',torch.tensor(librosa.filters.mel(sr=sr,n_fft=1024,n_mels=80)))
    def forward(self,pred,target):
        logs=[];convergence=[];mel=None
        for size in (256,512,1024,2048):
            w=getattr(self,f'window{size}')
            a=torch.stft(pred,size,size//4,window=w,return_complex=True,pad_mode='constant').abs().clamp_min(1e-5)
            b=torch.stft(target,size,size//4,window=w,return_complex=True,pad_mode='constant').abs().clamp_min(1e-5)
            logs.append((a.log()-b.log()).abs().mean())
            convergence.append((a-b).norm()/b.norm().clamp_min(1e-5))
            if size==1024:mel=((self.mel@a).clamp_min(1e-5).log()-(self.mel@b).clamp_min(1e-5).log()).abs().mean()
        parts=dict(log_stft=torch.stack(logs).mean(),spectral_convergence=torch.stack(convergence).mean(),mel=mel)
        return sum(parts.values()),parts
