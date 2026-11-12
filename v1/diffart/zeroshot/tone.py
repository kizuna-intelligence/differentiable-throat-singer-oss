"""OpenVoice v2 reference tone-colour embedding."""

import json
import sys
from pathlib import Path

import librosa
import numpy as np
import torch


class ToneColour:
    def __init__(self, converter_dir, device="cpu", openvoice_root=None):
        if openvoice_root is None:
            raise ValueError("openvoice_root is required")
        root = str(Path(openvoice_root).expanduser())
        if root not in sys.path:
            sys.path.insert(0, root)
        from openvoice.mel_processing import spectrogram_torch
        from openvoice.models import ReferenceEncoder

        folder = Path(converter_dir).expanduser()
        cfg = json.loads((folder / "config.json").read_text())
        self.data = cfg["data"]
        self.device = device
        self.spectrogram = spectrogram_torch
        self.encoder = ReferenceEncoder(
            self.data["filter_length"] // 2 + 1,
            cfg["model"]["gin_channels"],
        ).to(device).eval()
        state = torch.load(folder / "checkpoint.pth", map_location="cpu", weights_only=False)["model"]
        selected = {key[len("ref_enc."):]: value for key, value in state.items()
                    if key.startswith("ref_enc.")}
        self.encoder.load_state_dict(selected)
        self.encoder.requires_grad_(False)

    @torch.inference_mode()
    def __call__(self, wav_44k):
        waveform = librosa.resample(
            np.asarray(wav_44k, dtype=np.float32), orig_sr=44100,
            target_sr=self.data["sampling_rate"],
        )
        waveform = torch.from_numpy(waveform)[None].to(self.device)
        spec = self.spectrogram(
            waveform, self.data["filter_length"], self.data["sampling_rate"],
            self.data["hop_length"], self.data["win_length"], center=False,
        )
        return self.encoder(spec.transpose(1, 2))[0].cpu().numpy()

    def audio(self, wav, sample_rate):
        signal = np.asarray(wav, dtype=np.float32)
        if sample_rate != 44100:
            signal = librosa.resample(signal, orig_sr=sample_rate, target_sr=44100)
        return self(signal)
