"""Synthesize a score with a reference tone-colour embedding."""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diffart.zeroshot.data import HOP, SR, read_score, resample
from diffart.zeroshot.model import ZeroShotSynthesizer, fit_body
from diffart.zeroshot.tone import ToneColour
from diffart.losses import SpectralLoss


def synthesize(model, context, pos, f0, embedding, body=None, seed=1234,
               return_controls=False):
    device = next(model.parameters()).device
    inputs = [torch.as_tensor(value, device=device) for value in (context, pos, f0)]
    reference = torch.as_tensor(embedding, dtype=torch.float32, device=device).reshape(-1)
    with torch.inference_mode():
        return model(*inputs, reference, body_override=body, seed=seed,
                     return_controls=return_controls)


def load_model(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if set(checkpoint) != {"model", "config"}:
        raise ValueError("checkpoint must contain model and config only")
    config = checkpoint["config"]
    required = {"vocab", "embedding_dim", "encoder", "body_parameters"}
    if set(config) != required:
        raise ValueError("checkpoint config has an unsupported shape")
    if config["encoder"] != "openvoice":
        raise ValueError("this command expects an OpenVoice embedding")
    model = ZeroShotSynthesizer(len(config["vocab"]), config["embedding_dim"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--reference", required=True, help="reference voice WAV")
    parser.add_argument("--score", required=True, help="phoneme, duration, and F0 JSON")
    parser.add_argument("--out", required=True, help="output WAV path")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--reference-seconds", type=float, default=10.0)
    parser.add_argument("--fit-steps", type=int, default=0)
    parser.add_argument("--reference-score", help="aligned phoneme, duration, and F0 JSON for body fitting")
    parser.add_argument("--save-controls", help="optional control arrays in NPZ format")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--openvoice-root", default=os.environ.get("OPENVOICE_ROOT"),
                        help="OpenVoice checkout; may also be set with OPENVOICE_ROOT")
    parser.add_argument("--converter-dir", default="checkpoints_v2/converter",
                        help="converter directory relative to --openvoice-root")
    args = parser.parse_args(argv)
    if args.openvoice_root is None:
        parser.error("set OPENVOICE_ROOT or provide --openvoice-root")
    if args.fit_steps < 0:
        parser.error("--fit-steps must be nonnegative")
    if args.fit_steps and not args.reference_score:
        parser.error("--fit-steps requires --reference-score aligned to the reference WAV")

    torch.set_num_threads(2)
    model, config = load_model(args.checkpoint, args.device)
    waveform, sample_rate = sf.read(args.reference, dtype="float32", always_2d=True)
    waveform = resample(waveform.mean(axis=1), sample_rate)
    waveform = waveform[:int(args.reference_seconds * SR)]
    if len(waveform) < 3 * SR:
        raise ValueError("reference WAV must contain at least three seconds")
    root = Path(args.openvoice_root).expanduser()
    converter = Path(args.converter_dir).expanduser()
    if not converter.is_absolute():
        converter = root / converter
    encoder = ToneColour(converter, args.device, root)
    embedding = encoder.audio(waveform, SR).reshape(-1)
    context, pos, f0 = read_score(args.score, config["vocab"])

    body = None
    if args.fit_steps:
        ref_context, ref_pos, ref_f0 = read_score(args.reference_score, config["vocab"])
        frame_count = min(len(ref_f0), len(waveform) // HOP)
        if frame_count < 1:
            raise ValueError("reference controls do not overlap the reference WAV")
        aligned = [torch.as_tensor(value[:frame_count], device=args.device)
                   for value in (ref_context, ref_pos, ref_f0)]
        target = torch.as_tensor(waveform[:frame_count * HOP], device=args.device)
        body, _ = fit_body(
            model, *aligned, torch.as_tensor(embedding, device=args.device), target,
            SpectralLoss().to(args.device), steps=args.fit_steps,
        )

    result = synthesize(model, context, pos, f0, embedding, body, args.seed,
                        return_controls=bool(args.save_controls))
    if args.save_controls:
        audio, controls = result
    else:
        audio = result
        controls = None
    output = Path(args.out).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output, audio.detach().cpu().numpy(), SR, subtype="FLOAT")
    if controls is not None:
        control_path = Path(args.save_controls).expanduser()
        control_path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {key: value.detach().cpu().numpy() for key, value in controls.items()}
        arrays.update(context=context, pos=pos, f0=f0)
        arrays["length_cm"] = np.asarray(float(model.physics.speaker.length.detach()) * 100,
                                         dtype=np.float32)
        arrays["nose_area"] = model.physics.speaker.nose.detach().cpu().numpy()
        np.savez_compressed(control_path, **arrays)


if __name__ == "__main__":
    main()
