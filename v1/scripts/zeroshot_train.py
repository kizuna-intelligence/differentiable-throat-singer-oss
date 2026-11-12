"""Train a zero-shot throat singer from JSONL examples."""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diffart.losses import SpectralLoss
from diffart.zeroshot.data import HOP, SR, load_example, read_jsonl
from diffart.zeroshot.model import BODY_NAMES, ZeroShotSynthesizer, contrastive
from diffart.zeroshot.tone import ToneColour


def package_config(vocab, embedding_dim):
    return {
        "vocab": vocab,
        "embedding_dim": embedding_dim,
        "encoder": "openvoice",
        "body_parameters": list(BODY_NAMES),
    }


def train(rows, base_dir, vocab, embedding_dim, device="cpu", steps=1000,
          batch_size=4, segment_frames=100, learning_rate=5e-4,
          openvoice_root=None, converter_dir="checkpoints_v2/converter"):
    examples = [load_example(row, vocab, base_dir) for row in rows]
    if any(example["embedding"] is None for example in examples):
        if openvoice_root is None:
            raise ValueError("examples without embedding require OPENVOICE_ROOT or --openvoice-root")
        root = Path(openvoice_root).expanduser()
        converter = Path(converter_dir).expanduser()
        if not converter.is_absolute():
            converter = root / converter
        encoder = ToneColour(converter, device, root)
        for example in examples:
            if example["embedding"] is None:
                example["embedding"] = encoder.audio(example["y"], SR).reshape(-1)
    for example in examples:
        if len(example["embedding"]) != embedding_dim:
            raise ValueError("embedding dimension does not match the model configuration")

    torch.manual_seed(0)
    random = np.random.default_rng(0)
    model = ZeroShotSynthesizer(len(vocab), embedding_dim).to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = SpectralLoss().to(device)
    for step in range(steps):
        order = random.choice(len(examples), size=min(batch_size, len(examples)), replace=False)
        optimizer.zero_grad(set_to_none=True)
        embeddings = []
        speakers = []
        total = 0.0
        for index in order:
            example = examples[int(index)]
            frames = len(example["f0"])
            count = min(segment_frames, frames)
            start = int(random.integers(frames - count + 1))
            end = start + count
            context = torch.as_tensor(example["context"][start:end], device=device)
            pos = torch.as_tensor(example["pos"][start:end], device=device)
            f0 = torch.as_tensor(example["f0"][start:end], device=device)
            target = torch.as_tensor(example["y"][start * HOP:end * HOP], device=device)
            embedding = torch.as_tensor(example["embedding"], dtype=torch.float32, device=device)
            prediction, controls = model(context, pos, f0, embedding,
                                         seed=step * batch_size + int(index),
                                         return_controls=True)
            loss, _ = criterion(prediction, target)
            loss = loss + model.regularization(controls)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite optimization loss")
            (loss / len(order)).backward()
            embeddings.append(embedding)
            speakers.append(example["speaker"])
            total += float(loss.detach()) / len(order)
        if len(embeddings) > 1:
            auxiliary = contrastive(model, torch.stack(embeddings), torch.stack(embeddings), speakers)
            (0.05 * auxiliary).backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        if not torch.isfinite(norm):
            raise FloatingPointError("nonfinite gradient")
        optimizer.step()
        model.project()
        print(f"step {step + 1}/{steps} loss={total:.6f}", flush=True)
    model.eval()
    return model, package_config(vocab, embedding_dim)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--segment-frames", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--openvoice-root", default=os.environ.get("OPENVOICE_ROOT"),
                        help="OpenVoice checkout; may also be set with OPENVOICE_ROOT")
    parser.add_argument("--converter-dir", default="checkpoints_v2/converter")
    args = parser.parse_args(argv)
    if min(args.steps, args.batch_size, args.segment_frames) < 1:
        parser.error("steps, batch-size, and segment-frames must be positive")
    if args.learning_rate <= 0:
        parser.error("learning-rate must be positive")

    torch.set_num_threads(2)
    repo = Path(__file__).resolve().parents[1]
    bundled = torch.load(repo / "models/zeroshot_v1.pt", map_location="cpu", weights_only=True)
    vocab = bundled["config"]["vocab"]
    embedding_dim = bundled["config"]["embedding_dim"]

    rows, base_dir = read_jsonl(args.jsonl)
    model, config = train(
        rows, base_dir, vocab, embedding_dim, device=args.device, steps=args.steps,
        batch_size=args.batch_size, segment_frames=args.segment_frames,
        learning_rate=args.learning_rate, openvoice_root=args.openvoice_root,
        converter_dir=args.converter_dir,
    )
    output = Path(args.out).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": config}, output)


if __name__ == "__main__":
    main()
