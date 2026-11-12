"""JSONL input helpers for zero-shot synthesis and training."""

import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

SR = 24000
HOP = 240
FRAME_SECONDS = HOP / SR


def resample(waveform, sample_rate, target_rate=SR):
    values = np.asarray(waveform, dtype=np.float32)
    if sample_rate == target_rate:
        return values
    divisor = math.gcd(int(sample_rate), int(target_rate))
    return resample_poly(values, target_rate // divisor,
                         sample_rate // divisor).astype(np.float32)


def read_f0(value, base_dir):
    if isinstance(value, (list, tuple, np.ndarray)):
        result = np.asarray(value, dtype=np.float32)
    elif isinstance(value, str):
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = Path(base_dir) / path
        result = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
    else:
        raise TypeError("f0 must be a numeric array or an npy path")
    result = result.reshape(-1)
    if result.size == 0 or not np.isfinite(result).all() or np.any(result < 0):
        raise ValueError("f0 must contain finite nonnegative values")
    return result


def phone_features(phonemes, vocab, f0, durations=None, frames=None):
    phones = list(phonemes)
    if not phones:
        raise ValueError("phonemes must not be empty")
    unknown = sorted(set(phones) - set(vocab))
    if unknown:
        raise ValueError(f"unknown phonemes: {unknown}")
    pitch = np.asarray(f0, dtype=np.float32).reshape(-1)
    if pitch.size == 0 or not np.isfinite(pitch).all() or np.any(pitch < 0):
        raise ValueError("f0 must contain finite nonnegative values")
    if (durations is None) == (frames is None):
        raise ValueError("provide exactly one of durations or frames")

    if frames is not None:
        counts = np.asarray(frames, dtype=np.int64).reshape(-1)
        if len(counts) != len(phones) or np.any(counts < 0):
            raise ValueError("frames must give one nonnegative count per phoneme")
        ends = np.cumsum(counts)
        indices = np.searchsorted(ends, np.arange(len(pitch)), side="right")
        starts = np.concatenate(([0], ends[:-1]))
        safe = np.minimum(indices, len(phones) - 1)
        spans = np.maximum(counts[safe], 1)
        pos = np.clip((np.arange(len(pitch)) - starts[safe]) / spans, 0, 1)
        active = indices < len(phones)
    else:
        seconds = np.asarray(durations, dtype=np.float64).reshape(-1)
        if len(seconds) != len(phones) or not np.isfinite(seconds).all() or np.any(seconds < 0):
            raise ValueError("durations must give one finite nonnegative value per phoneme")
        ends = np.cumsum(seconds)
        starts = ends - seconds
        times = np.arange(len(pitch), dtype=np.float64) * FRAME_SECONDS
        indices = np.searchsorted(ends, times, side="right")
        safe = np.minimum(indices, len(phones) - 1)
        spans = np.maximum(seconds[safe], FRAME_SECONDS)
        pos = np.clip((times - starts[safe]) / spans, 0, 1)
        active = indices < len(phones)

    phone_ids = np.asarray([vocab[phone] for phone in phones], dtype=np.int64)
    pause_id = vocab.get("pau", vocab.get("Edge", 0))
    safe = np.minimum(indices, len(phones) - 1)
    context = np.stack([
        phone_ids[np.clip(safe + offset, 0, len(phones) - 1)]
        for offset in (-1, 0, 1)
    ], axis=-1)
    context[~active] = pause_id
    pos = np.asarray(pos, dtype=np.float32)
    pos[~active] = 0
    return context, pos, pitch


def read_score(path, vocab):
    score_path = Path(path).expanduser()
    value = json.loads(score_path.read_text())
    f0 = read_f0(value["f0"], score_path.parent)
    return phone_features(value["phonemes"], vocab, f0,
                          durations=value.get("durations"), frames=value.get("frames"))


def load_embedding(value, base_dir):
    if value is None:
        return None
    if isinstance(value, str):
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = Path(base_dir) / path
        return np.asarray(np.load(path, allow_pickle=False), dtype=np.float32).reshape(-1)
    return np.asarray(value, dtype=np.float32).reshape(-1)


def load_example(row, vocab, base_dir):
    required = {"wav", "phonemes", "f0", "speaker"}
    missing = required - set(row)
    if missing:
        raise ValueError(f"missing JSONL keys: {sorted(missing)}")
    if not isinstance(row["speaker"], str) or not row["speaker"]:
        raise ValueError("speaker must be a nonempty string")
    if ("durations" in row) == ("frames" in row):
        raise ValueError("provide exactly one of durations or frames")

    wav_path = Path(row["wav"]).expanduser()
    if not wav_path.is_absolute():
        wav_path = Path(base_dir) / wav_path
    audio, sample_rate = sf.read(wav_path, dtype="float32", always_2d=True)
    mono = resample(audio.mean(axis=1), sample_rate)
    f0 = read_f0(row["f0"], base_dir)
    context, pos, f0 = phone_features(
        row["phonemes"], vocab, f0,
        durations=row.get("durations"), frames=row.get("frames"),
    )
    samples = len(f0) * HOP
    if len(mono) < samples:
        mono = np.pad(mono, (0, samples - len(mono)))
    else:
        mono = mono[:samples]
    embedding = load_embedding(row.get("embedding"), base_dir)
    return {
        "context": context,
        "pos": pos,
        "f0": f0,
        "y": np.asarray(mono, dtype=np.float32),
        "embedding": embedding,
        "speaker": row["speaker"],
    }


def read_jsonl(path):
    jsonl_path = Path(path).expanduser()
    rows = [json.loads(line) for line in jsonl_path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("input must contain at least one JSON object")
    return rows, jsonl_path.parent
