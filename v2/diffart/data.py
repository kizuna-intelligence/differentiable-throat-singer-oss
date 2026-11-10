"""Inference frame grid; no database preparation or corpus inventory."""
from pathlib import Path
import numpy as np
SR,HOP=24000,240
ROOT=Path(__file__).resolve().parents[1]

def frame_labels(rows, n, vocab):
    times = np.arange(n) * HOP / SR
    end = np.array([b for a, b, p in rows])
    idx = np.searchsorted(end, times, side='right')
    idx = np.clip(idx, 0, len(rows) - 1)
    phones = np.array([vocab[p] for a, b, p in rows])
    context = np.stack([phones[np.clip(idx + d, 0, len(rows) - 1)] for d in (-1, 0, 1)], -1)
    start = np.array([a for a, b, p in rows])
    dur = end - start
    pos = np.clip((times - start[idx]) / dur[idx], 0, 1)
    outside = (times < rows[0][0]) | (times >= rows[-1][1])
    context[outside] = vocab['pau']
    pos[outside] = 0
    return (context.astype(np.int64), pos.astype(np.float32))
