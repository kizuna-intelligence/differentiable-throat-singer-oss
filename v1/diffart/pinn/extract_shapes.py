"""Read the frozen main model ONLY to extract training-frame area functions.

Run: python -m diffart.pinn.extract_shapes
No audio rendering, acoustic transfer calculation, or source-model fitting.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / 'results' / 'shapes'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def extract(out=OUT, rank=6, stride=10):
    # This is the only PINN module importing the frozen acoustic implementation.
    from diffart.physics import Synthesizer

    torch.set_num_threads(2)
    checkpoint = ROOT / 'artifacts/main/best.pt'
    physics = ROOT / 'diffart/physics.py'
    before = {str(p.relative_to(ROOT)): digest(p) for p in (checkpoint, physics)}
    manifest = json.loads((ROOT / 'data/manifest.json').read_text())
    model = Synthesizer(len(manifest['vocab'])).eval().requires_grad_(False)
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True)['model'])
    areas, labels, provenance = [], [], []
    with torch.inference_mode():
        for song in manifest['songs']:
            seconds = song.get('train_seconds', 0)
            if song['split'] != 'train' or seconds <= 0:
                continue
            feature = ROOT / 'data/features' / (song['id'] + '.npz')
            pitch = ROOT / 'data/f0' / (song['id'] + '.npz')
            with np.load(feature) as d, np.load(pitch) as f:
                n = min(int(seconds * 100), len(d['context']), len(f['f0']))
                ix = np.arange(0, n, stride)
                context = torch.as_tensor(d['context'][ix])
                controls = model.controls(context, torch.as_tensor(d['pos'][ix]),
                                          torch.as_tensor(f['f0'][ix]), model.speaker)
                areas.append(model.speaker.area(controls['art']).numpy())
                labels.append(context[:, 1].numpy())
            provenance.append(dict(song=song['id'], frames=len(ix),
                                   feature_sha256=digest(feature), f0_sha256=digest(pitch)))
        length = float(model.speaker.length)
    a = np.concatenate(areas).astype(np.float64)
    phone = np.concatenate(labels)
    # Log-area PCA preserves positivity; cm² -> m² happens before PCA.
    log_a = np.log(a * 1e-4)
    mean = log_a.mean(0)
    _, singular, vt = np.linalg.svd(log_a - mean, full_matrices=False)
    basis = vt[:rank]
    coeff = (log_a - mean) @ basis.T
    variance = singular**2 / np.sum(singular**2)
    means = {name: a[phone == ident].mean(0).tolist()
             for name, ident in manifest['vocab'].items() if np.any(phone == ident)}
    for p in (checkpoint, physics):
        if digest(p) != before[str(p.relative_to(ROOT))]:
            raise RuntimeError('Frozen input changed during extraction: ' + str(p))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / 'area_pca.npz', log_area_mean_m2=mean, basis=basis,
                        coefficients=coeff, phone_ids=phone, area_cm2=a,
                        length_m=length, explained_variance_ratio=variance[:rank])
    radius = np.sqrt(a * 1e-4 / np.pi)
    summary = dict(source_sha256=before, manifest_sha256=digest(ROOT / 'data/manifest.json'),
                   training_songs=provenance, frames=len(a), stride_frames=stride,
                   length_m=length, pca_rank=rank,
                   pca_explained_variance=float(variance[:rank].sum()),
                   area_cm2_range=[float(a.min()), float(a.max())],
                   radius_m_range=[float(radius.min()), float(radius.max())],
                   vowel_mean_area_cm2={v: means.get(v) for v in ('a', 'i', 'u', 'e', 'o')},
                   phone_mean_area_cm2=means,
                   interpretation='Fitted area functions, not measured anatomy; train split only.')
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: summary[k] for k in ('frames', 'length_m', 'pca_explained_variance',
                                             'area_cm2_range', 'radius_m_range')}))
    return summary


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=OUT)
    ap.add_argument('--rank', type=int, default=6)
    ap.add_argument('--stride', type=int, default=10)
    args = ap.parse_args()
    extract(args.out, args.rank, args.stride)
