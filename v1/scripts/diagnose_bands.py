"""Read-only held-out reproduction diagnosis (DESIGN_pinn_and_fix.md A).

Run: PYTHONPATH=. CUDA_VISIBLE_DEVICES='' .venv/bin/python scripts/diagnose_bands.py
Only artifacts/diagnosis is written. No Assessment (which writes references),
EcapaMetrics (which writes a shared model cache), or evaluation entrypoint is used.
"""
import argparse
import csv
import json
import os
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import numpy as np
import soundfile as sf
from scipy import signal
from scipy.ndimage import gaussian_filter1d
import torch
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from diffart.data import ROOT, SR, HOP, sha
from diffart.dataset import Corpus
from diffart.physics import Synthesizer, lf_flow, transfer
from diffart.metrics import SpeakerMetrics
from diffart.runtime import save_json

BANDS = [(0, 1000), (1000, 2000), (2000, 4000), (4000, 6000),
         (6000, 8000), (8000, 12000)]
NFFT = 2048
FREQ = np.fft.rfftfreq(NFFT, 1 / SR)
VOWELS = ['a', 'i', 'u', 'e', 'o']


def band_name(lo, hi):
    return f'{lo // 1000}-{hi // 1000}k'


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(rows)


def stft(y):
    return librosa.stft(y, n_fft=NFFT, hop_length=HOP, pad_mode='constant')


def istft(z, length):
    return librosa.istft(z, hop_length=HOP, length=length).astype(np.float32)


def spectrum_diagnosis(song, d, y, out):
    n = len(d['f0'])
    ref = d['y'][:n * HOP]
    zr, zs = stft(ref)[:, :n], stft(y)[:, :n]
    rms = np.sqrt(np.mean(np.abs(zr) ** 2, axis=0) / NFFT)
    # Reference-only activity selection, excluding labeled silences, including br.
    active = (d['context'][:, 1] != song['pause_id']) & (rms > max(rms.max() * .01, 1e-6))
    gain = np.sqrt(np.sum(abs(zr[:, active]) ** 2) / np.sum(abs(zs[:, active]) ** 2))
    floor = abs(zr).max() * 1e-5
    rows, curves = [], {}
    for kind, mask in [('voiced', active & (d['f0'] > 0)), ('unvoiced', active & (d['f0'] == 0))]:
        if not mask.any():
            continue
        curves[kind] = {}
        for mode, g in [('raw', 1.), ('rms_matched', gain)]:
            r, s = abs(zr[:, mask]), abs(zs[:, mask]) * g
            dbdiff = 20 * np.log10(np.maximum(s, floor) / np.maximum(r, floor))
            ltas_r = 10 * np.log10(np.maximum(np.mean(r ** 2, axis=1), floor ** 2))
            ltas_s = 10 * np.log10(np.maximum(np.mean(s ** 2, axis=1), floor ** 2))
            curves[kind][mode] = (ltas_r, ltas_s)
            for lo, hi in BANDS:
                ix = (FREQ >= lo) & (FREQ < hi)
                rows.append(dict(song=song['id'], voicing=kind, normalization=mode,
                    band=band_name(lo, hi), frames=int(mask.sum()),
                    lsd_db=float(np.sqrt(np.mean(dbdiff[ix] ** 2, axis=0)).mean()),
                    ltas_bias_db=float((ltas_s[ix] - ltas_r[ix]).mean()),
                    ltas_distance_db=float(np.sqrt(np.mean((ltas_s[ix] - ltas_r[ix]) ** 2))),
                    synth_to_reference_gain_db=float(20 * np.log10(gain))))
    arrays = {f'{v}_{mode}_{name}': pair[i] for v, modes in curves.items()
              for mode, pair in modes.items() for i, name in enumerate(['real', 'synth'])}
    np.savez_compressed(out / f"{song['id']}__ltas.npz", frequency_hz=FREQ, **arrays)
    return rows, curves, active, gain


def band_weight(lo, hi):
    # Transition is wholly INSIDE each band: 8-12 kHz cannot leak below 8 kHz.
    w = ((FREQ >= lo) & (FREQ < hi)).astype(float)
    if lo:
        ix = (FREQ >= lo) & (FREQ < lo + 100)
        w[ix] *= .5 - .5 * np.cos(np.pi * (FREQ[ix] - lo) / 100)
    if hi < SR / 2:
        ix = (FREQ >= hi - 100) & (FREQ < hi)
        w[ix] *= .5 - .5 * np.cos(np.pi * (hi - FREQ[ix]) / 100)
    return w


def select_clips(active, count, seconds):
    frames = round(seconds * SR / HOP)
    starts = np.arange(0, len(active) - frames + 1, 100)
    occupancy = np.array([active[a:a + frames].mean() for a in starts])
    candidates = starts[occupancy >= .6]
    if len(candidates) < count:
        raise ValueError('Not enough reference-active clip candidates')
    # Quantiles in time, fixed before any speaker metric is computed.
    picked = candidates[np.round(np.linspace(0, len(candidates) - 1, count)).astype(int)]
    if count > 1 and np.min(np.diff(picked)) < frames:
        raise ValueError('Selected clips overlap')
    return [(int(a), int(a + frames)) for a in picked]


class ReadOnlyEcapa:
    def __init__(self, out):
        import torchaudio
        if not hasattr(torchaudio, 'list_audio_backends'):
            torchaudio.list_audio_backends = lambda: ['soundfile']
        from speechbrain.inference.speaker import EncoderClassifier
        source = ROOT / 'artifacts/models/ecapa'
        # Fetching/symlinks are confined to our directory; shared source is read-only.
        self.model = EncoderClassifier.from_hparams(source=str(source),
            savedir=str(out / 'ecapa_cache'), overrides={'pretrained_path': str(source)},
            run_opts={'device': 'cpu'})

    @torch.inference_mode()
    def embedding(self, y):
        e = self.model.encode_batch(torch.tensor(y.copy())[None]).flatten()
        return torch.nn.functional.normalize(e, dim=0).cpu().numpy()


def interventions(song, d, y, clips, out, metrics, anchors):
    rows, checks = [], []
    for ci, (a, b) in enumerate(clips):
        r = d['y'][a * HOP:b * HOP]
        s = y[a * HOP:b * HOP]
        zr, zs = stft(r), stft(s)
        floor = max(abs(zr).max(), abs(zs).max()) * 1e-5
        # 150 Hz Gaussian smoothing of log magnitude. Donor phase/fine structure
        # is never copied. Bound ratio to +/-30 dB to avoid near-zero explosions.
        er = gaussian_filter1d(np.log(np.maximum(abs(zr), floor)), 150 / (SR / NFFT), axis=0)
        es = gaussian_filter1d(np.log(np.maximum(abs(zs), floor)), 150 / (SR / NFFT), axis=0)
        raw_delta = er - es
        delta = np.clip(raw_delta, -np.log(10) * 30 / 20, np.log(10) * 30 / 20)
        variants = {'synth': s, 'recording': r, 'roundtrip': istft(zs, len(s))}
        for lo, hi in BANDS:
            variants[band_name(lo, hi)] = istft(zs * np.exp(delta * band_weight(lo, hi)[:, None]), len(s))
        variants['all_envelope'] = istft(zs * np.exp(delta), len(s))
        # Scalar-gain control. All intervention clips are RMS-matched to synth.
        variants['gain_only'] = s * (np.sqrt(np.mean(r ** 2)) / np.sqrt(np.mean(s ** 2)))
        target_embeddings = {name: metric.embedding(signal.resample_poly(r, 2, 3)) for name, metric in metrics.items()}
        baseline = {}
        for condition, audio in variants.items():
            if condition not in ['synth', 'recording', 'roundtrip', 'gain_only']:
                audio *= np.sqrt(np.mean(s ** 2) / max(np.mean(audio ** 2), 1e-20))
            assert np.isfinite(audio).all()
            y16 = signal.resample_poly(audio, 2, 3).astype(np.float32)
            row = dict(song=song['id'], clip=ci, start_seconds=a * .01, end_seconds=b * .01,
                       condition=condition, envelope_ratio_clipped_fraction=float(np.mean(raw_delta != delta)))
            for name, metric in metrics.items():
                emb = target_embeddings[name] if condition == 'recording' else metric.embedding(y16)
                row[name + '_anchor_cosine'] = float((anchors[name] @ emb).mean())
                row[name + '_paired_cosine'] = float(target_embeddings[name] @ emb)
                if condition == 'synth':
                    baseline[name] = (row[name + '_anchor_cosine'], row[name + '_paired_cosine'])
                row[name + '_anchor_delta'] = row[name + '_anchor_cosine'] - baseline[name][0]
                row[name + '_paired_delta'] = row[name + '_paired_cosine'] - baseline[name][1]
            rows.append(row)
            sf.write(out / 'audio' / f"{song['id']}__clip{ci}__{condition}.wav", audio, SR, subtype='FLOAT')
        checks.append(dict(song=song['id'], clip=ci,
            roundtrip_relative_rms=float(np.sqrt(np.mean((variants['roundtrip'] - s) ** 2) / np.mean(s ** 2))),
            high_band_16k_relative_rms=float(np.sqrt(np.mean((signal.resample_poly(variants['8-12k'], 2, 3) - signal.resample_poly(s, 2, 3)) ** 2) / np.mean(signal.resample_poly(s, 2, 3) ** 2)))))
        print('interventions', song['id'], ci, flush=True)
    return rows, checks


def lpc_poles(y, order, sr=16000):
    x = signal.lfilter([1, -.97], [1], y)
    a = librosa.lpc(x * np.hanning(len(x)), order=order)
    roots = np.roots(a)
    roots = roots[np.imag(roots) > 0]
    f = np.angle(roots) * sr / (2 * np.pi)
    bw = -sr / np.pi * np.log(np.maximum(abs(roots), 1e-12))
    ix = (f > 90) & (f < 5500) & (bw > 10) & (bw < 1500)
    sort = np.argsort(f[ix])
    return a, np.stack([f[ix][sort], bw[ix][sort]], axis=1)[:4]


def source_features(flow, f0, sr=16000):
    # Acoustic proxies, NOT physiological open quotient or absolute airflow.
    x = np.asarray(flow, float)
    x = x - x.mean()
    if abs(np.gradient(x).max()) > abs(np.gradient(x).min()):
        x = -x
    p = sr / f0
    peaks, _ = signal.find_peaks(-np.gradient(x), distance=max(1, int(.65 * p)), prominence=np.std(np.gradient(x)) * .25)
    cycles = []
    for a, b in zip(peaks[:-1], peaks[1:]):
        if .8 * p < b - a < 1.2 * p:
            c = np.interp(np.arange(128) / 128, np.arange(b - a) / (b - a), x[a:b])
            c = (c - c.min()) / max(np.ptp(c), 1e-12)
            cycles.append(c)
    if len(cycles) < 3:
        return None, None
    cycles = np.stack(cycles)
    cycle = cycles.mean(axis=0)
    cycle = (cycle - cycle.min()) / max(np.ptp(cycle), 1e-12)
    spectrum = abs(np.fft.rfft(cycle))
    harmonic = np.arange(1, min(int(4000 / f0), 20, len(spectrum) - 1) + 1)
    slope = np.polyfit(np.log2(harmonic * f0), 20 * np.log10(np.maximum(spectrum[harmonic], 1e-8)), 1)[0]
    # Align maximum flow, then measure time until falling below 10%.
    peak = int(np.argmax(cycle))
    after = np.roll(cycle, -peak)
    below = np.flatnonzero(after < .1)
    return dict(open_fraction_10pct=float(np.mean(cycle > .1)),
        return_fraction_10pct=float(below[0] / 128) if len(below) else 1.,
        return_sharpness=float(-np.min(np.diff(np.r_[cycle, cycle[0]])) * 128),
        spectral_tilt_db_oct=float(slope),
        cycle_variation_ratio=float(np.mean((cycles - cycles.mean(axis=0)) ** 2) / max(np.var(cycles), 1e-12)),
        cycles=len(cycles)), cycle


def physical_resonances(model, p, i):
    f = torch.linspace(0, 6000, 2049)
    with torch.inference_mode():
        h = transfer(p['areas'][i], model.speaker.length, model.speaker.wall, f,
                     model.speaker.nose, p['velum'][i])
    db = 20 * np.log10(np.maximum(abs(h.numpy()), 1e-10))
    freq = f.numpy()
    peaks, _ = signal.find_peaks(db, prominence=2., distance=25)
    peaks = [ix for ix in peaks if 90 < freq[ix] < 5500][:4]
    result = []
    for ix in peaks:
        left, right = ix, ix
        while left > 0 and db[left] > db[ix] - 3:
            left -= 1
        while right < len(db) - 1 and db[right] > db[ix] - 3:
            right += 1
        width = None
        if left > 0 and right < len(db) - 1:
            lf = np.interp(db[ix] - 3, db[left:left + 2], freq[left:left + 2])
            rf = np.interp(db[ix] - 3, db[right - 1:right + 1][::-1], freq[right - 1:right + 1][::-1])
            width = float(rf - lf)
        result.append((float(freq[ix]), width))
    return result


@torch.inference_mode()
def vowel_diagnosis(song, d, y, model, vocab, out, per_vowel):
    n = len(d['f0'])
    phones = torch.as_tensor(d['context'])
    p = model.controls(phones, torch.as_tensor(d['pos']), torch.as_tensor(d['f0']), model.speaker)
    p['areas'] = model.speaker.area(p['art'])
    formants, sources, cycles = [], [], []
    for vowel in VOWELS:
        # Central, sustained, voiced 120-ms windows; deterministic time quantiles.
        candidates = []
        for i in range(8, n - 8):
            if d['context'][i, 1] != vocab[vowel] or not .25 < d['pos'][i] < .75:
                continue
            f = d['f0'][i - 6:i + 6]
            if np.min(f) <= 0 or np.std(f) / np.mean(f) > .08:
                continue
            if not np.all(d['context'][i - 6:i + 6, 1] == vocab[vowel]):
                continue
            if not candidates or i - candidates[-1] >= 12:
                candidates.append(i)
        if not candidates:
            continue
        selected = np.array(candidates)[np.unique(np.round(np.linspace(0, len(candidates) - 1, min(per_vowel, len(candidates)))).astype(int))]
        for i in selected:
            i = int(i)
            f0 = float(np.median(d['f0'][i - 6:i + 6]))
            a, b = (i - 6) * HOP, (i + 6) * HOP
            rec = signal.resample_poly(d['y'][a:b], 2, 3)
            syn = signal.resample_poly(y[a:b], 2, 3)
            if np.sqrt(np.mean(rec ** 2)) < 1e-4:
                continue
            physical = physical_resonances(model, p, i)
            signals = {'recording': rec, 'synth_inverse': syn}
            for order in [18, 22]:
                for kind, audio in signals.items():
                    coeff, poles = lpc_poles(audio, order)
                    for j, (frequency, bandwidth) in enumerate(poles):
                        phys = physical[j] if j < len(physical) else (None, None)
                        formants.append(dict(song=song['id'], vowel=vowel, frame=i, f0_hz=f0,
                            high_f0=f0 >= 350, signal=kind, lpc_order=order, formant=j + 1,
                            frequency_hz=float(frequency), bandwidth_hz=float(bandwidth),
                            physical_frequency_hz=phys[0], physical_bandwidth_hz=phys[1]))
                    # LPC fitted to preemphasized audio, applied to un-preemphasized
                    # lips pressure; leaky integration approximates undoing radiation.
                    residual = signal.lfilter(coeff, [1], audio)
                    flow = signal.lfilter([1], [1, -.99], residual)[320:-160]
                    feat, cycle = source_features(flow, f0)
                    if feat:
                        sources.append(dict(song=song['id'], vowel=vowel, frame=i, f0_hz=f0,
                            signal=kind, lpc_order=order, **feat))
                        if order == 18:
                            cycles.append((vowel, kind, cycle))
            rd = p['rd'][i]
            ph = torch.arange(round(16000 * .12)) * f0 / 16000
            flow = lf_flow(ph.remainder(1), rd).numpy()
            feat, cycle = source_features(flow, f0)
            if feat:
                sources.append(dict(song=song['id'], vowel=vowel, frame=i, f0_hz=f0,
                    signal='lf_known', lpc_order=0, rd=float(rd), **feat))
                cycles.append((vowel, 'lf_known', cycle))
        print('vowels', song['id'], vowel, len(selected), flush=True)
    arrays = {f'{v}_{kind}': np.stack([c for vv, kk, c in cycles if vv == v and kk == kind])
              for v in VOWELS for kind in ['recording', 'synth_inverse', 'lf_known']
              if any(vv == v and kk == kind for vv, kk, _ in cycles)}
    np.savez_compressed(out / f"{song['id']}__source_cycles.npz", **arrays)
    return formants, sources


def aggregate_bands(rows):
    summary = []
    for mode in ['raw', 'rms_matched']:
        for voiced in ['voiced', 'unvoiced']:
            for lo, hi in BANDS:
                r = [r for r in rows if r['normalization'] == mode and r['voicing'] == voiced and r['band'] == band_name(lo, hi)]
                if r:
                    summary.append(dict(normalization=mode, voicing=voiced, band=band_name(lo, hi),
                        songs=len(r), frames=sum(x['frames'] for x in r),
                        **{k:float(np.mean([x[k] for x in r])) for k in ['lsd_db', 'ltas_bias_db', 'ltas_distance_db']}))
    return summary


def aggregate_interventions(rows):
    result = []
    for cond in dict.fromkeys(r['condition'] for r in rows):
        r = [x for x in rows if x['condition'] == cond]
        songmeans = [{k: float(np.mean([x[k] for x in r if x['song'] == song]))
                      for k in r[0] if k.startswith(('wavlm_', 'ecapa_'))} for song in sorted({x['song'] for x in r})]
        result.append(dict(condition=cond, songs=len(songmeans), clips=len(r),
            **{k:float(np.mean([x[k] for x in songmeans])) for k in songmeans[0]},
            **{k + '_song_min':float(min(x[k] for x in songmeans)) for k in songmeans[0] if k.endswith('delta')},
            **{k + '_song_max':float(max(x[k] for x in songmeans)) for k in songmeans[0] if k.endswith('delta')}))
    return result


def paired_summary(rows, keys, values, comparison, baseline, order):
    result = []
    for group in sorted({tuple(r[k] for k in keys) for r in rows}):
        r = [x for x in rows if tuple(x[k] for k in keys) == group and x['lpc_order'] == order]
        a = {(x['song'], x['frame']):x for x in r if x['signal'] == comparison}
        b = {(x['song'], x['frame']):x for x in r if x['signal'] == baseline}
        common = a.keys() & b.keys()
        if common:
            row = dict(zip(keys, group), n=len(common), lpc_order=order, comparison=comparison, baseline=baseline)
            for value in values:
                row[value + '_recording_median'] = float(np.median([b[k][value] for k in common]))
                row[value + '_synth_median'] = float(np.median([a[k][value] for k in common]))
                row[value + '_paired_median_delta'] = float(np.median([a[k][value] - b[k][value] for k in common]))
                row[value + '_paired_median_abs_delta'] = float(np.median([abs(a[k][value] - b[k][value]) for k in common]))
            result.append(row)
    return result


def figures(out, spectra, bands, interventions, formants):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, (song, curves) in zip(axes.flat, spectra.items()):
        for kind, color in [('voiced', 'C0'), ('unvoiced', 'C1')]:
            if kind not in curves:
                continue
            r, s = curves[kind]['rms_matched']
            ax.plot(FREQ / 1000, s-r, color=color, label=kind)
        ax.axhline(0, color='k', lw=.5)
        ax.set(title=song, xlabel='Frequency (kHz)', ylabel='LTAS synth - real (dB)', xlim=(0, 12), ylim=(-40, 40))
        ax.legend()
    fig.tight_layout(); fig.savefig(out / 'ltas_difference.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    names = [band_name(*b) for b in BANDS]
    for kind in ['voiced', 'unvoiced']:
        r = [next(x for x in bands if x['normalization']=='rms_matched' and x['voicing']==kind and x['band']==b) for b in names]
        axes[0].plot(names, [x['lsd_db'] for x in r], 'o-', label=kind)
    axes[0].set(ylabel='Log spectral distance (dB)', xlabel='Band'); axes[0].legend()
    for name in ['wavlm', 'ecapa']:
        r = [next(x for x in interventions if x['condition']==b) for b in names]
        means = np.array([x[name+'_anchor_delta'] for x in r])
        lower = [x[name+'_anchor_delta_song_min'] for x in r]
        upper = [x[name+'_anchor_delta_song_max'] for x in r]
        axes[1].errorbar(names, means, yerr=[means-lower, upper-means], marker='o', label=name)
    axes[1].axhline(0, color='k', lw=.5)
    axes[1].set(ylabel='Anchor cosine change (mean; range of 4 songs)', xlabel='Replaced envelope band')
    axes[1].legend(); fig.tight_layout(); fig.savefig(out / 'bands_and_similarity.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for formant in range(1, 5):
        r = [x for x in formants if x['formant']==formant and x['lpc_order']==18]
        r = sorted(r, key=lambda x:VOWELS.index(x['vowel']))
        for ax, value in zip(axes, ['frequency_hz', 'bandwidth_hz']):
            ax.plot([x['vowel'] for x in r], [x[value+'_paired_median_delta'] for x in r], 'o-', label=f'F{formant}')
            ax.set(ylabel=f'Synth - real {value} (Hz)', xlabel='Vowel')
            ax.axhline(0, color='k', lw=.5)
    axes[0].legend(); fig.tight_layout(); fig.savefig(out / 'formant_differences.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 5, figsize=(17, 3))
    for ax, v in zip(axes, VOWELS):
        for kind in ['recording', 'synth_inverse', 'lf_known']:
            allcycles = []
            for path in sorted(out.glob('*__source_cycles.npz')):
                with np.load(path) as z:
                    if f'{v}_{kind}' in z:
                        allcycles.extend(z[f'{v}_{kind}'])
            if allcycles:
                ax.plot(np.arange(128)/128, np.mean(allcycles,axis=0), label=kind)
        ax.set(title=v, xlabel='Cycle phase (closure aligned)')
    axes[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(out / 'glottal_cycles.png', dpi=160); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--clips-per-song', type=int, default=3)
    ap.add_argument('--clip-seconds', type=float, default=6.)
    ap.add_argument('--vowel-windows', type=int, default=20)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0); np.random.seed(0)
    out = ROOT / 'artifacts/diagnosis'; out.mkdir(parents=True, exist_ok=True)
    (out / 'audio').mkdir(exist_ok=True)
    corpus = Corpus(); manifest = corpus.manifest
    songs = [r for r in manifest['songs'] if r['split']=='test']
    assert len(songs) == 4 and all(not r['excluded'] for r in songs)
    ck = ROOT / 'artifacts/main/best.pt'
    ckhash = sha(ck)
    provenance = json.loads((ROOT / 'artifacts/evaluation/provenance.json').read_text())
    assert provenance['checkpoint_sha256'] == ckhash
    assert provenance['manifest_sha256'] == sha(ROOT / 'data/manifest.json')
    inputs = [ck, ROOT/'data/manifest.json', ROOT/'artifacts/evaluation/provenance.json', ROOT/'artifacts/evaluation/references.json']
    refs = json.loads((ROOT / 'artifacts/evaluation/references.json').read_text())
    for ref in refs:
        assert ref['split']=='train' and next(s for s in manifest['songs'] if s['id']==ref['song'])['split']=='train'
        inputs.append(Path(ref['path']))
    for song in songs:
        inputs.extend([ROOT/'artifacts/evaluation'/f"{song['id']}__reproduction_+0.json",
                       ROOT/'listen'/song['id']/'reproduction_+0.wav',
                       ROOT/'data/audio24'/f"{song['id']}.wav",
                       ROOT/'data/features'/f"{song['id']}.npz", ROOT/'data/f0'/f"{song['id']}.npz"])
    before = {str(p.relative_to(ROOT)):sha(p) for p in inputs}
    save_json(out / 'provenance.json', dict(checkpoint_sha256=ckhash, inputs_sha256=before,
        device='cpu', args=vars(args), torch_version=torch.__version__, numpy_version=np.__version__,
        selection='all held-out songs; full-length spectral analysis; time-quantile active clips; central steady vowel windows',
        input_policy='Existing evaluation/model/audio/features read only; diagnostic output only under artifacts/diagnosis',
        speaker_reference='same 3 fixed independent TRAIN anchors as existing evaluation',
        band_metric_limit='16 kHz speaker models do not directly observe 8-12 kHz; resampling leakage and scalar gain may still cause small deltas'))
    model = Synthesizer(len(manifest['vocab']))
    model.load_state_dict(torch.load(ck, map_location='cpu', weights_only=True)['model']); model.eval()
    metrics = {'wavlm': SpeakerMetrics('cpu'), 'ecapa': ReadOnlyEcapa(out)}
    anchors = {}
    for name, metric in metrics.items():
        anchors[name] = np.stack([metric.embedding(signal.resample_poly(sf.read(r['path'], dtype='float32')[0], 2, 3)) for r in refs])
    bands, spectral_curves, swaps, checks, formants, sources, selections = [], {}, [], [], [], [], []
    for song in songs:
        d = corpus.load(song)
        cache = json.loads((ROOT/'artifacts/evaluation'/f"{song['id']}__reproduction_+0.json").read_text())
        assert cache['checkpoint_sha256'] == ckhash and cache['condition']=='reproduction' and cache['model']=='physics'
        y, sr = sf.read(ROOT/'listen'/song['id']/'reproduction_+0.wav', dtype='float32')
        assert sr == SR and len(y)==len(d['f0'])*HOP and np.isfinite(y).all()
        song = dict(song, pause_id=manifest['vocab']['pau'])
        br, curves, active, gain = spectrum_diagnosis(song, d, y, out)
        bands.extend(br); spectral_curves[song['id']] = curves
        clips = select_clips(active, args.clips_per_song, args.clip_seconds)
        selections.append(dict(song=song['id'], total_seconds=len(y)/SR, active_frames=int(active.sum()),
            voiced_active_frames=int((active & (d['f0']>0)).sum()), unvoiced_active_frames=int((active & (d['f0']==0)).sum()),
            rms_match_gain_db=float(20*np.log10(gain)), clips_frames=clips))
        ir, cr = interventions(song, d, y, clips, out, metrics, anchors)
        swaps.extend(ir); checks.extend(cr)
        fr, so = vowel_diagnosis(song, d, y, model, manifest['vocab'], out, args.vowel_windows)
        formants.extend(fr); sources.extend(so)
        for filename, rows in [('bands.csv',bands), ('interventions.csv',swaps), ('formants.csv',formants), ('source.csv',sources)]:
            write_csv(out/filename, rows)
        save_json(out/'selection.json', selections)
    band_summary = aggregate_bands(bands)
    swap_summary = aggregate_interventions(swaps)
    formant_summary = sum([paired_summary(formants, ['vowel','formant'], ['frequency_hz','bandwidth_hz'], 'synth_inverse', 'recording', order) for order in [18,22]], [])
    source_values = ['open_fraction_10pct','return_fraction_10pct','return_sharpness','spectral_tilt_db_oct','cycle_variation_ratio']
    source_summary = sum([paired_summary(sources, ['vowel'], source_values, 'synth_inverse', 'recording', order) for order in [18,22]], [])
    # Calibrate inverse-filter source estimates against the known model LF source.
    calibration = []
    for order in [18,22]:
        inv = {(r['song'],r['frame']):r for r in sources if r['signal']=='synth_inverse' and r['lpc_order']==order}
        known = {(r['song'],r['frame']):r for r in sources if r['signal']=='lf_known'}
        common = inv.keys() & known.keys()
        calibration.append(dict(lpc_order=order, n=len(common),
            **{k+'_median_abs_error':float(np.median([abs(inv[i][k]-known[i][k]) for i in common])) for k in source_values}))
    physical_calibration = []
    for order in [18,22]:
        for j in range(1,5):
            r = [r for r in formants if r['signal']=='synth_inverse' and r['lpc_order']==order and r['formant']==j and r['physical_bandwidth_hz'] is not None]
            physical_calibration.append(dict(lpc_order=order, formant=j, n=len(r),
                frequency_median_abs_error_hz=float(np.median([abs(x['frequency_hz']-x['physical_frequency_hz']) for x in r])),
                bandwidth_median_abs_error_hz=float(np.median([abs(x['bandwidth_hz']-x['physical_bandwidth_hz']) for x in r]))))
    for name, rows in [('band_summary',band_summary),('intervention_summary',swap_summary),('formant_summary',formant_summary),('source_summary',source_summary),('source_calibration',calibration),('formant_calibration',physical_calibration)]:
        write_csv(out/f'{name}.csv',rows)
    summary = dict(bands=band_summary,interventions=swap_summary,formants=formant_summary,
        source=source_summary,source_calibration=calibration,formant_calibration=physical_calibration,
        numerical_checks=checks, vowel_windows=sum(r['signal']=='lf_known' for r in sources),
        high_f0_fraction=float(np.mean([r['f0_hz']>=350 for r in sources if r['signal']=='lf_known'])),
        aggregation='band and speaker metrics: equal weight per song; source/formants: pooled paired vowel-window medians; min/max are song ranges, not confidence intervals')
    save_json(out/'summary.json',summary)
    figures(out,spectral_curves,band_summary,swap_summary,formant_summary)
    after = {str(p.relative_to(ROOT)):sha(p) for p in inputs}
    assert before==after, 'A read-only input changed while diagnosis was running'
    save_json(out/'integrity.json',dict(input_hashes_unchanged=True,checked_files=len(inputs),inputs_sha256_after=after))
    print('DONE', out, flush=True)


if __name__ == '__main__':
    main()
