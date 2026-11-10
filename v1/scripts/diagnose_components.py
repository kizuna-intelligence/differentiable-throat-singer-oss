"""Small controlled path ablations, output confined to artifacts/diagnosis.

Each song's preselected middle 6-s clip; regenerate all conditions with identical
seed/F0/chunking. This diagnoses existing paths, not the effect of adding parts.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import json
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import torch
from diffart.data import ROOT, SR, HOP, sha
from diffart.dataset import Corpus
from diffart.physics import Synthesizer
from diffart.render import render
from diffart.metrics import SpeakerMetrics
from diffart.runtime import save_json
from diagnose_bands import ReadOnlyEcapa, BANDS, FREQ, stft, write_csv, band_name


def main():
    torch.set_num_threads(2)
    out = ROOT/'artifacts/diagnosis'
    corpus = Corpus()
    model = Synthesizer(len(corpus.manifest['vocab']))
    model.load_state_dict(torch.load(ROOT/'artifacts/main/best.pt', map_location='cpu', weights_only=True)['model'])
    model.eval()
    # Reuse cached local models/diagnostic cache without calling shared evaluators.
    metrics = {'wavlm':SpeakerMetrics('cpu'), 'ecapa':ReadOnlyEcapa(out)}
    refs = json.loads((ROOT/'artifacts/evaluation/references.json').read_text())
    anchors = {name:np.stack([metric.embedding(resample_poly(sf.read(r['path'],dtype='float32')[0],2,3)) for r in refs]) for name,metric in metrics.items()}
    conditions = {'full':None, 'no_breath':'breath', 'no_turbulence':'turbulence', 'closed_nose':'velum'}
    selections = json.loads((out/'selection.json').read_text())
    rows, path_bands = [], []
    for selected in selections:
        song = next(s for s in corpus.manifest['songs'] if s['id']==selected['song'])
        assert song['split']=='test'
        a,b = selected['clips_frames'][len(selected['clips_frames'])//2]
        d = corpus.load(song)
        inp = [d[k][a:b] for k in ['context','pos','f0']]
        ref = d['y'][a*HOP:b*HOP]
        zr = stft(ref)
        # Reference-only activity; local 10-ms frames.
        active = (inp[0][:,1] != corpus.manifest['vocab']['pau'])
        active &= np.sqrt(np.mean(abs(zr[:,:b-a])**2,axis=0)/2048) > max(1e-6,np.sqrt(np.mean(abs(zr)**2,axis=0)/2048).max()*.01)
        base, base_scores = None, {}
        for condition, control in conditions.items():
            hook = None
            if control:
                def mute(module, inputs, output):
                    return {**output, control:torch.zeros_like(output[control])}
                hook = model.controls.register_forward_hook(mute)
            try:
                y = render(model,*inp,device='cpu')
            finally:
                if hook:
                    hook.remove()
            if condition=='full':
                base = y
            row = dict(song=song['id'],condition=condition,start_seconds=a*.01,end_seconds=b*.01)
            for name,metric in metrics.items():
                emb = metric.embedding(resample_poly(y,2,3))
                score = float((anchors[name]@emb).mean())
                if condition=='full':
                    base_scores[name] = score
                row[name+'_anchor_cosine'] = score
                row[name+'_anchor_delta'] = score-base_scores[name]
            rows.append(row)
            if condition!='full':
                zd, zb, zy = stft(base-y)[:,:b-a],stft(base)[:,:b-a],stft(y)[:,:b-a]
                floor = abs(zr).max()*1e-5
                for voice, mask in [('voiced',active & (inp[2]>0)),('unvoiced',active & (inp[2]==0))]:
                    if not mask.any():
                        continue
                    for lo,hi in BANDS:
                        ix = (FREQ>=lo)&(FREQ<hi)
                        delta,full = abs(zd[ix][:,mask])**2,abs(zb[ix][:,mask])**2
                        reference = abs(zr[ix,:b-a][:,mask])
                        full_lsd = np.sqrt(np.mean((20*np.log10(np.maximum(abs(zb[ix][:,mask]),floor)/np.maximum(reference,floor)))**2,axis=0)).mean()
                        ablated_lsd = np.sqrt(np.mean((20*np.log10(np.maximum(abs(zy[ix][:,mask]),floor)/np.maximum(reference,floor)))**2,axis=0)).mean()
                        path_bands.append(dict(song=song['id'],condition=condition,voicing=voice,band=band_name(lo,hi),frames=int(mask.sum()),
                            path_change_rms_relative_db=float(10*np.log10(max(np.mean(delta),1e-30)/max(np.mean(full),1e-30))),
                            full_lsd_db=float(full_lsd),ablated_lsd_db=float(ablated_lsd),lsd_change_db=float(ablated_lsd-full_lsd)))
            sf.write(out/'audio'/f"{song['id']}__components__{condition}.wav",y,SR,subtype='FLOAT')
            print('components',song['id'],condition,flush=True)
        write_csv(out/'components.csv',rows)
        write_csv(out/'component_bands.csv',path_bands)
    save_json(out/'components.json',dict(checkpoint_sha256=sha(ROOT/'artifacts/main/best.pt'),
        method='Middle preselected 6-s clip per held-out song; deterministic re-render with only breath, turbulence, or velum zeroed by a temporary forward hook; compare to SAME re-render full, not cached full-song waveform. Local phase origin/chunk boundaries differ from cached evaluation.',
        caveat='Ablation is a test of the existing route, not evidence about missing branches or an optimized replacement. Nasal closure changes all paths; component RMS ratios are not additive. LSD is raw-gain and time aligned.',
        rows=rows,bands=path_bands))


if __name__=='__main__':
    main()
