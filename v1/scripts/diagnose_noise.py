"""Separately ablate the trained glottal-breath and constriction-noise paths."""
import argparse,json
import numpy as np
import soundfile as sf
from diffart.data import ROOT,SR,HOP,read_lab,sha
from diffart.runtime import device_lease,save_json

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--device',default='cpu');args=ap.parse_args()
    with device_lease(args.device) as device:
        import torch
        from diffart.dataset import Corpus
        from diffart.physics import Synthesizer
        torch.set_num_threads(3);corpus=Corpus();ck=ROOT/'artifacts/main/best.pt'
        model=Synthesizer(len(corpus.manifest['vocab'])).to(device)
        model.load_state_dict(torch.load(ck,map_location=device,weights_only=True)['model']);model.eval()
        rows=[];dest=ROOT/'listen/noise_diagnostic';dest.mkdir(exist_ok=True)
        with torch.inference_mode():
            for song in [r for r in corpus.manifest['songs'] if r['split']=='test']:
                for phone in ['h','s','a','pau']:
                    candidates=[(a,b) for a,b,p in read_lab(song['lab']) if p==phone and b-a>=.04 and a>=.5]
                    if not candidates:continue
                    a,b=candidates[0];start=max(0,int((a-.3)*100));data=corpus.excerpt(song,start,100,device)
                    full,p=model(*data[:3],seed=1234,return_controls=True)
                    components={}
                    for control in ['breath','turbulence']:
                        def mute(module,inputs,output):
                            return {**output,control:torch.zeros_like(output[control])}
                        hook=model.controls.register_forward_hook(mute)
                        try:without=model(*data[:3],seed=1234)
                        finally:hook.remove()
                        delta=(full-without).cpu().numpy();components[control]=float(np.sqrt(np.mean(delta**2)))
                        sf.write(dest/f"{song['id']}__{phone}__{control}_only.wav",delta,SR,subtype='FLOAT')
                    sf.write(dest/f"{song['id']}__{phone}__full.wav",full.cpu().numpy(),SR,subtype='FLOAT')
                    sf.write(dest/f"{song['id']}__{phone}__recording.wav",data[-1].cpu().numpy(),SR,subtype='FLOAT')
                    mask=data[0][:,1]==corpus.manifest['vocab'][phone]
                    rows.append(dict(song=song['id'],phone=phone,start_seconds=start*.01,
                        glottal_breath_output_rms=components['breath'],constriction_turbulence_output_rms=components['turbulence'],
                        full_output_rms=float(full.square().mean().sqrt()),
                        phone_mean_breath=float(p['breath'][mask].mean()),phone_mean_turbulence=float(p['turbulence'][mask].mean()),
                        phone_mean_amplitude=float(p['amplitude'][mask].mean()),
                        mean_glottal_noise_gain=float((.03*p['amplitude']*p['breath']).mean()),
                        mean_constriction_noise_gain_before_area_and_filter=float((.03*p['amplitude']*p['turbulence']).mean())))
        save_json(ROOT/'artifacts/evaluation/noise_paths.json',dict(checkpoint_sha256=sha(ck),rows=rows,
            method='same F0, phase, seed and trained model; full minus output with each control zeroed separately',
            all_paths_nonzero=all(r['glottal_breath_output_rms']>0 and r['constriction_turbulence_output_rms']>0 for r in rows),
            interpretation='nonzero routes show operation, not intelligibility or naturalness'))
        print('noise diagnostics',len(rows),'done',flush=True)
if __name__=='__main__':main()
