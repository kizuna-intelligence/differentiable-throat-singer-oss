"""Shared evaluation: fixed independent TRAIN anchors and same-time segments."""
from diffart.paths import input_path, required_path
import sys
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from diffart.data import ROOT,SR,HOP
from diffart.runtime import save_json
from diffart.metrics import SpeakerMetrics,EcapaMetrics,Transcriptor,kana,cer

class Assessment:
    def __init__(self,corpus,device):
        self.sv=SpeakerMetrics(device);self.ecapa=EcapaMetrics(device)
        self.ht=None;self.device=device
        sys.path.insert(0,str(required_path('RMVPE_ROOT')))
        from preprocess.algorithms.rmvpe import RMVPEPitchAlgorithm
        self.pitch=RMVPEPitchAlgorithm(model_path=str(required_path('RMVPE_WEIGHTS')),
            device='cuda' if device.startswith('cuda') else 'cpu',sample_rate=SR,hop_size=HOP,fmin=31.7,fmax=2005.5)
        refs=[];meta=[]
        for row in [r for r in corpus.manifest['songs'] if r['split']=='train']:
            phrase=corpus.phrase(row)
            if phrase:
                d=corpus.load(row);a=phrase['start_frame']*HOP;b=a+phrase['frames']*HOP
                path=ROOT/'artifacts/evaluation'/f'reference_{len(refs)+1}.wav'
                path.parent.mkdir(parents=True,exist_ok=True)
                if not path.exists():sf.write(path,d['y'][a:b],SR,subtype='FLOAT')
                else:
                    existing,ref_sr=sf.read(path,dtype='float32');assert ref_sr==SR and np.array_equal(existing,d['y'][a:b])
                refs.append(resample_poly(d['y'][a:b],2,3));meta.append({**phrase,'path':str(path),'split':'train'})
            if len(refs)==3:break
        self.anchors={name:np.stack([metric.embedding(y) for y in refs]) for name,metric in [('wavlm',self.sv),('ecapa',self.ecapa)]}
        save_json(ROOT/'artifacts/evaluation/references.json',meta)
    def assess(self,y,f0,active_wave=None,expected_text=None):
        n=len(f0);y=np.pad(y[:n*HOP],(0,max(0,n*HOP-len(y))))
        data={'wavlm':[],'ecapa':[],'transcripts':[]};cents=[];given=representable=both_count=within50=0;segments=[]
        for a in range(0,n,2000):
            b=min(a+2000,n)
            if b-a<100:continue
            if active_wave is not None:
                if np.mean(active_wave[a*HOP:b*HOP]**2)<1e-7:continue
            elif not (f0[a:b]>0).any():continue
            part=y[a*HOP:b*HOP];y16=resample_poly(part,2,3)
            for name,metric in [('wavlm',self.sv),('ecapa',self.ecapa)]:
                emb=metric.embedding(y16);data[name].append(float((self.anchors[name]@emb).mean()))
            if expected_text is not None:
                if self.ht is None:self.ht=Transcriptor(self.device)
                data['transcripts'].append(self.ht.transcript(y16))
            # RMVPE wrapper requires [-1, 1]; some renders (e.g. Irodori) peak above 1. Scale only for pitch analysis.
            peak=float(abs(part).max()) if len(part) else 0.
            estimated,confidence=self.pitch.extract_continuous_periodicity(part/peak if peak>1 else part)
            wanted=f0[a:b][:len(estimated)];estimated=estimated[:len(wanted)];confidence=confidence[:len(wanted)]
            valid=(wanted>=31.7)&(wanted<=2005.5);detected=valid&(confidence>=.03)&(estimated>0)
            given+=int((wanted>0).sum());representable+=int(valid.sum());both_count+=int(detected.sum())
            e=np.abs(1200*np.log2(estimated[detected]/wanted[detected]));cents.extend(e.tolist());within50+=int((e<=50).sum())
            segments.append(dict(start_seconds=a*.01,end_seconds=b*.01,wavlm_cosine=data['wavlm'][-1],ecapa_cosine=data['ecapa'][-1],wanted_voiced=int((wanted>0).sum()),detected_voiced=int(detected.sum())))
        return dict(wavlm_cosine=float(np.mean(data['wavlm'])),ecapa_cosine=float(np.mean(data['ecapa'])),
            kana_cer=cer(expected_text,kana(''.join(data['transcripts']))) if expected_text is not None else None,
            expected_kana=expected_text,transcripts=data['transcripts'],segments=segments,
            pitch_median_abs_cents=float(np.median(cents)) if cents else None,
            pitch_voiced_recall=both_count/representable if representable else None,
            pitch_within50_given_voiced=within50/representable if representable else None,
            rmvpe_range_coverage=representable/given if given else None,
            voiced_frames=given,representable_voiced_frames=representable,detected_voiced_frames=both_count)

def slope_rows(results):
    out=[]
    for model in sorted({r['model'] for r in results if r['condition']=='score'}):
        for metric in ['wavlm_cosine','ecapa_cosine']:
            for sign in [-1,1]:
                rows=[r for r in results if r['model']==model and r['condition']=='score' and r['semitones']*sign>=0]
                baseline={r['song']:r[metric] for r in rows if r['semitones']==0}
                x=np.array([abs(r['semitones']) for r in rows]);y=np.array([r[metric]-baseline[r['song']] for r in rows])
                out.append(dict(model=model,metric=metric,direction=sign,
                    cosine_change_per_abs_semitone=float(np.polyfit(x,y,1)[0]),
                    definition='OLS of similarity minus same song zero-shift similarity vs absolute semitones',n=len(rows)))
    return out
