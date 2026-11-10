"""Training-phrase diagnostics only; does not access test audio."""
from diffart.paths import input_path, required_path
import sys,json
import numpy as np
import torch
from scipy.signal import resample_poly
from diffart.data import ROOT,SR,HOP
from diffart.dataset import Corpus
from diffart.metrics import SpeakerMetrics,EcapaMetrics,audio16,mel_cepstral_distortion
from diffart.runtime import save_json
import soundfile as sf

torch.set_num_threads(3);c=Corpus();p=json.load(open(ROOT/'artifacts/overfit/phrase.json'))
r=next(r for r in c.manifest['songs'] if r['id']==p['song']);d=c.load(r);a=p['start_frame'];b=a+p['frames']
ref,_=sf.read(ROOT/'artifacts/overfit/reference.wav',dtype='float32');y,_=sf.read(ROOT/'artifacts/overfit/best.wav',dtype='float32')
result={'mel_cepstral_distortion':mel_cepstral_distortion(ref,y),'scope':'Training phrase only; no generalization or extrapolation conclusion'}
for name,cls in [('wavlm',SpeakerMetrics),('ecapa',EcapaMetrics)]:
    metric=cls();real=metric.embedding(resample_poly(ref,2,3));pred=metric.embedding(resample_poly(y,2,3))
    scores=[];real_scores=[];anchors=[]
    for row in [r for r in c.manifest['songs'] if r['split']=='train' and r['id']!=p['song']]:
        phrase=c.phrase(row)
        if not phrase:continue
        data=c.load(row);start=phrase['start_frame']*HOP;end=start+phrase['frames']*HOP
        emb=metric.embedding(resample_poly(data['y'][start:end],2,3));anchors.append(phrase)
        scores.append(float(pred@emb));real_scores.append(float(real@emb))
        if len(scores)==3:break
    result[name]=dict(synth_same_excerpt_cosine=float(real@pred),synth_other_recordings_mean_cosine=float(np.mean(scores)),
        real_other_recordings_mean_cosine=float(np.mean(real_scores)),reference_phrases=anchors)
    save_json(ROOT/'artifacts/overfit/diagnostics.json',result);print(name,result[name],flush=True)
sys.path.insert(0,str(required_path('RMVPE_ROOT')))
from preprocess.algorithms.rmvpe import RMVPEPitchAlgorithm
rm=RMVPEPitchAlgorithm(model_path=str(required_path('RMVPE_WEIGHTS')),device='cpu',sample_rate=SR,hop_size=HOP,fmin=31.7,fmax=2005.5)
f0,confidence=rm.extract_continuous_periodicity(y);wanted=d['f0'][a:b];n=min(len(f0),len(wanted));given=wanted[:n]>0;both=given&(confidence[:n]>.03)&(f0[:n]>0)
result['pitch']=dict(median_abs_cents=float(np.median(np.abs(1200*np.log2(f0[:n][both]/wanted[:n][both])))),
    voiced_recall=float(both.sum()/max(1,given.sum())),within_50_cents=float(np.mean(np.abs(1200*np.log2(f0[:n][both]/wanted[:n][both]))<50)))
save_json(ROOT/'artifacts/overfit/diagnostics.json',result);print('pitch',result['pitch'],flush=True)
