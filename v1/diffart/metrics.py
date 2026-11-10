"""Metrics with explicit reference selection; no claimed identity thresholds."""
from diffart.paths import input_path, required_path
import math
import re
import sys
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from scipy.fft import dct
import torch
import librosa

WAVLM=input_path('WAVLM_DIR','external/wavlm-base-plus-sv')
HT=input_path('HEART_TRANSCRIPTOR_DIR','external/HeartTranscriptor-oss')

def audio16(path):
    y,sr=sf.read(path,dtype='float32',always_2d=True);y=y.mean(1)
    g=math.gcd(sr,16000)
    return resample_poly(y,16000//g,sr//g).astype(np.float32)

def cer(ref,hyp):
    prev=list(range(len(hyp)+1))
    for i,a in enumerate(ref,1):
        cur=[i]
        for j,b in enumerate(hyp,1):cur.append(min(prev[j]+1,cur[-1]+1,prev[j-1]+(a!=b)))
        prev=cur
    return prev[-1]/max(1,len(ref))

def kana(text):
    import pyopenjtalk
    text=pyopenjtalk.g2p(text,kana=True)
    return ''.join(chr(ord(c)-96) if 'ァ'<=c<='ヶ' else c for c in text if 'ァ'<=c<='ヶ' or 'ぁ'<=c<='ゖ' or c=='ー')

def label_kana(phones):
    # Reuse the established Sinsy mora table, but import no upstream trainer.
    rev={}
    table=input_path('MORA_TABLE','external/ja.mora')
    for line in table.read_text().splitlines():
        p=line.split('#')[0].split()
        if p and re.fullmatch('[ぁ-ゟ]+',p[0]):rev.setdefault(tuple(p[1:]),p[0])
    out=[];i=0
    while i<len(phones):
        if phones[i] in ['pau','sil','br','Edge','GlottalStop']:
            i+=1;continue
        for n in range(min(4,len(phones)-i),0,-1):
            key=tuple(phones[i:i+n])
            if key in rev:out.append(rev[key]);i+=n;break
        else:i+=1
    return ''.join(out)

class SpeakerMetrics:
    def __init__(self,device='cpu'):
        from transformers import AutoFeatureExtractor,WavLMForXVector
        self.device=device
        self.fe=AutoFeatureExtractor.from_pretrained(WAVLM,local_files_only=True)
        self.model=WavLMForXVector.from_pretrained(WAVLM,local_files_only=True).eval().to(device)
    @torch.inference_mode()
    def embedding(self,y):
        x=self.fe(y,sampling_rate=16000,return_tensors='pt',return_attention_mask=True).to(self.device)
        return torch.nn.functional.normalize(self.model(**x).embeddings[0],dim=-1).cpu().numpy()

class Transcriptor:
    def __init__(self,device='cpu'):
        from transformers import WhisperProcessor,WhisperForConditionalGeneration
        self.device=device
        self.proc=WhisperProcessor.from_pretrained(HT,local_files_only=True)
        self.model=WhisperForConditionalGeneration.from_pretrained(HT,local_files_only=True).eval().to(device)
    @torch.inference_mode()
    def transcript(self,y):
        x=self.proc(y,sampling_rate=16000,return_tensors='pt',return_attention_mask=True).to(self.device)
        ids=self.model.generate(**x,language='ja',task='transcribe',do_sample=False,num_beams=1,max_new_tokens=128)
        return self.proc.batch_decode(ids,skip_special_tokens=True)[0]

def mel_cepstral_distortion(a,b,sr=24000):
    """Aligned mel cepstra (24 coefficients, c0 excluded), NOT WORLD/SPTK MCD.

    Formula: 10/ln(10)*sqrt(2*sum((c_ref-c_pred)^2)). Fixed same-time grid.
    Excludes silent target frames. This implementation reports its exact definition.
    """
    def cep(y):
        spec=librosa.feature.melspectrogram(y=y,sr=sr,n_fft=1024,hop_length=240,n_mels=80,power=1.)
        return dct(np.log(np.maximum(spec,1e-5)),type=2,axis=0,norm='ortho')[1:25]
    ca,cb=cep(a),cep(b);n=min(ca.shape[1],cb.shape[1]);ca,cb=ca[:,:n],cb[:,:n]
    rms=librosa.feature.rms(y=a,frame_length=1024,hop_length=240)[0][:n]
    active=rms>max(1e-4,rms.max()*.03)
    val=10/np.log(10)*np.sqrt(2*((ca-cb)**2).sum(0))
    return float(val[active].mean()) if active.any() else None

class EcapaMetrics:
    def __init__(self,device='cpu'):
        import torchaudio
        # SpeechBrain still checks this API on torchaudio 2.11; all actual I/O
        # here uses SoundFile, no monkey-patched decoder is invoked.
        if not hasattr(torchaudio,'list_audio_backends'):torchaudio.list_audio_backends=lambda:['soundfile']
        from speechbrain.inference.speaker import EncoderClassifier
        p=Path(__file__).resolve().parents[1]/'artifacts/models/ecapa'
        self.model=EncoderClassifier.from_hparams(source=str(p),savedir=str(p/'loaded'),
            overrides={'pretrained_path':str(p)},run_opts={'device':device})
        self.device=device
    @torch.inference_mode()
    def embedding(self,y):
        x=self.model.encode_batch(torch.tensor(y.copy(),device=self.device)[None]).flatten()
        return torch.nn.functional.normalize(x,dim=0).cpu().numpy()
