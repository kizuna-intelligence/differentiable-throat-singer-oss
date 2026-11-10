"""User-supplied audio and timed singing phones or aligned speech, at 100 Hz.

Speech alignment follows the experiment's Narabas CTC monotonic Viterbi path.
Aligner/F0 weights and corpus material are obtained separately by the user.
"""
import json
import math
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from diffart.data import frame_labels
from .japanese_inventory import IPA

VOCAB={p:i for i,p in enumerate(sorted(IPA))}


def _validate_rows(rows, duration):
    result=[];previous=0.
    for start,end,phone in rows:
        start,end=float(start),float(end)
        if (not np.isfinite([start,end]).all() or start<previous-1e-7 or end<=start or
            end>duration+.011 or phone not in VOCAB):
            raise ValueError('Invalid ordered phone intervals in seconds')
        if start>previous+1e-7:
            result.append((previous,start,'pau'))
        result.append((start,end,phone));previous=end
    if not result:
        raise ValueError('Empty alignment')
    if previous<duration:
        result.append((previous,duration,'pau'))
    return result


def align_speech(audio, text, weights, symbols):
    """Return timed phones from text/reading, never an ASR transcript."""
    import pyopenjtalk
    import onnxruntime as ort
    import ast
    # Read the user-obtained model vocabulary as literal data; never execute it.
    tree=ast.parse(Path(symbols).read_text())
    entries=[ast.literal_eval(node.value) for node in tree.body
             if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='phoneme_symbols' for t in node.targets)]
    if len(entries)!=1 or not isinstance(entries[0],list) or not all(isinstance(p,str) for p in entries[0]):
        raise ValueError('Expected the Narabas phoneme_symbols literal list')
    phoneme_to_id={p:i for i,p in enumerate(entries[0])}
    from numba import njit
    normalize={'ny':['n','y'],'py':['p','y'],'kw':['k','w'],'gw':['g','w'],
               'v':['b'],'ty':['ch'],'dy':['j'],'sil':['pau'],'A':['a'],'E':['e']}
    phones=[q for p in pyopenjtalk.g2p(text).split() for q in normalize.get(p,[p])]
    if not phones or any(p not in VOCAB for p in phones):
        raise ValueError('Unsupported speech phones; provide a corrected reading')
    ids=np.array([1]+[phoneme_to_id[p.lower() if p in ('I','U','O') else p] for p in phones]+[2],np.int64)
    options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
    session=ort.InferenceSession(str(weights),sess_options=options,providers=['CPUExecutionProvider'])
    hop=int(session.get_modelmeta().custom_metadata_map['hop_length'])/16000
    signal=resample_poly(audio,2,3)
    logits=session.run(['output'],{'input':np.pad(signal,(4800,4800))[None].astype(np.float32)})[0][0]
    skip=round(.3/hop);logits=logits[skip:-skip].astype(np.float64)
    lp=logits-logits.max(1,keepdims=True);lp-=np.log(np.exp(lp).sum(1,keepdims=True))

    @njit(cache=False)
    def viterbi(lp, ids):
        nt,ns=len(lp),len(ids)
        sc=np.full((nt+1,ns+1),-1e30);sc[0,0]=0.
        moves=np.zeros((nt,ns+1),np.uint8)
        for t in range(nt):
            sc[t+1,0]=sc[t,0]+lp[t,0]
            for s in range(1,ns+1):
                stay=sc[t,s]+max(lp[t,0],lp[t,ids[s-1]])
                move=sc[t,s-1]+lp[t,ids[s-1]]
                moves[t,s]=move>=stay;sc[t+1,s]=max(stay,move)
        starts=np.zeros(ns,np.int64);s=ns
        for t in range(nt-1,-1,-1):
            if s>0 and moves[t,s]:
                starts[s-1]=t;s-=1
        if s!=0:
            raise ValueError('Alignment has no complete monotonic path')
        return starts

    starts=viterbi(lp,ids)*hop;duration=len(audio)/24000
    bounds=np.maximum.accumulate(np.clip(np.concatenate(([0.],starts[1:],[duration])),0,duration))
    rows=[(float(a),float(b),p) for a,b,p in zip(bounds[:-1],bounds[1:],['pau']+phones+['pau']) if b>a]
    return _validate_rows(rows,duration)


def prepare(manifest, out, aligner=None, symbols=None):
    manifest=Path(manifest).resolve();out=Path(out)
    rows=json.loads(manifest.read_text())
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Use an empty prepared directory')
    out.mkdir(parents=True,exist_ok=True);items=[];seen=set()
    def local(path):
        return (manifest.parent/path).resolve()
    for i,row in enumerate(rows):
        split=row['split']
        if split not in ('train','val','test') or row['kind'] not in ('singing','speech'):
            raise ValueError('Specify kind and train/val/test split')
        wav=local(row['audio'])
        if wav in seen:
            raise ValueError('An audio file must occur in only one split')
        seen.add(wav)
        audio,sr=sf.read(wav,dtype='float32',always_2d=True);audio=audio.mean(1)
        divisor=math.gcd(sr,24000)
        audio=resample_poly(audio,24000//divisor,sr//divisor).astype(np.float32)
        n=len(audio)//240;audio=audio[:n*240]
        if n<2 or not np.isfinite(audio).all():
            raise ValueError('Invalid or too short audio')
        if row.get('phones'):
            intervals=_validate_rows(row['phones'],len(audio)/24000)
        elif row.get('lab'):
            intervals=[]
            for line in local(row['lab']).read_text().splitlines():
                if line.strip():
                    a,b,p=line.split();intervals.append((float(a)*1e-7,float(b)*1e-7,p))
            intervals=_validate_rows(intervals,len(audio)/24000)
        elif row['kind']=='speech' and aligner is not None and symbols is not None:
            intervals=align_speech(audio,row.get('reading') or row['text'],aligner,symbols)
        else:
            raise ValueError('Singing needs timed phones; speech needs text and --aligner or reviewed timing')
        context,pos=frame_labels(intervals,n,VOCAB)
        if row.get('f0'):
            f0=np.load(local(row['f0']),allow_pickle=False).astype(np.float32)
            if f0.shape!=(n,):
                raise ValueError('F0 must contain one Hz value per 10 ms frame')
        else:
            import librosa
            values,voiced,_=librosa.pyin(audio,sr=24000,frame_length=2048,hop_length=240,fmin=32.,fmax=2000.)
            f0=np.where(voiced,values,0.)[:n].astype(np.float32)
        if f0.shape!=(n,) or not np.isfinite(f0).all() or np.any(f0<0):
            raise ValueError('F0 must be finite and nonnegative')
        name=f'item_{i:06d}.npz'
        np.savez_compressed(out/name,context=context,pos=pos,f0=f0,audio=audio)
        items.append(dict(file=name,kind=row['kind'],split=split))
    (out/'manifest.json').write_text(json.dumps(dict(vocab=VOCAB,sr=24000,hop=240,items=items),indent=2)+'\n')
