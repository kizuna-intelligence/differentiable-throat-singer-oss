"""24kHz / 10ms frame grid; exact LAB timing; song-level, group-safe split."""
from diffart.paths import input_path, required_path
import hashlib
import json
import math
import random
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import mido
from .runtime import save_json

SR, HOP = 24000, 240
ROOT = Path(__file__).resolve().parents[1]
DB = input_path('RITSU_DB','data/ritsu/DATABASE')
VAL = input_path('RITSU_EXCLUSION_MANIFEST','data/exclusion.json')

def sha(path):
    h = hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def read_lab(path):
    rows=[]
    for line in Path(path).read_text().splitlines():
        a,b,p=line.split()
        # Dataset contract is 100ns integers, not guessed seconds.
        a,b=int(a)/1e7,int(b)/1e7
        if b<=a or (rows and a < rows[-1][1]-1e-7):
            raise ValueError(f'Nonmonotonic LAB: {path}: {line}')
        rows.append((a,b,p))
    return rows

def group(rows):
    phones=' '.join(p for _,_,p in rows if p not in ('sil','pau'))
    return 'ja_work_'+hashlib.sha256(phones.encode()).hexdigest()

def midi_notes(path):
    mid=mido.MidiFile(path)
    active={}; notes=[]; t=0.
    for msg in mid:  # mido merged iterator accounts for tempo changes, seconds
        t+=msg.time
        if msg.type=='note_on' and msg.velocity:
            active[(msg.channel,msg.note)]=t
        elif msg.type=='note_off' or (msg.type=='note_on' and not msg.velocity):
            key=(msg.channel,msg.note)
            if key in active:
                notes.append((active.pop(key),t,msg.note))
    return sorted(notes)

def score_f0(notes,times):
    out=np.zeros(len(times),np.float32)
    for a,b,p in notes: out[(times>=a)&(times<b)]=440*2**((p-69)/12)
    return out

def frame_labels(rows,n,vocab):
    times=np.arange(n)*HOP/SR
    end=np.array([b for a,b,p in rows]); idx=np.searchsorted(end,times,side='right')
    idx=np.clip(idx,0,len(rows)-1)
    phones=np.array([vocab[p] for a,b,p in rows])
    context=np.stack([phones[np.clip(idx+d,0,len(rows)-1)] for d in (-1,0,1)],-1)
    start=np.array([a for a,b,p in rows]); dur=end-start
    pos=np.clip((times-start[idx])/dur[idx],0,1)
    # Preserve start/end silence; reject interior gaps in audit.
    outside=(times<rows[0][0])|(times>=rows[-1][1])
    context[outside]=vocab['pau'];pos[outside]=0
    return context.astype(np.int64),pos.astype(np.float32)

def inventory():
    old=json.loads(VAL.read_text())['samples']
    excluded={r['group'] for r in old if '/ritsu/' in r.get('source_audio','')}
    songs=[]; vocab={'pau'}
    for wav in sorted(DB.glob('*/*.wav')):
        lab=wav.with_suffix('.lab'); mid=wav.with_suffix('.mid')
        rows=read_lab(lab); info=sf.info(wav); notes=midi_notes(mid)
        vocab.update(p for a,b,p in rows)
        pitches=np.array([p for a,b,p in notes])
        if not len(pitches): raise ValueError(f'Empty score {mid}')
        gaps=[(rows[i-1][1],a) for i,(a,b,p) in enumerate(rows) if i and a-rows[i-1][1]>1e-5]
        songs.append(dict(id=wav.stem,wav=str(wav),lab=str(lab),midi=str(mid),
            group=group(rows),duration=info.duration,sr=info.samplerate,channels=info.channels,
            score_p05=float(np.quantile(pitches,.05)),score_p95=float(np.quantile(pitches,.95)),
            wav_sha256=sha(wav),lab_sha256=sha(lab),midi_sha256=sha(mid),
            label_end=rows[-1][1],label_gaps=gaps,excluded=group(rows) in excluded))
    eligible=[r for r in songs if not r['excluded']]
    # Hold out two lowest-range and two highest-range songs, by score before fitting.
    lo=sorted(eligible,key=lambda r:(r['score_p05'],r['id']))[:2]
    hi=sorted([r for r in eligible if r not in lo],key=lambda r:(-r['score_p95'],r['id']))[:2]
    test={r['group'] for r in lo+hi}
    rest=sorted({r['group'] for r in eligible}-test)
    rng=random.Random(20261109);rng.shuffle(rest);val=set(rest[:4])
    for r in songs:r['split']='excluded' if r['excluded'] else 'test' if r['group'] in test else 'val' if r['group'] in val else 'train'
    assert not {r['group'] for r in eligible}&excluded
    for a,b in [('train','test'),('train','val'),('test','val')]:
        assert not {r['group'] for r in songs if r['split']==a}&{r['group'] for r in songs if r['split']==b}
    selected=[r for r in songs if r['split']=='train'];rng.shuffle(selected)
    remaining=3600.
    for r in selected:
        r['train_seconds']=min(r['duration'],remaining);remaining=max(0,remaining-r['duration'])
    manifest=dict(seed=20261109,sr=SR,hop=HOP,excluded_groups=sorted(excluded),
        exclusion_source=str(VAL),exclusion_sha256=sha(VAL),vocab={p:i for i,p in enumerate(sorted(vocab))},songs=songs)
    save_json(ROOT/'data/manifest.json',manifest)
    save_json(ROOT/'artifacts/data_audit.json',dict(songs=len(songs),duration_hours=sum(r['duration'] for r in songs)/3600,
       splits={s:[r['id'] for r in songs if r['split']==s] for s in ['train','val','test','excluded']},
       train_seconds=sum(r.get('train_seconds',0) for r in songs),label_gaps={r['id']:r['label_gaps'] for r in songs if r['label_gaps']}))
    return manifest

def prepare_audio(manifest):
    for r in manifest['songs']:
        if r['excluded']:continue
        dst=ROOT/'data/audio24'/f"{r['id']}.wav"
        if not dst.exists():
            y,sr=sf.read(r['wav'],dtype='float32',always_2d=True);y=y.mean(1)
            g=math.gcd(sr,SR);y=resample_poly(y,SR//g,sr//g).astype(np.float32)
            dst.parent.mkdir(parents=True,exist_ok=True);sf.write(dst,y,SR,subtype='FLOAT')
        n=sf.info(dst).frames//HOP
        context,pos=frame_labels(read_lab(r['lab']),n,manifest['vocab'])
        score=score_f0(midi_notes(r['midi']),np.arange(n)*HOP/SR)
        p=ROOT/'data/features'/f"{r['id']}.npz";p.parent.mkdir(parents=True,exist_ok=True)
        if not p.exists():np.savez_compressed(p,context=context,pos=pos,score_f0=score)
        print('audio24',r['id'],flush=True)
