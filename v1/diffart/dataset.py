import json
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from .data import ROOT,SR,HOP,read_lab,sha
from .runtime import save_json

class Corpus:
    def __init__(self):
        self.manifest=json.loads((ROOT/'data/manifest.json').read_text())
        self.cache={}
    def load(self,song):
        sid=song['id']
        if sid not in self.cache:
            feat=dict(np.load(ROOT/'data/features'/f'{sid}.npz'))
            pitch=dict(np.load(ROOT/'data/f0'/f'{sid}.npz'))
            y,_=sf.read(ROOT/'data/audio24'/f'{sid}.wav',dtype='float32')
            feat.update(f0=pitch['f0'],y=y)
            assert len(feat['context'])==len(feat['f0'])
            self.cache[sid]=feat
        return self.cache[sid]
    def excerpt(self,song,start,frames,device='cpu'):
        d=self.load(song);a=int(start);b=a+int(frames)
        if a<0 or b>len(d['f0']):raise ValueError((a,b,len(d['f0'])))
        return tuple(torch.as_tensor(np.array(d[k][a:b]),device=device) for k in ['context','pos','f0'])+(torch.tensor(d['y'][a*HOP:b*HOP],device=device),)
    def phrase(self,song,min_seconds=2.5,max_seconds=5.):
        rows=read_lab(song['lab']);candidates=[];cur=[]
        for a,b,p in rows+[(rows[-1][1],rows[-1][1]+1,'pau')]:
            if p=='pau' and b-a>=.2:
                if cur:
                    start=max(0,cur[0][0]-.08);end=cur[-1][1]+.08
                    if min_seconds<=end-start<=max_seconds:
                        candidates.append((start,end,cur))
                cur=[]
            else:cur.append((a,b,p))
        if not candidates:return None
        # Choose first complete phrase, no evaluation result influences selection.
        a,b,phones=candidates[0]
        return dict(song=song['id'],start_frame=int(a*SR/HOP),frames=int((b-a)*SR/HOP),
             phones=[p for _,_,p in phones],start_seconds=int(a*SR/HOP)*HOP/SR)

def stage1_audit():
    c=Corpus();m=c.manifest;checks=[]
    for r in m['songs']:
        if r['excluded']:continue
        d=c.load(r);n=len(d['f0'])
        assert len(d['y'])//HOP==n
        assert np.isfinite(d['y']).all() and np.isfinite(d['f0']).all()
        assert np.all((d['f0']==0)|((d['f0']>=31)&(d['f0']<2010)))
        checks.append(dict(song=r['id'],frames=n,voiced_frames=int((d['f0']>0).sum())))
    assert (ROOT/'artifacts/歌声データベース利用規約.txt').exists()
    assert sum(r.get('train_seconds',0) for r in m['songs'])==3600
    save_json(ROOT/'artifacts/STAGE1_PASS.json',dict(songs=checks,manifest_sha256=sha(ROOT/'data/manifest.json'),
         license_notes_sha256=sha(ROOT/'LICENSE_NOTES.md'),external_comparison='Irodori blocked by I-003'))
    return c
