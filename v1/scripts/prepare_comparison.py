"""Build both model inputs ONLY from the authorized original composition.

First verse = Intro+A+Pre+Chorus, beats [0,144), 54 seconds.
Consonants occupy at most 60ms / 30% of a note at its onset; remainder vowel.
No DB score, MIDI or LAB is read by this frontend.
"""
from diffart.paths import input_path, required_path
import json
from pathlib import Path
import numpy as np
from diffart.data import ROOT, SR, HOP, frame_labels, sha
from diffart.runtime import save_json

SOURCE=input_path('COMPARISON_SCORE','data/comparison.json')
UPSTREAM=input_path('COMPARISON_ROOT','external/comparison')

def main():
    score=json.loads(SOURCE.read_text());end_beat=144;seconds=end_beat*60/score['bpm']
    notes=[n for n in score['notes'] if 0<=n['start']<end_beat]
    assert all(n['start']+n['dur']<=end_beat for n in notes)
    out=ROOT/'artifacts/comparison';out.mkdir(exist_ok=True)
    save_json(out/'original_verse1.json',{**score,'notes':notes,'length_beats':end_beat,
        'sections':[s for s in score['sections'] if s['from_beat']<end_beat],
        'chords':[c for c in score['chords'] if c['beat']<end_beat]})
    mora={}
    for line in (UPSTREAM/'upstream/PrettyPitch/dict/ja.mora').read_text().splitlines():
        p=line.split('#')[0].split()
        if p:mora.setdefault(p[0],p[1:])
    rows=[];pitch=[];cursor=0.
    for note in notes:
        a=note['start']*60/score['bpm'];b=(note['start']+note['dur'])*60/score['bpm']
        if a>cursor:rows.append((cursor,a,'pau'));pitch.append(0.)
        assert a>=cursor-1e-6,(a,cursor)
        phones=mora[note['lyric']];cons=min(.06,(b-a)*.3) if len(phones)>1 else 0.
        for j,p in enumerate(phones):
            left=a+cons*j/max(1,len(phones)-1) if j<len(phones)-1 else a+cons
            right=a+cons*(j+1)/max(1,len(phones)-1) if j<len(phones)-1 else b
            # Voicing is the same for both models; unvoiced consonants have zero F0.
            voiced=p not in ['k','ky','s','sh','t','ch','ts','h','hy','f','p','cl']
            rows.append((left,right,p));pitch.append(440*2**((note['midi']-69)/12) if voiced else 0.)
        cursor=b
    if cursor<seconds:rows.append((cursor,seconds,'pau'));pitch.append(0.)
    vocab=json.loads((ROOT/'data/manifest.json').read_text())['vocab']
    assert all(p in vocab for _,_,p in rows)
    n=round(seconds*SR/HOP);context,pos=frame_labels(rows,n,vocab)
    ends=np.array([b for a,b,p in rows]);idx=np.searchsorted(ends,np.arange(n)*HOP/SR,side='right').clip(0,len(rows)-1)
    f0=np.array(pitch,np.float32)[idx]
    np.savez_compressed(out/'physics_front.npz',context=context,pos=pos,f0=f0)
    tab=[l.split()[0] for name in ['ja_en.phonemes','zh.phonemes'] for l in (UPSTREAM/'dict'/name).read_text().splitlines() if l.strip() and not l.startswith('#')]
    fps=44100/256;bounds=np.rint(np.array([0.]+[b for a,b,p in rows])*fps).astype(int)
    durations=np.diff(bounds);assert np.all(durations>0)
    ff=np.repeat(np.array(pitch,np.float32),durations)
    np.savez_compressed(out/'irodori_front.npz',offset=0.,start=np.array([0.]),
        n_ph=np.array([len(rows)]),ph=np.array([tab.index(p) for a,b,p in rows],np.int16),
        durs=durations,n_fr=np.array([len(ff)]),f0=ff,uv=(ff>0).astype(np.float32))
    save_json(out/'frontend_provenance.json',dict(source=str(SOURCE),source_sha256=sha(SOURCE),
        source_is_original_composition=True,db_score_used=False,beats=[0,end_beat],seconds=seconds,
        notes=len(notes),phoneme_intervals=rows,phoneme_f0=pitch,
        lyrics=''.join(n['lyric'] for n in notes),
        timing='note onset consonant <=60ms and <=30%; remaining note vowel; no learned alignment',
        physics_front_sha256=sha(out/'physics_front.npz'),irodori_front_sha256=sha(out/'irodori_front.npz')))
    print('Original verse prepared:',len(notes),'notes',seconds,'seconds')
if __name__=='__main__':main()
