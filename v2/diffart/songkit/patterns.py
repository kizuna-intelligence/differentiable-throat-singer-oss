"""Accompaniment generators. Register a function with @pattern('name') to extend."""
from __future__ import annotations
import math

PATTERNS = {}
def pattern(name):
    def register(fn):
        PATTERNS[name] = fn
        return fn
    return register

ROOTS = {'C':0,'C#':1,'Db':1,'D':2,'D#':3,'Eb':3,'E':4,'F':5,'F#':6,'Gb':6,'G':7,'G#':8,'Ab':8,'A':9,'A#':10,'Bb':10,'B':11}
def tones(chord):
    root = chord[:2] if len(chord)>1 and chord[1] in '#b' else chord[0]
    suffix = chord[len(root):]
    return ROOTS[root], ([0,3,7] if suffix.startswith('m') and not suffix.startswith('maj') else [0,4,7]) + ([10] if '7' in suffix else [])
def chord_at(spec, beat):
    out=spec['chords'][0]['chord']
    for x in spec['chords']:
        if x['beat'] > beat + 1e-9: break
        out=x['chord']
    return out

def note(beat, length, midi, lyric, **kw):
    return dict(start=float(beat), dur=float(length), midi=int(midi), lyric=lyric, pitched=True, section='song', **kw)
def fit_pitch(pc, low, high):
    choices=[m for m in range(low-12,high+13) if m%12==pc%12]
    return min(choices,key=lambda m:(max(low-m,m-high,0),abs(m-(low+high)/2)))

@pattern('bass_root_fifth')
def bass(spec,p):
    end=p.get('end',spec['length_beats']); start=p.get('start',0)
    for b in range(math.ceil(start),math.ceil(end)):
        root,_=tones(chord_at(spec,b)); midi=fit_pitch(root+(7 if (b-int(start))%spec['beats_per_bar']%2 else 0),*p.get('range',[45,52]))
        syll=p.get('syllable','どぅん')
        if syll=='どぅん':
            for off,dur,lyr in [(0,.15,'ど'),(.15,.12,'う'),(.27,.08,'ん')]: yield note(b+off,dur,midi,lyr)
        else: yield note(b,p.get('duration',.35),midi,syll)

@pattern('offbeat_chord')
def offbeat(spec,p):
    n=p.get('voices',3); idx=p.get('voice',0); low,high=p.get('range',[60,69])
    for b in range(math.ceil(p.get('start',0)),math.ceil(p.get('end',spec['length_beats']))):
        root,steps=tones(chord_at(spec,b+.5)); chord=chord_at(spec,b+.5)
        voicing=p.get('voicings',{}).get(chord)
        if voicing is None:
            candidates=sorted({fit_pitch(root+s,low,high) for s in steps})
            voicing=[candidates[i%len(candidates)] for i in range(n)]
        yield note(b+.5,p.get('duration',.25),voicing[idx],p.get('syllable','ぱ'))

@pattern('sustain_chord')
def sustain(spec,p):
    b=p.get('start',0);end=p.get('end',spec['length_beats'])
    changes=[b]+[c['beat'] for c in spec['chords'] if b<c['beat']<end]+[end]
    for a,z in zip(changes,changes[1:]):
        root,steps=tones(chord_at(spec,a)); idx=p.get('voice',0)
        midi=p.get('midi',fit_pitch(root+steps[idx%len(steps)],*p.get('range',[60,69])))
        yield note(a,z-a,midi,p.get('syllable','あ'))

@pattern('mouth_perc')
def percussion(spec,p):
    a=p.get('start',0);end=p.get('end',spec['length_beats']);step=p.get('step',.5)
    t=a+p.get('offset',.5)
    while t<end-1e-8:
        yield dict(note(t,p.get('duration',.15),60,p.get('syllable','つ')),pitched=False)
        t+=step

@pattern('harmony_third_below')
def harmony(spec,p):
    key=spec['key'].split();root=ROOTS[key[0]];minor='minor' in spec['key'].lower()
    scale={(root+x)%12 for x in ([0,2,3,5,7,8,10] if minor else [0,2,4,5,7,9,11])}
    lead=next(x for x in spec['parts'] if x['id']==p.get('source','lead'))
    for n in lead['notes']:
        if p.get('start',0)<=n['start']<p.get('end',spec['length_beats']):
            lower=[m for m in range(n['midi']-12,n['midi']) if m%12 in scale]
            yield dict(n,midi=lower[-2])

def expand(spec):
    for part in spec['parts']:
        events=[dict(n) for n in part.get('notes',[])]
        for conf in part.get('patterns',[]):
            for n in PATTERNS[conf['type']](spec,conf):
                n['section']=conf.get('section','song');events.append(n)
        for n in events:
            n.setdefault('pitched',True);n.setdefault('section','song')
            if n['start']<0 or n['dur']<=0 or n['start']+n['dur']>spec['length_beats']+1e-7: raise ValueError(n)
        part['notes']=sorted(events,key=lambda n:(n['start'],n['dur']))
        if not part['notes']: raise ValueError(f"No events: {part['id']}")
    return spec
