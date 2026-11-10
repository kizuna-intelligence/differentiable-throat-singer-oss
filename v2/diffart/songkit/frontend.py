from pathlib import Path
import numpy as np
from diffart.data import SR,HOP,frame_labels
CONS_DUR = {'k': .092, 's': .106, 'sh': .120, 't': .084, 'ch': .123, 'ts': .127, 'h': .062,
            'r': .036, 'm': .052, 'n': .050, 'w': .058, 'y': .081, 'g': .055, 'd': .054,
            'b': .048, 'p': .050, 'z': .090, 'j': .085, 'f': .046, 'ky': .09, 'gy': .055,
            'hy': .062, 'my': .052, 'ry': .036, 'by': .048}
CL_DUR = .12
UNVOICED = {'k', 'ky', 's', 'sh', 'ch', 't', 'ts', 'p', 'cl', 'h', 'hy', 'f'}
VOWELS = {'a', 'i', 'u', 'e', 'o', 'N'}


def mora_table():
    """Minimal hiragana spelling rules authored for this project (not a copied dictionary)."""
    table={v:[p] for v,p in zip('あいうえお','aiueo')}
    for chars,cons in [('かきくけこ','k'),('がぎぐげご','g'),('さしすせそ','s'),('ざじずぜぞ','z'),('たちつてと','t'),('だぢづでど','d'),('なにぬねの','n'),('はひふへほ','h'),('ばびぶべぼ','b'),('ぱぴぷぺぽ','p'),('まみむめも','m'),('らりるれろ','r')]:
        for kana,v in zip(chars,'aiueo'):table[kana]=[cons,v]
    table.update({'し':['sh','i'],'じ':['j','i'],'ち':['ch','i'],'ぢ':['j','i'],'つ':['ts','u'],'づ':['z','u'],'ふ':['f','u'],'や':['y','a'],'ゆ':['y','u'],'よ':['y','o'],'わ':['w','a'],'を':['o'],'ん':['N']})
    for base,cons in [('き','ky'),('ぎ','gy'),('し','sh'),('じ','j'),('ち','ch'),('に','n'),('ひ','hy'),('び','by'),('ぴ','p'),('み','my'),('り','ry')]:
        for small,v in zip('ゃゅょ','auo'):
            table[base+small]=([cons,'y',v] if base in ('に','ぴ') else [cons,v])
    return table

def build_rows(score, table):
    spb = 60 / score['bpm']
    rows = []  # (start, end, phone)
    pitch_notes = []  # (start, end, midi)
    cursor = 0.0
    last_vowel = None
    for n in score['notes']:
        a = n['start'] * spb
        b = (n['start'] + n['dur']) * spb
        if a > cursor + 1e-9:
            rows.append([cursor, a, 'pau'])
            last_vowel = None
        lyric = n['lyric']
        if lyric == 'ー':
            assert last_vowel in 'aiueo', n
            rows.append([a, b, last_vowel])
        elif lyric == 'っ':
            assert last_vowel in 'aiueo', n
            cl = min(CL_DUR, b - a)
            if b - a - cl > 1e-6:
                rows[-1][1] = b - cl  # previous vowel is held into the first part of the note
            rows.append([b - cl, b, 'cl'])
        else:
            phones = table[lyric]
            cons, vowel = phones[:-1], phones[-1]
            t = a
            budget = .4 * (b - a)
            total = sum(CONS_DUR[c] for c in cons)
            scale = min(1.0, budget / total) if total else 1.0
            for c in cons:
                d = max(.030, CONS_DUR[c] * scale)
                if t + d > b:
                    t = b - d
                if rows and rows[-1][1] > t: rows[-1][1] = t
                rows.append([t, t + d, c])
                t += d
            rows.append([t, b, vowel])
            last_vowel = vowel if vowel in 'aiueo' else None
        pitch_notes.append((a, b, n['midi']))
        cursor = b
    total = score['seconds']
    if cursor < total:
        rows.append([cursor, total, 'pau'])
    if rows[0][2] != 'pau' and rows[0][0] > 0:
        rows.insert(0, [0.0, rows[0][0], 'pau'])
    return [tuple(r) for r in rows], pitch_notes


def build_straight_f0(pitch_notes, rows, n, portamento_ms=20):
    """Score F0 with 20 ms boundary smoothing and no vibrato."""
    times = np.arange(n) * HOP / SR
    semitones = np.full(n, np.nan, dtype=np.float32)
    for a, b, midi in pitch_notes:
        semitones[(times >= a) & (times < b)] = midi
    valid = np.flatnonzero(np.isfinite(semitones))
    if not len(valid):
        raise ValueError('score has no vocal notes')
    first, last = valid[0], valid[-1]
    semitones[:first] = semitones[first]
    for i in range(first + 1, last + 1):
        if not np.isfinite(semitones[i]):
            semitones[i] = semitones[i - 1]
    semitones[last + 1:] = semitones[last]

    # A symmetric Hann smoother gives a 20 ms half-width at this 10 ms frame rate.
    half_frames = max(1, round(portamento_ms / (1000 * HOP / SR)))
    kernel = np.hanning(2 * half_frames + 3)[1:-1]
    kernel /= kernel.sum()
    pad = len(kernel) // 2
    smoothed = np.convolve(np.pad(semitones, pad, mode='edge'), kernel, mode='valid')
    f0 = 440.0 * np.exp2((smoothed - 69.0) / 12.0)

    ends = np.asarray([r[1] for r in rows])
    row_ids = np.searchsorted(ends, times, side='right').clip(0, len(rows) - 1)
    phones = np.asarray([r[2] for r in rows])[row_ids]
    unvoiced = {'k', 'ky', 's', 'sh', 'ch', 't', 'ts', 'p', 'cl', 'h', 'hy', 'f', 'pau'}
    f0[np.isin(phones, list(unvoiced))] = 0.0
    return f0.astype(np.float32)



def inputs(spec,part,vocab):
    score=dict(spec,notes=part['notes'])
    rows,pitches=build_rows(score,mora_table())
    if part.get('role')=='percussion':
        rows=[(a,b,{'u':'ts','i':'ch'}.get(phone,phone)) for a,b,phone in rows]
    rows=[r for r in rows if r[1]>r[0]]
    missing={p for _,_,p in rows}-set(vocab)
    if missing: raise ValueError(missing)
    n=round(spec['seconds']*100)
    ctx,pos=frame_labels(rows,n,vocab)
    f0=build_straight_f0(pitches,rows,n)
    times=np.arange(n)/100
    for note in part['notes']:
        if not note['pitched']:
            f0[(times>=note['start']*60/spec['bpm'])&(times<(note['start']+note['dur'])*60/spec['bpm'])]=0
    return ctx,pos,f0,rows
