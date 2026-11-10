"""Vocal-only mix, specified panning, short reverb and measured loudness."""
from pathlib import Path
import json, math, re, subprocess
import numpy as np
import soundfile as sf
from scipy.signal import butter, fftconvolve, resample_poly, sosfilt, sosfiltfilt
from .pipeline import write,sha,fingerprint
SR=44100
def db(value: float) -> float:
    return 20 * math.log10(max(value, 1e-12))


def band_rms(signal: np.ndarray, lo: float, hi: float) -> float:
    sos = butter(4, [lo, hi], btype='bandpass', fs=SR, output='sos')
    filtered = sosfiltfilt(sos, signal.astype(np.float64))
    return float(np.sqrt(np.mean(filtered * filtered)))


def rms_frames(signal: np.ndarray, frame: int) -> np.ndarray:
    samples = signal[:, None] if signal.ndim == 1 else signal
    count = math.ceil(len(samples) / frame)
    padded = np.pad(samples, ((0, count * frame - len(samples)), (0, 0)))
    return np.sqrt(np.mean(padded.reshape(count, frame, samples.shape[1]) ** 2, axis=(1, 2)))


def pan_stereo(mono: np.ndarray, pan: float) -> np.ndarray:
    angle = (float(np.clip(pan, -1.0, 1.0)) + 1.0) * math.pi / 4.0
    return np.stack((mono * math.cos(angle), mono * math.sin(angle)), axis=-1)


def diffuse_reverb(mono: np.ndarray, wet_db: float, seed: int, seconds: float = .8) -> np.ndarray:
    tail = int(round(seconds * SR))
    total = len(mono) + tail
    dry = np.pad(mono.astype(np.float64), (0, tail))
    t = np.arange(tail, dtype=np.float64) / SR
    rng = np.random.default_rng(seed)
    impulse = np.zeros(tail, dtype=np.float64)
    impulse[0] = 0.32
    impulse[1:] = rng.standard_normal(tail - 1) * np.exp(-t[1:] / 0.17)
    impulse = sosfilt(butter(1, 6200, btype='lowpass', fs=SR, output='sos'), impulse)
    norm = np.sqrt(np.sum(impulse * impulse))
    if norm:
        impulse /= norm
    wet = fftconvolve(dry, impulse, mode='full')[:total]
    wet_rms = float(np.sqrt(np.mean(wet * wet)))
    dry_rms = float(np.sqrt(np.mean(dry * dry)))
    if wet_rms:
        wet *= dry_rms / wet_rms
    return (dry + (10 ** (wet_db / 20)) * wet).astype(np.float32)


def ebur128(path: Path) -> dict:
    cmd = ['ffmpeg', '-hide_banner', '-nostats', '-i', str(path), '-filter_complex',
           'ebur128=peak=true', '-f', 'null', '-']
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    text = result.stderr
    integrated = re.findall(r'Integrated loudness:\s*I:\s*(-?[0-9.]+) LUFS', text)
    peaks = re.findall(r'True peak:\s*Peak:\s*(-?[0-9.]+) dBFS', text)
    return dict(integrated_lufs=float(integrated[-1]) if integrated else None,
                true_peak_dbtp=float(peaks[-1]) if peaks else None)


def encode_mp3(source: Path, output: Path, title: str, target_lufs=-14, target_tp=-1) -> dict:
    first = subprocess.run([
        'ffmpeg', '-hide_banner', '-nostats', '-i', str(source), '-af',
        f'loudnorm=I={target_lufs}:TP={target_tp-.6}:LRA=7:print_format=json', '-f', 'null', '-'],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    blocks = re.findall(r'\{\s*"input_i".*?\}', first.stderr, re.S)
    if not blocks:
        raise RuntimeError('FFmpeg loudnorm analysis did not return measurements')
    measured = json.loads(blocks[-1])
    filt = (f'loudnorm=I={target_lufs}:TP={target_tp-.6}:LRA=7:'
            f'measured_I={measured["input_i"]}:measured_TP={measured["input_tp"]}:'
            f'measured_LRA={measured["input_lra"]}:measured_thresh={measured["input_thresh"]}:'
            f'offset={measured["target_offset"]}:linear=true:print_format=summary')
    subprocess.run([
        'ffmpeg', '-hide_banner', '-nostats', '-y', '-i', str(source), '-af', filt,
        '-c:a', 'libmp3lame', '-b:a', '192k', '-id3v2_version', '3',
        '-metadata', f'title={title}', '-metadata', 'comment=精度目標未達・比較試聴用',
        str(output)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    encoded=ebur128(output);correction=0.;iterations=[]
    for _ in range(3):
        difference=target_lufs-encoded['integrated_lufs']
        headroom=target_tp-encoded['true_peak_dbtp']-.1
        if abs(difference)<=.5 and headroom>=0:break
        adjustment=min(difference,headroom) if difference>0 else difference
        if abs(adjustment)<.05:break
        correction+=adjustment
        subprocess.run(['ffmpeg','-hide_banner','-nostats','-y','-i',str(source),'-af',filt+f',volume={correction}dB','-c:a','libmp3lame','-b:a','192k','-id3v2_version','3','-metadata',f'title={title}','-metadata','comment=精度目標未達・比較試聴用',str(output)],capture_output=True,text=True,check=True)
        encoded=ebur128(output);iterations.append(dict(gain_correction_db=correction,encoded=encoded))
    return dict(analysis=measured, encoded=encoded, correction_iterations=iterations)



def mix(spec,out,synthesis,visual):
    token=fingerprint(dict(spec=spec,wavs=[sha(p['wav']) for p in synthesis['parts']],implementation=sha(Path(__file__))))
    cache=out/'mix_report.json'
    if cache.exists():
        old=json.loads(cache.read_text())
        if old.get('fingerprint')==token and all((out/f).exists() for f in ('master.mp3','visualizer_input.json')):
            return old
    config=spec.get('mix',{});seconds=config.get('reverb_seconds',.8)
    sources={};bands={}
    lead=next((p['id'] for p in synthesis['parts'] if p['role']=='lead'),synthesis['parts'][0]['id'])
    for p in synthesis['parts']:
        x,sr=sf.read(p['wav'],dtype='float32');x=resample_poly(x,SR//math.gcd(sr,SR),sr//math.gcd(sr,SR))
        if p['role']=='bass':
            low=sosfiltfilt(butter(2,200,btype='lowpass',fs=SR,output='sos'),x)
            x=1.35*low+.75*(x-low)
        for note in p['notes']:
            if note['section']=='ending':
                a=round(note['start']*60/spec['bpm']*SR);b=round((note['start']+note['dur'])*60/spec['bpm']*SR)
                x[a:b]*=np.linspace(1,0,b-a);break
        sources[p['id']]=x;bands[p['id']]=band_rms(x,1000,4000)
    stems=[];reports=[];vmap={p['id']:p for p in visual}
    lead_band=max(bands[lead],1e-12)
    for i,p in enumerate(synthesis['parts']):
        key=p['id'];gain=1 if key==lead else min(100,lead_band*10**(p['gain_db']/20)/max(bands[key],1e-12))
        signal=sources[key]*gain
        wet=config.get('chord_reverb_wet_db',-15) if p['role']=='chord' else config.get('reverb_wet_db',-18)
        rv=diffuse_reverb(signal,wet,20261110+i,seconds)
        pan=np.full(len(rv),p['pan'])
        if key=='chord1':
            for n in p['notes']:
                if n['section']=='A3':
                    a=round(n['start']*60/spec['bpm']*SR);b=round((n['start']+n['dur'])*60/spec['bpm']*SR);pan[a:b]=-.2
        angle=(pan+1)*math.pi/4
        stereo=np.stack((rv*np.cos(angle),rv*np.sin(angle)),-1)
        stems.append(stereo);sf.write(out/'parts'/key/'mix.wav',stereo,SR,subtype='FLOAT')
        # Dry envelope follows this throat's activity; normalize per throat for visible 0.3..1.
        env=rms_frames(np.pad(signal,(0,len(rv)-len(signal))),round(SR/100))
        top=max(float(np.percentile(env[env>0],90)) if np.any(env>0) else 0,1e-12)
        levels=.3+.7*np.clip(env/top,0,1)**.5
        times=np.arange(math.ceil(len(rv)/SR*30))/30
        vp=vmap[key]
        for field in ('states','outer','inner','velum','areas','lengths','nasal_areas','nasal_lengths','ph_areas','or_areas','glottal_area','glottal_flow','f0'):
            values=vp[field];vp[field]=[values[min(len(values)-1,j)] for j in range(len(times))]
        vp['levels']=np.interp(times,np.arange(len(levels))/100,levels).tolist()
        reports.append(dict(id=key,name=p['name'],gain_linear=gain,gain_db=db(gain),pan=p['pan'],reverb_wet_db=wet,lead_advantage_1_4khz_db=db(lead_band)-db(bands[key]*gain)))
    total=np.sum(stems,axis=0)
    if not np.isfinite(total).all():raise RuntimeError('Nonfinite mix')
    source=out/'mix.wav';sf.write(source,total,SR,subtype='FLOAT')
    master=encode_mp3(source,out/'master.mp3',spec['title'],config.get('lufs',-14),config.get('dbtp',-1))
    visual_input=dict(version=2,sample_rate=24000,frame_rate=30,state_rate=30,levels_rate=30,bpm=spec['bpm'],beats_per_bar=spec['beats_per_bar'],total_beats=spec['length_beats'],song_seconds=spec['seconds'],audio_seconds=len(total)/SR,accuracy_unmet_audition=True,parts=visual)
    (out/'visualizer_input.json').write_text(json.dumps(visual_input,ensure_ascii=False,separators=(',',':'))+'\n')
    report=dict(fingerprint=token,master=master,parts=reports,audio_seconds=len(total)/SR,lead_1_4khz_advantage_db=min((p['lead_advantage_1_4khz_db'] for p in reports if p['id']!=lead),default=0),no_synthetic_instruments=True,visual_input_sha256=sha(out/'visualizer_input.json'))
    write(cache,report);return report
