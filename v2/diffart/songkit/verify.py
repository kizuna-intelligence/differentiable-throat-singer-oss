"""CPU autocorrelation pitch, timing, loudness, replay geometry and full video decode."""
from pathlib import Path
import json,math,re,subprocess,sys
import numpy as np
import soundfile as sf
from .pipeline import sha,write
SR=24000
HOP=240
CONFIDENCE=.03
FMIN=31.7
FMAX=2005.5
def measure_loudness(path: Path) -> dict:
    result = subprocess.run(['ffmpeg', '-hide_banner', '-nostats', '-i', str(path),
                             '-filter_complex', 'ebur128=peak=true', '-f', 'null', '-'],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    text = result.stderr
    integrated = re.findall(r'Integrated loudness:\s*I:\s*(-?[0-9.]+) LUFS', text)
    peaks = re.findall(r'True peak:\s*Peak:\s*(-?[0-9.]+) dBFS', text)
    return dict(integrated_lufs=float(integrated[-1]) if integrated else None,
                true_peak_dbtp=float(peaks[-1]) if peaks else None)


def pitch_report(estimator, part: dict, bpm: float, out: Path) -> dict:
    part_dir = out / 'parts' / part['id']
    audio, sr = sf.read(part_dir / f"{part['id']}.wav", dtype='float32')
    if sr != SR:
        raise RuntimeError(f'Unexpected sample rate in {part_dir}: {sr}')
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    normalized = audio / max(peak, 1e-8) * (0.9 if peak > 1.0 else 1.0)
    estimate, confidence = estimator.extract_continuous_periodicity(normalized.astype(np.float32))
    inputs = np.load(part_dir / 'inputs.npz')
    target = inputs['f0'].astype(np.float64)
    estimate = np.asarray(estimate, dtype=np.float64).reshape(-1)
    confidence = np.asarray(confidence, dtype=np.float64).reshape(-1)
    count = min(len(target), len(estimate), len(confidence))
    target, estimate, confidence = target[:count], estimate[:count], confidence[:count]
    representable = (target > 0) & (target >= FMIN) & (target <= FMAX)
    detected = representable & (estimate > 0) & (confidence >= CONFIDENCE)
    cents = np.full(count, np.nan, dtype=np.float64)
    cents[detected] = 1200 * np.log2(estimate[detected] / target[detected])
    within = detected & (np.abs(cents) <= 50)

    schedule = json.loads((part_dir / 'score.json').read_text())
    spb = 60.0 / bpm
    times = np.arange(count, dtype=np.float64) * HOP / SR
    note_rows = []
    note_values = []
    for note in schedule['notes']:
        if not note['pitched']:
            continue
        start = float(note['start']) * spb
        end = (float(note['start']) + float(note['dur'])) * spb
        mask = (times >= start + 0.02) & (times < end - 0.02) & representable
        valid = mask & (estimate > 0) & (confidence >= CONFIDENCE)
        if not np.any(mask):
            continue
        valid_cents = cents[valid]
        median = float(np.median(np.abs(valid_cents))) if len(valid_cents) else None
        within_rate = float(np.mean(np.abs(valid_cents) <= 50)) if len(valid_cents) else None
        note_rows.append(dict(beat=float(note['start']), midi=int(note['midi']), lyric=note['lyric'],
                              available_frames=int(mask.sum()), detected_frames=int(valid.sum()),
                              median_abs_cents=median, within_50_cents=within_rate))
        if median is not None:
            note_values.append(median)

    detected_cents = cents[detected]
    return dict(
        method='Project autocorrelation check; CPU; 24 kHz; 10 ms hop; NOT the historical RMVPE evaluator',
        notes_total=sum(bool(x['pitched']) for x in schedule['notes']),
        notes_measured=len(note_values),
        note_median_abs_cents=float(np.median(note_values)) if note_values else None,
        note_median_within_50_cents_rate=float(np.mean(np.asarray(note_values) <= 50)) if note_values else None,
        target_frames=int(representable.sum()), detected_frames=int(detected.sum()),
        voiced_recall=float(detected.sum() / max(1, representable.sum())),
        frames_within_50_cents=int(within.sum()),
        within_50_cents_share=float(within.sum() / max(1, representable.sum())),
        within_50_cents_given_detected=float(within.sum() / max(1, detected.sum())),
        median_abs_cents_detected=float(np.median(np.abs(detected_cents))) if len(detected_cents) else None,
        pitch_range_midi=[int(min(x['midi'] for x in schedule['notes'] if x['pitched']))
                          if any(x['pitched'] for x in schedule['notes']) else None,
                          int(max(x['midi'] for x in schedule['notes'] if x['pitched']))
                          if any(x['pitched'] for x in schedule['notes']) else None],
        per_note=note_rows)



def replay_area_check(out,visual):
    reports={}
    for part in visual['parts']:
        with np.load(out/'parts'/part['id']/'controls.npz') as controls:
            areas=controls['solver_areas'];volumes=controls['solver_volumes'];lengths=controls['solver_lengths'];tubes=json.loads(str(controls['tube_indices_json']))
            oral=tubes['pharynx']+tubes['oral'];nasal=tubes['nasal'];maximum=0
            for frame in range(len(part['areas'])):
                x=min(len(areas)-1,min(frame,part.get('source_video_frames',math.ceil(len(controls['states'])/100*30))-1)/30*1000);a=int(x);b=min(a+1,len(areas)-1);u=x-a
                for key,ix in [('areas',oral),('nasal_areas',nasal)]:
                    values=((1-u)*volumes[a,ix]+u*volumes[b,ix])/((1-u)*lengths[a,ix]+u*lengths[b,ix])
                    maximum=max(maximum,float(np.max(np.abs(values-part[key][frame]))))
            reports[part['id']]=dict(maximum_solver_to_replay_area_error_cm2=maximum,frames=len(part['areas']),nasal_sections=len(nasal),oral_sections=len(oral))
    return reports

def onset_report(out,part,bpm):
    audio,sr=sf.read(out/'parts'/part['id']/(part['id']+'.wav'),dtype='float32')
    hop=round(sr*.005);env=np.sqrt(np.mean(np.pad(audio,(0,(-len(audio))%hop)).reshape(-1,hop)**2,axis=1))
    slope=np.diff(env,prepend=env[0]);offsets=[];quant=[];rows=[]
    for n in part['notes']:
        t=n['start']*60/bpm
        quant.append(abs(round(t*100)/100-t)*1000)
        a=max(0,round((t-.06)/.005));b=min(len(slope),round((t+.08)/.005))
        if b<=a:continue
        ix=a+int(np.argmax(slope[a:b]));prom=float(slope[ix]);valid=prom>max(float(env.max())*.002,1e-15)
        if valid:offsets.append((ix*.005-t)*1000)
        rows.append(dict(beat=n['start'],scheduled_seconds=t,acoustic_energy_rise_seconds=ix*.005 if valid else None,offset_ms=(ix*.005-t)*1000 if valid else None,prominence=prom))
    return dict(method='5 ms RMS positive energy rise within -60..+80 ms of authored onset. Overlapping/internal syllables may have no distinct attack; these are excluded.',onsets=len(part['notes']),distinct_attacks=len(offsets),maximum_input_grid_quantization_ms=max(quant,default=0),median_attack_offset_ms=float(np.median(offsets)) if offsets else None,p95_absolute_attack_offset_ms=float(np.percentile(np.abs(offsets),95)) if offsets else None,per_event=rows)

def verify(spec,out,synthesis,mix,video):
    import torch
    from .pitch import AutocorrelationPitch
    torch.set_num_threads(2)
    model=AutocorrelationPitch()
    pitch={};timing={};loudness={}
    for p in synthesis['parts']:
        key=p['id'];pitch[key]=dict(excluded='unvoiced mouth percussion') if p['role']=='percussion' else pitch_report(model,p,spec['bpm'],out)
        timing[key]=onset_report(out,p,spec['bpm'])
        loudness[key]=measure_loudness(out/'parts'/key/'mix.wav')
        print('verified pitch '+key,flush=True)
    loudness['master']=measure_loudness(out/'master.mp3')
    raw=subprocess.check_output(['ffmpeg','-v','error','-i',str(out/'master.mp3'),'-f','f32le','-acodec','pcm_f32le','-'])
    decoded=np.frombuffer(raw,dtype=np.float32)
    clip=dict(finite=bool(np.isfinite(decoded).all()),sample_peak=float(abs(decoded).max()),full_scale_samples=int((abs(decoded)>=1).sum()))
    decode=subprocess.run(['ffmpeg','-v','error','-i',str(video),'-f','null','-'],capture_output=True,text=True)
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-show_streams','-show_format','-of','json',str(video)]))
    stream=next(s for s in probe['streams'] if s['codec_type']=='video');write(out/'video_ffprobe.json',probe)
    vis=json.loads((out/'visualizer_input.json').read_text());areas=replay_area_check(out,vis)
    validation=json.loads((out/'visualization/state_validation.json').read_text());pixels=json.loads((out/'visualization/pixel_comparison.json').read_text())
    l=loudness['master'];target=spec.get('mix',{})
    summary=dict(master_lufs=l['integrated_lufs'],master_true_peak_dbtp=l['true_peak_dbtp'],loudness_targets_met=abs(l['integrated_lufs']-target.get('lufs',-14))<=.5 and l['true_peak_dbtp']<=target.get('dbtp',-1),clipping=clip,stage3_pass=False,maximum_area_error_cm2=max(v['maximum_solver_to_replay_area_error_cm2'] for v in areas.values()),original_model_pixel_match_percent=pixels['metrics']['exact_pixel_match_percent'],video_decoded_frames=int(stream.get('nb_read_frames',0)),video_decode_success=decode.returncode==0 and not decode.stderr.strip(),pitch_within_50_cents_share={k:v.get('within_50_cents_share') for k,v in pitch.items()},motion_by_part=validation['parts'])
    report=dict(summary=summary,checkpoint=synthesis['checkpoint'],checkpoint_sha256=synthesis['checkpoint_sha256'],spec=dict(title=spec['title'],bpm=spec['bpm'],time_signature=spec['time_signature'],key=spec['key'],seconds=spec['seconds']),pitch_by_part=pitch,onsets_by_part=timing,loudness_by_part=loudness,solver_area_check=areas,visualization=validation,pixel_comparison=pixels,video=dict(file=str(video),sha256=sha(video),expected_frames=math.ceil(mix['audio_seconds']*30),decoded_frames=int(stream.get('nb_read_frames',0)),decode_stderr=decode.stderr,source_mp3_sha256=sha(out/'master.mp3')))
    checks=dict(
        solver_area=summary['maximum_area_error_cm2']<=1e-9,
        renderer_area=all(v['maximum_area_error_cm2']<=1e-9 for v in validation['parts'].values()),
        pixels=summary['original_model_pixel_match_percent']==100,
        motion=all(v['airway_raster_changed_transitions']>0 for v in validation['parts'].values()),
        loudness=summary['loudness_targets_met'],
        no_clipping=clip['finite'] and clip['full_scale_samples']==0,
        video=summary['video_decode_success'] and summary['video_decoded_frames']==math.ceil(mix['audio_seconds']*30))
    report['acceptance_checks']=checks;summary['artifact_checks_pass']=all(checks.values())
    write(out/'verification_report.json',report)
    if not all(checks.values()):raise RuntimeError(f'Artifact checks failed: {checks}')
    return report
