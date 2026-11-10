"""Resumable specification -> independent throats -> mix -> replay -> verification."""
from __future__ import annotations
import copy, gc, hashlib, inspect, json, math, shutil, subprocess, time
from pathlib import Path, PureWindowsPath
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from . import cpu
from .frontend import inputs
from .patterns import expand

REPO=Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT='models/throat_ritsu_v2_chimera.pt'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
def fingerprint(data):return hashlib.sha256(json.dumps(data,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
def run(cmd):subprocess.run([str(x) for x in cmd],check=True)

def _safe_path_component(value,label):
    """Reject spec values that could escape their output directory on any OS."""
    if (not isinstance(value,str) or not value or value in ('.','..')
            or any(ch in value for ch in ('/','\\','\0'))
            or any(ord(ch)<32 for ch in value)
            or value.endswith((' ','.'))
            or Path(value).is_absolute()
            or PureWindowsPath(value).is_absolute()
            or PureWindowsPath(value).drive):
        raise ValueError(f'{label} must be a single safe filename component')
    return value

def load_spec(path):
    if path.suffix in ('.yaml','.yml'):
        import yaml
        spec=yaml.safe_load(path.read_text())
    else:spec=json.loads(path.read_text())
    if 'output_name' in spec:spec['output_name']=_safe_path_component(spec['output_name'],'output_name')
    num,den=map(int,spec['time_signature'].split('/'))
    spec['beats_per_bar']=num*4/den
    spec.setdefault('length_beats',sum(s['bars'] for s in spec['sections'])*spec['beats_per_bar'])
    spec['seconds']=spec['length_beats']*60/spec['bpm']
    ids=[_safe_path_component(p.get('id'),'part id') for p in spec['parts']]
    if len(set(ids))!=len(ids):raise ValueError('Part IDs must be unique')
    for i,p in enumerate(spec['parts']):
        p.setdefault('name',p['id']);p.setdefault('body_scale',1.);p.setdefault('pan',0.);p.setdefault('gain_db',0 if p.get('role')=='lead' else -6);p['throat']=i+1
    return expand(spec)

def frame_replay(controls,f0,audio):
    """Sample coefficient interpolation at exact video sample instants, without rounding."""
    count=math.ceil(len(audio)/24000*30);times=np.arange(count)/30
    # Solver does piecewise linear compliance and length interpolation at 1 kHz.
    areas=controls['solver_areas'];lengths=controls['solver_lengths']
    def interp(series,rate):
        ix=np.clip(times*rate,0,len(series)-1);a=np.floor(ix).astype(int);b=np.minimum(a+1,len(series)-1)
        u=(ix-a).reshape((-1,)+(1,)*(series.ndim-1))
        return series[a]*(1-u)+series[b]*u
    # Stored area equals effective volume/length. Both are interpolated by solver.
    # Only tube nodes have an axial length (lumped lung/sinus cavities do not).
    volumes=controls['solver_volumes'] if 'solver_volumes' in controls else np.nan_to_num(areas*lengths,nan=0.,posinf=0.)
    sampled_volume=interp(volumes,1000);sampled_length=interp(lengths,1000)
    ar=np.divide(sampled_volume,sampled_length,out=np.zeros_like(sampled_volume),where=sampled_length>0)
    dx=interp(lengths,1000)
    tubes=json.loads(str(controls['tube_indices_json']))
    oral=tubes['pharynx']+tubes['oral'];ph=tubes['pharynx'];orr=tubes['oral'];nas=tubes['nasal']
    def geometric_replay(series):
        # Match align_corners=False 100 Hz -> 1 kHz (then the solver's sample interpolation).
        coordinates=np.clip((np.arange(len(series)*10)+.5)/10-.5,0,len(series)-1)
        a=np.floor(coordinates).astype(int);b=np.minimum(a+1,len(series)-1)
        u=(coordinates-a).reshape((-1,)+(1,)*(series.ndim-1))
        up=series[a]*(1-u)+series[b]*u
        return interp(up,1000).tolist()
    data={k:geometric_replay(controls[k]) for k in ('states','outer','inner','velum')}
    data.update(source_video_frames=count,areas=ar[:,oral].tolist(),lengths=dx[:,oral].tolist(),nasal_areas=ar[:,nas].tolist(),nasal_lengths=dx[:,nas].tolist(),ph_areas=ar[:,ph].tolist(),or_areas=ar[:,orr].tolist(),scale=float(controls['scale']),body=json.loads(str(controls['body_json'])),n_ph=len(ph))
    # Two cycles of the actual audio-rate source area/flow, starting at the video sample.
    ga=controls['glottal_area'];gf=controls['glottal_flow'];rate=int(controls['glottal_rate'])
    ga_rows=[];gf_rows=[]
    for t in times:
        hz=float(f0[min(len(f0)-1,round(t*100))]);length=round(2*rate/max(50,hz))
        ix=min(len(ga)-1,round(t*rate));end=min(len(ga),ix+length)
        if end-ix<2:ix=max(0,len(ga)-length);end=len(ga)
        # Resampling only the oscilloscope trace, not sound or area functions.
        axis=np.linspace(ix,end-1,256)
        ga_rows.append(np.interp(axis,np.arange(ix,end),ga[ix:end]).tolist())
        gf_rows.append(np.interp(axis,np.arange(ix,end),gf[ix:end]).tolist())
    data.update(glottal_area=ga_rows,glottal_flow=gf_rows,f0=interp(f0,100).tolist())
    return data

def synthesize(spec,out,checkpoint):
    torch.set_num_threads(4);cpu.CACHE=out/'cpu_cache'
    ck=torch.load(checkpoint,map_location='cpu',weights_only=True);vocab=ck['config'].get('vocab')
    if vocab is None:vocab=json.loads((REPO/'data/manifest.json').read_text())['vocab']
    parts=[];visual=[]
    source_hash=fingerprint({str(p.relative_to(REPO)):sha(p) for p in [REPO/'diffart/songkit/cpu.py',REPO/'diffart/songkit/frontend.py']+list((REPO/'diffart/chimera').glob('*.py'))})
    for index,part in enumerate(spec['parts']):
        folder=out/'parts'/part['id'];folder.mkdir(parents=True,exist_ok=True)
        token=fingerprint(dict(part=part,bpm=spec['bpm'],seconds=spec['seconds'],checkpoint=sha(checkpoint),source=source_hash))
        report_path=folder/'synthesis.json';wav=folder/(part['id']+'.wav')
        reused=False
        if report_path.exists():
            record=json.loads(report_path.read_text())
            reused=record.get('fingerprint')==token and all((folder/k).exists() for k in ('controls.npz','inputs.npz')) and wav.exists() and sha(wav)==record['wav_sha256']
        if not reused:
            ctx,pos,f0,rows=inputs(spec,part,vocab)
            np.savez_compressed(folder/'inputs.npz',context=ctx,pos=pos,f0=f0)
            write(folder/'rows.json',[dict(start=float(a),end=float(b),phone=p) for a,b,p in rows]);write(folder/'score.json',dict(part={k:v for k,v in part.items() if k!='notes'},notes=part['notes']))
            model=cpu.make_model(ck['config'],vocab,ck['model'],part['body_scale'])
            started=time.time();audio,controls,record=cpu.cpu_render(model,ctx,pos,f0,1234+index)
            sf.write(wav,audio,24000,subtype='FLOAT');np.savez_compressed(folder/'controls.npz',**controls)
            record.update(fingerprint=token,wav_sha256=sha(wav),elapsed_seconds=time.time()-started,checkpoint_sha256=sha(checkpoint))
            write(report_path,record);del model,controls;gc.collect()
        audio,sr=sf.read(wav,dtype='float32');f0=np.load(folder/'inputs.npz')['f0']
        with np.load(folder/'controls.npz') as controls:data=frame_replay(controls,f0,audio)
        data.update({k:v for k,v in part.items() if k not in ('notes','patterns')});visual.append(data)
        parts.append(dict(part,wav=str(wav.resolve()),render=record,reused=reused))
        print(f"synthesis {part['id']}: {'reused' if reused else 'CPU rendered'}",flush=True)
    report=dict(checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),accuracy_unmet_audition=True,stage3_pass=False,parts=parts)
    write(out/'synthesis_report.json',report)
    return report,visual

def execute(spec_path,out,downloads=None,audio_only=False):
    out.mkdir(parents=True,exist_ok=True);spec=load_spec(spec_path);write(out/'expanded_score.json',spec)
    checkpoint=Path(spec.get('checkpoint',DEFAULT_CHECKPOINT));checkpoint=checkpoint if checkpoint.is_absolute() else REPO/checkpoint
    synthesis,visual=synthesize(spec,out,checkpoint)
    from .mix import mix
    mix_report=mix(spec,out,synthesis,visual)
    if audio_only:
        report=dict(summary=dict(audio_only=True,stage3_pass=False,parts=len(synthesis['parts']),finite=all(p['render'].get('internal_samples',0)>0 for p in synthesis['parts'])),checkpoint_sha256=synthesis['checkpoint_sha256'])
        write(out/'verification_report.json',report)
        print(json.dumps(report,ensure_ascii=False),flush=True)
        return report
    mp4=out/(spec['output_name']+'_喉の動き.mp4')
    visual_token=fingerprint(dict(input=sha(out/'visualizer_input.json'),audio=sha(out/'master.mp3'),sources={str(p.relative_to(REPO)):sha(p) for p in (REPO/'visualizer').rglob('*') if p.is_file() and 'node_modules' not in p.parts and p.suffix in ('.ts','.mjs','.html')}))
    cache=out/'video_cache.json'
    cached=json.loads(cache.read_text()) if cache.exists() else {}
    if cached.get('fingerprint')==visual_token and cached.get('file'): mp4=Path(cached['file'])
    elif mp4.exists():mp4=out/(spec['output_name']+'_'+visual_token[:8]+'_喉の動き.mp4')
    if not (cache.exists() and json.loads(cache.read_text()).get('fingerprint')==visual_token and mp4.exists() and sha(mp4)==cached.get('sha256')):
        run(['node',REPO/'visualizer/render.mjs',out/'visualizer_input.json',out/'master.mp3',mp4]);write(cache,dict(fingerprint=visual_token,sha256=sha(mp4),file=str(mp4)))
    from .verify import verify
    report=verify(spec,out,synthesis,mix_report,mp4)
    if downloads:
        downloads.mkdir(parents=True,exist_ok=True);published=[]
        for src,name in [(out/'master.mp3',spec['output_name']+'.mp3'),(mp4,spec['output_name']+'_喉の動き.mp4')]:
            dst=downloads/name
            if dst.exists() and sha(src)==sha(dst):published.append(str(dst));continue
            if dst.exists():
                stem=dst.stem; suffix=dst.suffix;i=2
                while dst.exists():dst=downloads/f'{stem}_{i}{suffix}';i+=1
            with dst.open('xb') as f,src.open('rb') as g:shutil.copyfileobj(g,f)
            assert sha(src)==sha(dst);published.append(str(dst))
        report['published']=published;write(out/'verification_report.json',report)
    print(json.dumps(dict(out=str(out),summary=report['summary'],published=report.get('published',[])),ensure_ascii=False),flush=True)
    return report
