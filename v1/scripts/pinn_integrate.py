"""Isolated C correction: short validation, one epoch, held-out voice comparison.

Never writes the legacy checkpoint or evaluation paths. Speaker metrics use the
same anchors, 20s windows, silence exclusions and song weighting as evaluate.py.
"""
import argparse
import json
from pathlib import Path
import random
import time
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import torch
from diffart.data import ROOT,SR,HOP,sha
from diffart.dataset import Corpus
from diffart.physics import Synthesizer
from diffart.improved import ImprovedSynthesizer
from diffart.pinn.correction import FEMCorrection,load_synthesizer
from diffart.losses import SpectralLoss
from diffart.render import render


def save(path,data):
    Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')


def validation_blocks(c):
    rng=random.Random(20261109)
    blocks=[]
    for song in c.manifest['songs']:
        if song['split']!='val':continue
        n=len(c.load(song)['f0']);start=rng.randint(0,max(0,n-3000))
        blocks.extend((song,a) for a in range(start,min(start+3000,n)-99,100))
    return blocks


def loss_validation(model,c,blocks,criterion,device):
    model.eval();values=[]
    with torch.no_grad():
        for song,a in blocks:
            data=c.excerpt(song,a,100,device)
            if data[-1].square().mean()<1e-7:continue
            y=model(*data[:3],seed=1234)
            loss,_=criterion(y,data[-1]);values.append(float(loss))
    return float(np.mean(values))


def train(args,c):
    out=args.out;device=args.device
    correction=FEMCorrection.load(args.surrogate)
    meta=torch.load(args.surrogate,map_location='cpu',weights_only=True)['metadata']
    model=ImprovedSynthesizer(len(c.manifest['vocab']),pinn_correction=correction).to(device)
    base=torch.load(ROOT/'artifacts/main/best.pt',map_location=device,weights_only=True)
    model.load_legacy(base['model'])
    criterion=SpectralLoss().to(device)
    blocks=validation_blocks(c)
    save(out/'validation_blocks.json',[dict(song=r['id'],start_frame=a,frames=100) for r,a in blocks])
    legacy=Synthesizer(len(c.manifest['vocab'])).to(device);legacy.load_state_dict(base['model'])
    cached=out/'losses.json'
    previous=json.loads(cached.read_text()) if cached.exists() else {}
    if previous.get('surrogate_sha256')==sha(args.surrogate):
        baseline=previous['baseline_loss'];corrected=previous['correction_only_loss']
        del legacy
    else:
        baseline=loss_validation(legacy,c,blocks,criterion,device);del legacy
        corrected=loss_validation(model,c,blocks,criterion,device)
    torch.save(dict(model=model.state_dict(),correction_metadata=meta,variant='correction_only',
                    base_sha256=sha(ROOT/'artifacts/main/best.pt')),out/'correction_only.pt')
    summary=dict(baseline_loss=baseline,correction_only_loss=corrected,validation_chunks=len(blocks),
                 surrogate_sha256=sha(args.surrogate),device=device)
    save(out/'losses.json',summary);print(json.dumps(summary),flush=True)
    if args.smoke:
        summary['smoke']=True;save(out/'losses.json',summary)
        return
    parameters=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.Adam(parameters,lr=args.lr)
    trainblocks=[(r,a) for r in c.manifest['songs'] if r['split']=='train' and r.get('train_seconds',0)>0
                 for a in range(int(r['train_seconds']) )]
    # Exact one-second partition, one pass over every complete block of the hour.
    trainblocks=[(r,a*100) for r,a in trainblocks]
    random.Random(20261109).shuffle(trainblocks)
    start=time.monotonic();model.train()
    initial_geometry=dict(length=float(model.speaker.length.detach()),wall=float(model.speaker.wall.detach()))
    with (out/'training.jsonl').open('w') as log:
        for cursor in range(0,len(trainblocks),args.batch):
            batch=trainblocks[cursor:cursor+args.batch]
            data=[c.excerpt(r,a,100,device) for r,a in batch]
            tensors=[torch.stack([v[j] for v in data]) for j in range(4)]
            optimizer.zero_grad(set_to_none=True)
            y,p=model(*tensors[:3],seed=cursor+1,return_controls=True)
            acoustic,parts=criterion(y,tensors[3]);loss=acoustic+model.regularization(p)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(parameters,5.)
            if not torch.isfinite(norm):raise RuntimeError('Nonfinite gradient')
            optimizer.step()
            if cursor//20!=(cursor+len(batch))//20:
                row=dict(samples=cursor+len(batch),epoch=(cursor+len(batch))/len(trainblocks),
                         train_loss=float(loss.detach()),seconds=time.monotonic()-start)
                log.write(json.dumps(row)+'\n');log.flush();print(json.dumps(row),flush=True)
    final=loss_validation(model,c,blocks,criterion,device)
    torch.save(dict(model=model.state_dict(),correction_metadata=meta,variant='one_epoch',
                    base_sha256=sha(ROOT/'artifacts/main/best.pt')),out/'one_epoch.pt')
    summary.update(one_epoch_loss=final,epoch_samples=len(trainblocks),epoch_seconds=time.monotonic()-start,
                   learning_rate=args.lr,geometry_before=initial_geometry,
                   geometry_after=dict(length=float(model.speaker.length.detach()),wall=float(model.speaker.wall.detach())))
    if device.startswith('cuda'):
        summary['cuda_peak_reserved_gb']=torch.cuda.max_memory_reserved()/2**30
        summary['cuda_peak_allocated_gb']=torch.cuda.max_memory_allocated()/2**30
    save(out/'losses.json',summary);print(json.dumps(summary),flush=True)


def metrics_setup(out,device='cpu'):
    from diffart.metrics import SpeakerMetrics,EcapaMetrics
    metrics={'wavlm':SpeakerMetrics(device)}
    import torchaudio
    if not hasattr(torchaudio,'list_audio_backends'):torchaudio.list_audio_backends=lambda:['soundfile']
    from speechbrain.inference.speaker import EncoderClassifier
    source=ROOT/'artifacts/models/ecapa'
    ecapa=EcapaMetrics.__new__(EcapaMetrics);ecapa.device=device
    ecapa.model=EncoderClassifier.from_hparams(source=str(source),savedir=str(out/'ecapa_cache'),
        overrides={'pretrained_path':str(source)},run_opts={'device':device})
    metrics['ecapa']=ecapa
    refs=json.loads((ROOT/'artifacts/evaluation/references.json').read_text())
    waves=[]
    for row in refs:
        path=Path(row['path'])
        if not path.exists():path=ROOT/'artifacts/evaluation'/path.name
        y,sr=sf.read(path,dtype='float32');assert sr==SR
        waves.append(resample_poly(y,2,3))
    anchors={key:np.stack([metric.embedding(y) for y in waves]) for key,metric in metrics.items()}
    return metrics,anchors


def move_metrics(metrics,device):
    for key,metric in metrics.items():
        metric.model.to(device);metric.device=device
        if key=='ecapa':metric.model.device=torch.device(device)


def evaluate(args,c):
    variant=args.variant
    if variant=='baseline':
        model=Synthesizer(len(c.manifest['vocab']))
        model.load_state_dict(torch.load(ROOT/'artifacts/main/best.pt',map_location='cpu',weights_only=True)['model'])
    else:model=load_synthesizer(args.out/f'{variant}.pt',len(c.manifest['vocab']))
    model=model.eval()
    metrics,anchors=metrics_setup(args.out,args.device if args.gpu_metrics else 'cpu')
    if args.gpu_metrics:
        move_metrics(metrics,'cpu');torch.cuda.empty_cache()
    model=model.to(args.device)
    rows=[];out=args.out/variant;out.mkdir(exist_ok=True)
    for song in [r for r in c.manifest['songs'] if r['split']=='test']:
        d=c.load(song);target=d['y'];n=len(d['f0'])
        if args.gpu_metrics:
            move_metrics(metrics,'cpu');torch.cuda.empty_cache();model.to(args.device)
        if variant!='baseline':
            with torch.no_grad():
                controls=model.controls(torch.tensor(d['context'][::10],device=args.device),
                    torch.tensor(d['pos'][::10],device=args.device),torch.tensor(d['f0'][::10],device=args.device),model.speaker)
                area=model.speaker.area(controls['art'])
                maxima=[float(model.pinn_correction.grid(a)[...,0].max().exp()) for a in area.split(50)]
                save(out/f"{song['id']}__ratio_range.json",dict(sampled_frames=len(area),max_abs=max(maxima)))
        conditions=[('reproduction',0)] if args.reproduction_only else [('reproduction',0),('score',0),('score',12)]
        for kind,shift in conditions:
            path=out/f"{song['id']}__{kind}_{shift:+d}.wav"
            result=path.with_suffix('.json')
            if result.exists():rows.append(json.loads(result.read_text()));continue
            f0=(d['f0'] if kind=='reproduction' else d['score_f0'])*2**(shift/12)
            if path.exists():y,sr=sf.read(path,dtype='float32');assert sr==SR
            else:
                if args.gpu_metrics:
                    move_metrics(metrics,'cpu');torch.cuda.empty_cache();model.to(args.device)
                y=render(model,d['context'],d['pos'],f0,device=args.device)
                assert np.isfinite(y).all();sf.write(path,y,SR,subtype='FLOAT')
            if args.gpu_metrics:
                model.cpu();torch.cuda.empty_cache();move_metrics(metrics,args.device)
            segments=[]
            for a in range(0,n,2000):
                b=min(a+2000,n)
                if b-a<100 or np.mean(target[a*HOP:b*HOP]**2)<1e-7:continue
                y16=resample_poly(y[a*HOP:b*HOP],2,3)
                segment=dict(start_seconds=a*.01,end_seconds=b*.01)
                for key,metric in metrics.items():
                    emb=metric.embedding(y16);segment[key]=float((anchors[key]@emb).mean())
                segments.append(segment)
            row=dict(variant=variant,song=song['id'],condition=kind,semitones=shift,
                     wavlm_cosine=float(np.mean([s['wavlm'] for s in segments])),
                     ecapa_cosine=float(np.mean([s['ecapa'] for s in segments])),segments=segments)
            rows.append(row);save(result,row);save(out/'results.json',rows)
            print(json.dumps({k:v for k,v in row.items() if k!='segments'}),flush=True)
    summary=[]
    for kind,shift in conditions:
        part=[r for r in rows if r['condition']==kind and r['semitones']==shift]
        summary.append(dict(condition=kind,semitones=shift,songs=len(part),
            wavlm_cosine=float(np.mean([r['wavlm_cosine'] for r in part])),
            ecapa_cosine=float(np.mean([r['ecapa_cosine'] for r in part]))))
    save(out/'summary.json',summary);print(json.dumps(summary),flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('action',choices=['train','evaluate'])
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--surrogate',type=Path)
    ap.add_argument('--device',default='cpu')
    ap.add_argument('--threads',type=int,default=4)
    ap.add_argument('--batch',type=int,default=2)
    ap.add_argument('--lr',type=float,default=.0002)
    ap.add_argument('--memory-fraction',type=float,default=.10)
    ap.add_argument('--smoke',action='store_true')
    ap.add_argument('--reproduction-only',action='store_true')
    ap.add_argument('--gpu-metrics',action='store_true',help='Offload synth/metrics in turn to keep GPU memory bounded')
    ap.add_argument('--variant',choices=['baseline','correction_only','one_epoch'],default='correction_only')
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(args.threads);torch.manual_seed(20261109)
    if args.device.startswith('cuda'):torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    c=Corpus()
    if args.action=='train':train(args,c)
    else:evaluate(args,c)


if __name__=='__main__':main()
