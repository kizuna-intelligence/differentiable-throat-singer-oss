"""Stage-gated training, fixed physical model, checkpoint selection by validation.

CPU is supported; CUDA requires a UUID shared lock before importing torch.
"""
import argparse
import json
import random
import time
from pathlib import Path
from diffart.runtime import device_lease, save_json
from diffart.data import ROOT,SR,HOP,sha

def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['overfit','main'])
    ap.add_argument('--device',default='cpu');ap.add_argument('--steps',type=int,default=1200)
    ap.add_argument('--threads',type=int,default=3);ap.add_argument('--resume',action='store_true')
    ap.add_argument('--batch-size',type=int,default=1);ap.add_argument('--seconds',type=float,default=1.0);ap.add_argument('--patience',type=int,default=8)
    args=ap.parse_args()
    assert (ROOT/'artifacts/STAGE1_PASS.json').exists(),'Stage 1 has not passed'
    units=json.loads((ROOT/'artifacts/unit_results.json').read_text())
    assert units.get('passed') is True,'Stage 2 has not passed'
    assert units.get('physics_sha256')==sha(ROOT/'diffart/physics.py'),'Physics changed since unit tests'
    if args.stage=='main':
        gate=json.loads((ROOT/'artifacts/OVERFIT_GATE.json').read_text())
        assert gate.get('passed') is True,'Overfit gate not passed (including intelligibility)'
        assert gate.get('checkpoint_sha256')==sha(ROOT/'artifacts/overfit/best.pt'),'Reviewed checkpoint has changed'
    with device_lease(args.device) as device:
        import torch
        import soundfile as sf
        from diffart.dataset import Corpus
        from diffart.physics import Synthesizer
        from diffart.losses import SpectralLoss
        torch.set_num_threads(args.threads);torch.manual_seed(20261109)
        if device.startswith('cuda'):torch.cuda.set_per_process_memory_fraction(.15)
        corpus=Corpus();m=corpus.manifest
        model=Synthesizer(len(m['vocab'])).to(device);criterion=SpectralLoss().to(device)
        out=ROOT/'artifacts'/args.stage;out.mkdir(exist_ok=True)
        rng=random.Random(20261109);optimizer=torch.optim.Adam(model.parameters(),lr=.002 if args.stage=='overfit' else .0005)
        songs=[r for r in m['songs'] if r['split']=='train' and r.get('train_seconds',0)>0]
        # Selection is restricted to the sampled hour, including final partial song.
        phrases=[]
        for song in songs:
            phrase=corpus.phrase(song)
            if phrase and (phrase['start_frame']+phrase['frames'])*HOP/SR<=song['train_seconds']:
                phrases.append((song,phrase))
        song,phrase=phrases[0];save_json(out/'phrase.json',phrase)
        if args.stage=='overfit':
            fixed=corpus.excerpt(song,phrase['start_frame'],phrase['frames'],device)
            sf.write(out/'reference.wav',fixed[-1].cpu().numpy(),SR,subtype='FLOAT')
        else:
            model.load_state_dict(torch.load(ROOT/'artifacts/overfit/best.pt',map_location=device,weights_only=True)['model'])
        first=0;best=float('inf');stale=0;initial=None;ck=None
        if args.resume and (out/'latest.pt').exists():
            ck=torch.load(out/'latest.pt',map_location=device,weights_only=True)
            assert ck['manifest_sha256']==sha(ROOT/'data/manifest.json')
            model.load_state_dict(ck['model']);optimizer.load_state_dict(ck['optimizer'])
            first=ck['step'];best=ck['best'];stale=ck.get('stale',0);initial=ck['initial'];rng.setstate(ck['rng']);torch.set_rng_state(ck['torch_rng'].cpu())
        def validation():
            model.eval();values=[]
            with torch.no_grad():
                if args.stage=='overfit':
                    y=model(*fixed[:3],seed=1234);loss,parts=criterion(y,fixed[3]);values.append(float(loss))
                else:
                    # Cover every complete nonoverlapping 1s block of all four val songs.
                    for row in [r for r in m['songs'] if r['split']=='val']:
                        n=len(corpus.load(row)['f0']);size=int(args.seconds*SR/HOP)
                        for a in range(0,n-size+1,size):
                            data=corpus.excerpt(row,a,size,device)
                            if data[-1].square().mean()<1e-7:continue
                            y=model(*data[:3],seed=1234);loss,_=criterion(y,data[3]);values.append(float(loss))
            model.train();return sum(values)/len(values)
        frames=int(args.seconds*SR/HOP)
        blocks=[(r,a) for r in songs for a in range(0,int(r['train_seconds']*SR/HOP)-frames+1,frames)]
        if ck and 'block_order' in ck:
            by_id={r['id']:r for r in songs};blocks=[(by_id[sid],a) for sid,a in ck['block_order']]
        if initial is None:initial=validation()
        start=time.monotonic();log=(out/'training.jsonl').open('a')
        history=[]
        def checkpoint(step,val,baseline=False):
            nonlocal best,stale
            improved=val<best-.002
            if improved:best=val;stale=0
            elif not baseline:stale+=1
            ck=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),step=step,best=best,
               initial=initial,manifest_sha256=sha(ROOT/'data/manifest.json'),rng=rng.getstate(),torch_rng=torch.get_rng_state(),args=vars(args),stale=stale,block_order=[(r['id'],a) for r,a in blocks])
            torch.save(ck,out/'latest.pt')
            if improved:torch.save(ck,out/'best.pt')
            metric=dict(step=step,val_loss=val,best=best,initial=initial,relative=val/initial,elapsed=time.monotonic()-start)
            history.append(metric);log.write(json.dumps(metric)+'\n');log.flush();print(json.dumps(metric),flush=True)
            save_json(out/'progress.json',metric)
            if args.stage=='overfit':
                with torch.no_grad():y=model(*fixed[:3],seed=1234)
                sf.write(out/f'step{step:05d}.wav',y.cpu().numpy(),SR,subtype='FLOAT')
        checkpoint(first,validation() if first else initial,baseline=bool(first))
        cursor=first
        while cursor<args.steps:
            if args.stage=='overfit':
                count=1
                offset=rng.randrange(max(1,phrase['frames']-frames+1))
                data=corpus.excerpt(song,phrase['start_frame']+offset,min(frames,phrase['frames']),device)
            else:
                if cursor%len(blocks)==0:rng.shuffle(blocks)
                count=min(args.batch_size,len(blocks)-cursor%len(blocks),args.steps-cursor)
                examples=[corpus.excerpt(*blocks[(cursor+i)%len(blocks)],frames,device) for i in range(count)]
                data=tuple(torch.stack([x[j] for x in examples]) for j in range(4)) if count>1 else examples[0]
            step=cursor+count
            model.train();optimizer.zero_grad(set_to_none=True)
            y,p=model(*data[:3],seed=step,return_controls=True)
            acoustic,parts=criterion(y,data[-1]);reg=model.regularization(p);loss=acoustic+reg
            if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite loss at {step}')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            if not torch.isfinite(norm):raise RuntimeError(f'Nonfinite gradient at {step}')
            optimizer.step()
            if step//10!=cursor//10:
                log.write(json.dumps(dict(step=step,train_loss=float(loss.detach()),grad_norm=float(norm),parts={k:float(v.detach()) for k,v in parts.items()},source_noise_gain=float((.03*p['amplitude']*p['breath']).mean().detach()),constriction_noise_gain=float((.03*p['amplitude']*p['turbulence']).mean().detach())))+'\n');log.flush()
            cursor=step
            interval=100 if args.stage=='overfit' else len(blocks)
            if step%interval==0 or step==args.steps:
                checkpoint(step,validation())
                if args.stage=='main' and stale>=args.patience:break
        log.close()
        result=dict(initial=initial,best=best,ratio=best/initial,step=step,elapsed=time.monotonic()-start,device=device,
            dataset_seconds=3600 if args.stage=='main' else phrase['frames']*HOP/SR,
            epoch_blocks=len(blocks),validation_history=history,completed_epochs=step/len(blocks),batch_size=args.batch_size,stopping_reason='validation_patience' if stale>=args.patience else 'maximum_sampled_blocks',checkpoint_sha256=sha(out/'best.pt'))
        save_json(out/'result.json',result)
        if args.stage=='overfit':
            # Numerical improvement is necessary; perceptual/phonetic gate is separate.
            save_json(ROOT/'artifacts/OVERFIT_GATE.json',dict(passed=False,
                loss_pass=best/initial<=.65,loss_ratio=best/initial,checkpoint_sha256=sha(out/'best.pt'),
                intelligibility='pending: human audition of reference.wav vs best.wav required',
                reason='Do not mark listening criterion passed from loss alone'))
            model.load_state_dict(torch.load(out/'best.pt',map_location=device,weights_only=True)['model'])
            with torch.no_grad():y=model(*fixed[:3],seed=1234)
            sf.write(out/'best.wav',y.cpu().numpy(),SR,subtype='FLOAT')
if __name__=='__main__':main()
