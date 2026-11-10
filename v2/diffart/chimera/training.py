"""Singing and speech share the main_round2 objective and Adam loop."""
import json
import math
import random
from pathlib import Path
import numpy as np
import torch
from .voice import HumanVoice
from .objective import SpectralLoss

SEED = 20261109


def default_config(vocab):
    """Corpus-independent physical defaults; never read a voice checkpoint."""
    return dict(vocab=dict(vocab), mode='parametric', sample_rate=96000,
                sections=30, control_rate=1000, source_controls=True,
                source_shape_controls=True, wall_controls=True,
                conservative_geometry=True, tongue_modes=4, nasal_controls=True,
                correction_coefficient=1.3067257008338073, nasal_correction_coefficient=0.,
                precise_phase=True, precision='torch.float32')


def make_model(config, state=None, device='cpu'):
    model=HumanVoice(config['vocab']).to(device=device,dtype=torch.float32)
    model.glottis.precise_phase=config.get('precise_phase',True)
    model.airway.correction_coefficient=float(config['correction_coefficient'])
    if state is not None:
        model.load_state_dict(state)
    return model


def save_checkpoint(path, model, config, **extra):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp')
    torch.save(dict(model=model.state_dict(),config=config,**extra),temporary)
    temporary.replace(path)


class Corpus:
    def __init__(self, directory, frames=100):
        self.root=Path(directory).resolve()
        self.manifest=json.loads((self.root/'manifest.json').read_text())
        self.vocab=self.manifest['vocab'];self.frames=frames;self.data={}
        self.blocks={s:[] for s in ('train','val','test')}
        for row in self.manifest['items']:
            path=(self.root/row['file']).resolve()
            if not path.is_relative_to(self.root):
                raise ValueError('Prepared file escapes the dataset directory')
            with np.load(path,allow_pickle=False) as archive:
                d={k:archive[k] for k in ('context','pos','f0','audio')}
            n=len(d['f0'])
            if (d['context'].shape!=(n,3) or d['pos'].shape!=(n,) or
                d['audio'].shape!=(n*240,) or not all(np.isfinite(v).all() for v in d.values()) or
                np.any(d['context']<0) or np.any(d['context']>=len(self.vocab)) or
                np.any(d['f0']<0) or np.any((d['pos']<0)|(d['pos']>1))):
                raise ValueError('Invalid prepared acoustic frames')
            key=row['file']
            if key in self.data:
                raise ValueError('Duplicate prepared item')
            self.data[key]=d
            self.blocks[row['split']].extend((key,i) for i in range(0,n-frames+1,frames))
        if not self.blocks['train'] or not self.blocks['val']:
            raise ValueError('Separate train and val blocks are required')

    def batch(self, items, device):
        result=[]
        for key in ('context','pos','f0','audio'):
            arrays=[]
            for name,start in items:
                scale=240 if key=='audio' else 1
                arrays.append(self.data[name][key][start*scale:(start+self.frames)*scale])
            result.append(torch.as_tensor(np.stack(arrays),device=device))
        return tuple(result)


def train_step(model, optimizer, criterion, batch, seed):
    model.train();optimizer.zero_grad(set_to_none=True)
    wave, controls=model(*batch[:3],seed=seed,return_controls=True)
    acoustic=criterion(wave,batch[3])[0]
    loss=acoustic+model.regularization(controls)
    if not torch.isfinite(loss):
        raise RuntimeError('Nonfinite training objective')
    loss.backward()
    parameters=[p for p in model.parameters() if p.grad is not None]
    norm=torch.stack([p.grad.double().square().sum() for p in parameters]).sum().sqrt()
    if not torch.isfinite(norm):
        raise RuntimeError('Nonfinite gradient')
    factor=5/norm.clamp_min(5)
    for p in parameters:
        p.grad.mul_(factor)
    optimizer.step()
    return float(loss.detach())


@torch.no_grad()
def evaluate(model, corpus, split='val', batch_size=16, device='cpu'):
    model.eval();criterion=SpectralLoss().to(device);values=[]
    for cursor in range(0,len(corpus.blocks[split]),batch_size):
        d=corpus.batch(corpus.blocks[split][cursor:cursor+batch_size],device)
        wave=model(*d[:3],seed=1234)
        for x,y in zip(wave,d[3]):
            if y.square().mean()>=1e-7:
                values.append(float(criterion(x,y)[0]))
    if not values:
        raise ValueError('No non-silent evaluation blocks')
    return float(np.mean(values))


@torch.no_grad()
def calibrate_gain(model, corpus, device):
    """Same TRAIN-only median global recording-gain calibration as speech T2."""
    blocks=corpus.blocks['train'];ratios=[]
    for i in np.unique(np.round(np.linspace(0,len(blocks)-1,min(64,len(blocks)))).astype(int)):
        d=corpus.batch([blocks[i]],device);reference=d[3].square().mean().sqrt()
        if reference<1e-3 or (d[2]>0).float().mean()<.5:
            continue
        rms=model(*d[:3],seed=1234).square().mean().sqrt()
        if rms>1e-8:
            ratios.append(float(reference/rms))
    if not ratios:
        raise ValueError('TRAIN requires voiced blocks for gain calibration')
    adjustment=float(np.median(ratios))
    model.microphone_log_gain.add_(math.log(adjustment))
    return adjustment


def fit(corpus, out, epochs=30, patience=3, batch_size=16, lr=.0005, device='cpu', init=None):
    torch.manual_seed(SEED);rng=random.Random(SEED)
    out=Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Use an empty output directory')
    config=default_config(corpus.vocab);state=None
    if init:
        ck=torch.load(init,map_location='cpu',weights_only=True)
        config=ck['config'];state=ck['model']
        if config['vocab']!=corpus.vocab:
            raise ValueError('Checkpoint vocabulary differs')
    model=make_model(config,state,device)
    adjustment=calibrate_gain(model,corpus,device) if init is None else None
    optimizer=torch.optim.Adam(model.parameters(),lr=lr);criterion=SpectralLoss().to(device)
    best=evaluate(model,corpus,batch_size=batch_size,device=device);stale=0;history=[]
    save_checkpoint(out/'best.pt',model,config,epoch=0,validation_loss=best)
    save_checkpoint(out/'initial.pt',model,config,epoch=0,initialization='same-voice checkpoint' if init else 'independent defaults',gain_adjustment=adjustment)
    for epoch in range(1,epochs+1):
        order=list(corpus.blocks['train']);rng.shuffle(order);losses=[]
        for cursor in range(0,len(order),batch_size):
            losses.append(train_step(model,optimizer,criterion,corpus.batch(order[cursor:cursor+batch_size],device),epoch*10000+cursor))
        val=evaluate(model,corpus,batch_size=batch_size,device=device)
        improved=val<best
        if improved:
            best=val;stale=0
            save_checkpoint(out/'best.pt',model,config,epoch=epoch,validation_loss=val)
        else:
            stale+=1
        row=dict(epoch=epoch,train_loss=float(np.mean(losses)),validation_loss=val,best=best)
        history.append(row);print(json.dumps(row),flush=True)
        save_checkpoint(out/'last.pt',model,config,epoch=epoch,optimizer=optimizer.state_dict(),validation_loss=val)
        (out/'history.json').write_text(json.dumps(history,indent=2)+'\n')
        if stale>=patience:
            break
    return history
