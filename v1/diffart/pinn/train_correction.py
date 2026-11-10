"""Short supervised surrogate fit, with validation split by whole shape."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from .correction import FEMCorrection


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--steps',type=int,default=4000)
    ap.add_argument('--seconds',type=float,default=180)
    ap.add_argument('--threads',type=int,default=4)
    args=ap.parse_args()
    torch.set_num_threads(args.threads);torch.manual_seed(20261109)
    d=np.load(args.data);n=int(d['train_count'])
    coeff=d['coeff'];ratio=d['ratio'];freq=d['freq']
    phase=np.unwrap(np.angle(ratio),axis=1)
    # R is close to +1 at low f; align all unwrapped phase branches there.
    phase-=2*np.pi*np.round(phase[:,:1]/(2*np.pi))
    y=np.stack((np.log(abs(ratio)),phase),-1)
    # Keep the final 10% of training shapes for model selection; untouched 200
    # held-out shapes are evaluated only after selecting the best checkpoint.
    fit=max(1,int(n*.9))
    meta=dict(mean=d['mean'],basis=d['basis'],coeff_mean=coeff[:fit].mean(0),
              coeff_std=coeff[:fit].std(0).clip(.01),target_mean=y[:fit].mean((0,1)),
              target_std=y[:fit].std((0,1)).clip(.01),frequencies=freq,width=128)
    meta={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in meta.items()}
    model=FEMCorrection(meta)
    inputs=np.concatenate((np.broadcast_to(((coeff-np.array(meta['coeff_mean']))/meta['coeff_std'])[:,None,:],(len(coeff),80,8)),
                           np.broadcast_to((freq/4000-1)[None,:,None],(len(coeff),80,1))),-1)
    x=torch.tensor(inputs,dtype=torch.float32);target=torch.tensor(y,dtype=torch.float32)
    xn=x[:fit].reshape(-1,9)
    yn=((target[:fit]-model.target_mean)/model.target_std).reshape(-1,2)
    optimizer=torch.optim.Adam(model.mlp.parameters(),lr=.002)
    args.out.mkdir(parents=True,exist_ok=True)
    start=time.monotonic();best=float('inf');best_step=0
    log=(args.out/'training.jsonl').open('w')
    for step in range(args.steps+1):
        if step%100==0:
            with torch.no_grad():
                pred=model.mlp(x[fit:n])*model.target_std+model.target_mean
                val=float(((pred-target[fit:n])/model.target_std).square().mean())
            row=dict(step=step,seconds=time.monotonic()-start,val_loss=val,
                     train_loss=float(loss.detach()) if step else None)
            log.write(json.dumps(row)+'\n');log.flush();print(json.dumps(row),flush=True)
            if val<best:
                best=val;best_step=step
                torch.save(dict(model=model.state_dict(),metadata=meta,step=step),args.out/'ratio.pt')
            if time.monotonic()-start>args.seconds:break
        ids=torch.randint(len(xn),(2048,))
        loss=(model.mlp(xn[ids])-yn[ids]).square().mean()
        optimizer.zero_grad();loss.backward();optimizer.step()
    log.close()
    model=FEMCorrection.load(args.out/'ratio.pt')
    with torch.no_grad():pred=(model.mlp(x[n:])*model.target_std+model.target_mean).numpy()
    logerr=pred[...,0]-y[n:,:,0]
    phaseerr=np.angle(np.exp(1j*(pred[...,1]-y[n:,:,1])))
    all_pred=[]
    with torch.no_grad():
        for a in torch.tensor(d['area_cm2'],dtype=torch.float32).split(50):all_pred.append(model.grid(a).numpy())
    all_pred=np.concatenate(all_pred)
    summary=dict(method='FEM-supervised MLP only; no Helmholtz residual/field loss',
        train_shapes=n,fit_shapes=fit,selection_shapes=n-fit,heldout_shapes=len(coeff)-n,
        steps=step,best_step=best_step,seconds=time.monotonic()-start,
        heldout_magnitude_mae_db=float(np.mean(abs(logerr))*20/np.log(10)),
        heldout_magnitude_rmse_db=float(np.sqrt(np.mean(logerr**2))*20/np.log(10)),
        heldout_phase_mae_rad=float(abs(phaseerr).mean()),heldout_phase_rmse_rad=float(np.sqrt(np.mean(phaseerr**2))),
        heldout_complex_relative_l2=float(np.linalg.norm(np.exp(pred[...,0]+1j*pred[...,1])-ratio[n:])/np.linalg.norm(ratio[n:])),
        predicted_abs_max=float(np.exp(all_pred[...,0]).max()),
        measured_abs_max=float(abs(ratio).max()),passive_contractivity=bool(all_pred[...,0].max()<=0),
        passivity_note='|R|>1 violates a standalone contractive correction; a transfer-ratio amplitude is not a power-normalized scattering passivity test.')
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
