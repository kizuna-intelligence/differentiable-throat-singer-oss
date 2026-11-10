"""Reference validation of an experimental PCA-conditioned MLP checkpoint."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .field import AreaGeometry,ParametricField
from .smooth_reference import solve
from .uniform import piston_impedance


def assess(checkpoint,out):
    torch.set_num_threads(2)
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    args=saved['args']
    with np.load(args['shapes']) as d:
        geometry=AreaGeometry(d['log_area_mean_m2'],d['basis'],float(d['length_m']))
        samples=d['coefficients'].copy()
    geometry.load_state_dict(saved['geometry'])
    model=ParametricField(samples.shape[1],width=args['width']).double()
    model.load_state_dict(saved['model']);model.eval()
    split=int(len(samples)*(1-args['holdout_fraction']))
    chosen=np.linspace(split,len(samples)-1,3,dtype=int)
    length=geometry.length
    rows=[]
    for index in chosen:
        coefficient=torch.as_tensor(samples[index:index+1],dtype=torch.float64)
        def radius(x):
            with torch.no_grad():
                return geometry.radius(torch.as_tensor(x[:,None],dtype=torch.float64),coefficient.expand(len(x),-1)).numpy().ravel()
        for f in (250.,1000.,3000.,5000.):
            z=complex(piston_impedance(f,float(radius(np.array([length]))[0])))
            coarse=solve(radius,length,f,z,nx=240,nr=16)
            ref=solve(radius,length,f,z,nx=480,nr=24)
            mesh_history=[dict(nx=240,nr=16,transfer_real=coarse.transfer.real,transfer_imag=coarse.transfer.imag)]
            for next_nx,next_nr in ((960,32),(1920,48),(3840,64)):
                change=float(abs(coarse.transfer/ref.transfer-1))
                mesh_history.append(dict(nodes=len(ref.nodes),relative_change=change))
                if change < .01:
                    break
                coarse=ref
                ref=solve(radius,length,f,z,nx=next_nx,nr=next_nr)
            x=torch.tensor(ref.nodes[:,0:1],dtype=torch.float64,requires_grad=True)
            rho=torch.tensor(ref.nodes[:,1:2],dtype=torch.float64,requires_grad=True)
            freq=torch.full_like(x,f)
            coeff=coefficient.expand(len(x),-1)
            with torch.no_grad():
                values=model(x,rho,freq,coeff,geometry).numpy()
            pressure=values[:,0]+1j*values[:,1]
            error=float(np.sqrt(np.sum(ref.weights*abs(pressure-ref.pressure)**2)/np.sum(ref.weights*abs(ref.pressure)**2)))
            # Integrate predicted outlet velocity from both p_x and Robin pressure.
            q,w=np.polynomial.legendre.leggauss(80)
            rend=radius(np.array([length]))[0];rin=radius(np.array([0.]))[0]
            r=rend*np.sqrt((q+1)/2)
            tx=torch.full((len(r),1),length,dtype=torch.float64,requires_grad=True)
            tr=torch.tensor(r[:,None],dtype=torch.float64,requires_grad=True)
            p=model(tx,tr,torch.full_like(tx,f),coefficient.expand(len(r),-1),geometry)
            derivatives=[torch.autograd.grad(p[:,i].sum(),tx,retain_graph=True)[0].detach().numpy().ravel() for i in (0,1)]
            px=derivatives[0]+1j*derivatives[1]
            h_derivative=np.sum(w*px)*rend**2/4*length/(rin**2/2)
            pv=p.detach().numpy();pc=pv[:,0]+1j*pv[:,1]
            h_robin=np.sum(w*(-1j*2*np.pi*f/350/z)*pc)*rend**2/4*length/(rin**2/2)
            row=dict(shape_index=int(index),frequency_hz=f,field_relative_l2=error,
                transfer_derivative_relative_error=float(abs(h_derivative/ref.transfer-1)),
                transfer_robin_relative_error=float(abs(h_robin/ref.transfer-1)),
                fem_grid_transfer_change=float(abs(coarse.transfer/ref.transfer-1)),
                reference_transfer=dict(real=ref.transfer.real,imag=ref.transfer.imag),
                predicted_derivative_transfer=dict(real=float(h_derivative.real),imag=float(h_derivative.imag)),
                fem_mesh_history=mesh_history,reference_converged=bool(abs(coarse.transfer/ref.transfer-1)<.01),
                fem_nodes=len(ref.nodes),reference_algebraic_residual=ref.algebraic_relative_residual)
            rows.append(row)
            print(json.dumps(row),flush=True)
    result=dict(checkpoint=str(checkpoint),shape_indices=chosen.tolist(),
        reference='Independent P1 axisymmetric triangular FEM; same smooth geometry and piston Robin',
        shape_split='Final contiguous 20% withheld from optimizer; shared PCA is not held out',
        pass_threshold=dict(field_relative_l2=.05,transfer_relative_error=.05),
        usable_correction_teacher=all(r['field_relative_l2']<.05 and r['transfer_derivative_relative_error']<.05 and
            r['fem_grid_transfer_change']<.01 for r in rows),rows=rows)
    out.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('checkpoint',type=Path)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();assess(a.checkpoint,a.out)
