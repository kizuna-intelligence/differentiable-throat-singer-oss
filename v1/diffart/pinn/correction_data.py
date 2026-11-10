"""Conforming sharp-step axisymmetric FEM, matched to legacy ABCD boundaries.

Pressure-release outlet, rigid wall/axis, prescribed inlet derivative. Complex
propagation gamma=wall+i*k matches physics.tube_matrices (Np/m, C=350 m/s).
No nose: R describes the oral tube; its application to a nasal shunt is approximate.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import time

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

C = 350.
SEED = 20261109


def mesh(area, length, nx=3, nr=6):
    radii = np.sqrt(np.asarray(area)*1e-4/np.pi)
    levels = [np.linspace(0, r, nr+1) for r in radii]
    nodes, lookup, triangles = [], {}, []

    def column(x, values):
        result = []
        for r in values:
            key = (round(float(x), 13), round(float(r), 13))
            if key not in lookup:
                lookup[key] = len(nodes)
                nodes.append((x, r))
            result.append(lookup[key])
        return result

    for j, radius in enumerate(radii):
        columns = []
        for i in range(nx+1):
            rr = levels[j]
            other = j-1 if i == 0 else j+1 if i == nx else j
            if 0 <= other < len(radii):
                rr = np.unique(np.round(np.r_[rr, levels[other][levels[other] < radius]], 13))
            columns.append(column((j+i/nx)*length/len(area), rr))
        for left, right in zip(columns[:-1], columns[1:]):
            a = b = 0
            while a < len(left)-1 or b < len(right)-1:
                if b == len(right)-1 or (a < len(left)-1 and nodes[left[a+1]][1] <= nodes[right[b+1]][1]):
                    triangles.append((left[a], right[b], left[a+1]))
                    a += 1
                else:
                    triangles.append((left[a], right[b], right[b+1]))
                    b += 1
    return np.asarray(nodes), np.asarray(triangles)


def assemble(area, length, nx=3, nr=6):
    nodes, triangles = mesh(area, length, nx, nr)
    tri = nodes[triangles]
    delta = tri[:, 1:]-tri[:, :1]
    det = delta[:, 0, 0]*delta[:, 1, 1]-delta[:, 0, 1]*delta[:, 1, 0]
    volume = abs(det)/2
    grad = np.stack((tri[:, [1,2,0],1]-tri[:, [2,0,1],1],
                     tri[:, [2,0,1],0]-tri[:, [1,2,0],0]), -1)/det[:,None,None]
    stiffness = volume[:,None,None]*tri[:,:,1].mean(-1)[:,None,None]*np.einsum('tia,tja->tij',grad,grad)
    bary = np.array([[1/3]*3,[.6,.2,.2],[.2,.6,.2],[.2,.2,.6]])
    mass = volume[:,None,None]*np.einsum('tq,q,qi,qj->tij',tri[:,:,1]@bary.T,np.array([-27,25,25,25])/48,bary,bary)
    rows = np.broadcast_to(triangles[:,:,None],mass.shape).ravel()
    cols = np.broadcast_to(triangles[:,None,:],mass.shape).ravel()
    shape = (len(nodes),len(nodes))
    K = sparse.coo_matrix((stiffness.ravel(),(rows,cols)),shape=shape).tocsc()
    M = sparse.coo_matrix((mass.ravel(),(rows,cols)),shape=shape).tocsc()
    inlet = np.where(abs(nodes[:,0]) < 1e-12)[0]
    outlet = np.where(abs(nodes[:,0]-length) < 1e-12)[0]
    inlet = inlet[np.argsort(nodes[inlet,1])]
    rhs = np.zeros(len(nodes),complex)
    a,b = inlet[:-1],inlet[1:]
    ra,rb = nodes[a,1],nodes[b,1]
    np.add.at(rhs,a,-(rb-ra)*(2*ra+rb)/6/length)
    np.add.at(rhs,b,-(rb-ra)*(ra+2*rb)/6/length)
    free = np.setdiff1d(np.arange(len(nodes)),outlet)
    return K,M,rhs,free,outlet


def reference(area,length,wall,freq,nx=3,nr=6):
    K,M,rhs,free,outlet = assemble(area,length,nx,nr)
    kf,mf=K[free][:,free],M[free][:,free]
    ko,mo=K[outlet][:,free].sum(axis=0),M[outlet][:,free].sum(axis=0)
    result=[]
    for f in freq:
        gamma=wall+2j*np.pi*f/C
        p=splu(kf+gamma**2*mf).solve(rhs[free])
        flux=np.asarray(ko+gamma**2*mo).ravel()@p
        result.append(-flux/rhs.sum())
    return np.asarray(result)


def plane(area,length,wall,freq):
    gamma=(wall+2j*np.pi*np.asarray(freq)/C)*length/len(area)
    co,si=np.cosh(gamma),np.sinh(gamma)
    h=np.broadcast_to(np.eye(2),(len(freq),2,2)).astype(complex).copy()
    for a in area:
        mat=np.stack((np.stack((co,si/a),-1),np.stack((si*a,co),-1)),-2)
        h=h@mat
    return 1/h[:,1,1]


def task(args):
    i,area,length,wall,freq,nx,nr,path=args
    path=Path(path)/f'{i:05d}.npz'
    if path.exists():
        d=np.load(path)
        return i,d['h3'],d['h1']
    h3=reference(area,length,wall,freq,nx,nr)
    h1=plane(area,length,wall,freq)
    np.savez(path,h3=h3,h1=h1)
    return i,h3,h1


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--shapes',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--count',type=int,default=200)
    ap.add_argument('--heldout',type=int,default=40)
    ap.add_argument('--workers',type=int,default=16)
    ap.add_argument('--nx',type=int,default=3)
    ap.add_argument('--nr',type=int,default=6)
    ap.add_argument('--wall',type=float,required=True)
    args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    cache=args.out/'cache';cache.mkdir(exist_ok=True)
    d=np.load(args.shapes)
    # Fixed held-out source frames, disjoint exact area vectors, across all sizes.
    areas,source=np.unique(d['area_cm2'],axis=0,return_index=True)
    order=np.random.default_rng(SEED).permutation(len(areas))
    held=order[:200][:args.heldout]
    train=order[200:200+args.count]
    selected=np.r_[train,held]
    area=areas[selected];source=source[selected]
    mean=d['log_area_mean_m2'];basis=d['basis'][:8]
    assert len(basis)==8
    coeff=(np.log(area*1e-4)-mean)@basis.T
    freq=np.linspace(50,8000,80);length=float(d['length_m'])
    work=[(int(i),a,length,args.wall,freq,args.nx,args.nr,cache) for i,a in zip(source,area)]
    start=time.monotonic();h3=np.empty((len(work),80),complex);h1=h3.copy()
    index={int(x):i for i,x in enumerate(source)}
    with ProcessPoolExecutor(args.workers) as executor:
        for n,(i,a,b) in enumerate(executor.map(task,work),1):
            j=index[i];h3[j]=a;h1[j]=b
            if n%20==0:print(json.dumps(dict(done=n,total=len(work),seconds=time.monotonic()-start)),flush=True)
    ratio=h3/h1
    np.savez_compressed(args.out/'dataset.npz',area_cm2=area,coeff=coeff,source_frames=source,
        freq=freq,h3=h3,h1=h1,ratio=ratio,train_count=len(train),length_m=length,wall=args.wall,
        mean=mean,basis=basis,phone_ids=d['phone_ids'][source])
    uniform=reference(np.ones(44),length,args.wall,freq,args.nx,args.nr)/plane(np.ones(44),length,args.wall,freq)
    checks=[]
    for i in range(min(10,len(area))):
        fine=reference(area[i],length,args.wall,freq,args.nx*2,args.nr*2)
        err=20/np.log(10)*np.log(abs(fine/h3[i]))
        phase=np.angle(fine/h3[i])
        checks.append(dict(shape=i,magnitude_mae_db=float(abs(err).mean()),magnitude_max_db=float(abs(err).max()),
                           phase_mae_rad=float(abs(phase).mean()),relative_l2=float(np.linalg.norm(fine-h3[i])/np.linalg.norm(fine))))
    summary=dict(train_shapes=len(train),heldout_shapes=len(held),unique_available=len(areas),frequencies=len(freq),
        seconds=time.monotonic()-start,nx=args.nx,nr=args.nr,wall_np_per_m=args.wall,length_m=length,
        ratio_abs_quantiles=np.quantile(abs(ratio),[0,.05,.5,.95,.99,1]).tolist(),
        uniform_low_log_max=float(abs(np.log(uniform[freq<1000])).max()),
        uniform_low_pass=bool(abs(np.log(uniform[freq<1000])).max()<.01),mesh_checks=checks,
        boundary='rigid wall, volume-flow inlet, pressure-release outlet; matches legacy transfer, not quick_a Z=.15',
        approximation='oral tube correction multiplied onto branched nasal transfer; length/wall fixed to base for labels',seed=SEED)
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary),flush=True)


if __name__=='__main__': main()
