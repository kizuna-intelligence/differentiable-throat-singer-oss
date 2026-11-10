"""main_round2 exact graph adjoint, with block replay on CPU or Triton GPU.

The CPU implementation emits the same algebra and sparse transpose as the GPU.
Only block boundaries persist; short blocks are replayed in the backward pass.
"""
import hashlib
import importlib.util
import os
import tempfile
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .kernel_graph import acoustic_graph
from .coefficients import fixed_parameters, initial_state
from .acoustics import bandlimit_noise

_CACHE = {}


def _compile(net, width):
    graph, states, coeff, signals, headers, outputs, out = acoustic_graph(net, width)
    seeds = [graph.input(f'g{i}') for i in range(len(states))]
    gy = graph.input('gy')
    grads = graph.reverse(outputs + [out], seeds + [gy], states + coeff + signals)
    source = ['import numpy as np', 'from numba import njit']
    for key, ix in graph.maps.items():
        source.append(f'idx_{key}=np.array({ix!r},dtype=np.int64)')

    def emit(name, needed, backward=False):
        reachable = set()
        def visit(i):
            if i in reachable:
                return
            reachable.add(i)
            for j in graph.nodes[i][1]:
                visit(j)
        for v in needed:
            visit(v.index)
        source.extend(['@njit(cache=False)', f'def {name}(S,C,H,noise,injection,glot' + (',GS,gy):' if backward else '):')])
        names = {}
        for i, (op, ids, meta) in enumerate(graph.nodes):
            if i not in reachable:
                continue
            if op == 'input':
                if meta in ('noise', 'injection', 'glot', 'gy'):
                    names[i] = meta
                else:
                    names[i] = {'s':'S', 'c':'C', 'h':'H', 'g':'GS'}[meta[0]] + '[' + meta[1:] + ']'
                continue
            if op == 'constant':
                names[i] = repr(meta)
                continue
            x = [names[j] for j in ids]
            names[i] = v = f'v{i}'
            if op in ('add','sub','mul','div','ge'):
                expr = f'({x[0]} ' + dict(add='+',sub='-',mul='*',div='/',ge='>=')[op] + f' {x[1]})'
            elif op in ('exp','sqrt','abs'):
                expr = f'np.{op}({x[0]})'
            elif op in ('min','max'):
                expr = f'np.{"minimum" if op=="min" else "maximum"}({x[0]},{x[1]})'
            elif op == 'where':
                expr = f'np.where({x[0]} != 0,{x[1]},{x[2]})'
            elif op == 'sum':
                expr = f'np.sum({x[0]})'
            elif op == 'broadcast':
                expr = f'({x[0]}+np.zeros({width}))'
            elif op == 'gather':
                expr = f'{x[0]}[idx_{meta}]'
            elif op == 'scatter':
                expr = f'np.bincount(idx_{meta},weights={x[0]},minlength={width})'
            else:
                raise ValueError(op)
            source.append(f'    {v}={expr}')
        values = [names[v.index] for v in needed]
        if backward:
            ns, nc = len(states), len(coeff)
            vectors = [f'({v}+np.zeros({width}))' for v in values]
            source.append('    return np.stack((' + ','.join(vectors[:ns]) + ',)),np.stack((' + ','.join(vectors[ns:ns+nc]) + ',)),' + ','.join(vectors[-3:-1]) + ',np.sum(' + vectors[-1] + ')')
        else:
            source.append('    return np.stack((' + ','.join(values[:-1]) + ',)),' + values[-1])
    emit('step', outputs + [out])
    emit('vjp', grads, True)
    # Compact orchestration keeps generated expressions out of the loop compiler.
    source.extend(['''
@njit(cache=False)
def forward(C,H,X,G,S0,hop,block):
    samples=len(G)
    check=np.empty(((samples+block-1)//block,S0.shape[0],S0.shape[1]))
    state=S0.copy()
    y=np.empty(samples)
    zero=np.zeros(S0.shape[1])
    for t in range(samples):
        if t%block==0:
            check[t//block]=state
        f=min(t//hop,len(C)-1)
        nf=min(f+1,len(C)-1)
        mix=(t%hop)/hop if f<len(C)-1 else 0.
        c=(1-mix)*C[f]+mix*C[nf]
        state,y[t]=step(state,c,H,X[t],zero,G[t])
    return y,check

@njit(cache=False)
def backward(C,H,X,G,check,GY,hop,block):
    ns,width=check.shape[1:]
    gs=np.zeros((ns,width))
    gc=np.zeros_like(C)
    gx=np.zeros_like(X)
    gg=np.zeros_like(G)
    zero=np.zeros(width)
    for bi in range(len(check)-1,-1,-1):
        start=bi*block
        end=min(start+block,len(G))
        tape=np.empty((end-start,ns,width))
        state=check[bi].copy()
        for t in range(start,end):
            tape[t-start]=state
            f=min(t//hop,len(C)-1)
            nf=min(f+1,len(C)-1)
            mix=(t%hop)/hop if f<len(C)-1 else 0.
            c=(1-mix)*C[f]+mix*C[nf]
            state,_=step(state,c,H,X[t],zero,G[t])
        for t in range(end-1,start-1,-1):
            f=min(t//hop,len(C)-1)
            nf=min(f+1,len(C)-1)
            mix=(t%hop)/hop if f<len(C)-1 else 0.
            c=(1-mix)*C[f]+mix*C[nf]
            gs,dc,gx[t],_,gg[t]=vjp(tape[t-start],c,H,X[t],zero,G[t],gs,GY[t])
            gc[f]+=(1-mix)*dc
            if f!=nf:
                gc[nf]+=mix*dc
    return gc,gx,gg,gs
'''])
    content='\n'.join(source)
    digest=hashlib.sha256(content.encode()).hexdigest()[:16]
    if digest not in _CACHE:
        directory=Path(os.environ.get('CHIMERA_CACHE', str(Path(tempfile.gettempdir())/'throat-singer-kernels')))
        directory.mkdir(parents=True, exist_ok=True)
        path=directory/f'adjoint_{digest}.py'
        path.write_text(content)
        spec=importlib.util.spec_from_file_location('adjoint_'+digest,path)
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _CACHE[digest]=module
    return _CACHE[digest]


class _CPUAcoustic(torch.autograd.Function):
    @staticmethod
    def forward(ctx, c, h, x, ga, s0, net, hop):
        module=_compile(net,c.shape[-1])
        def array(v):
            return np.ascontiguousarray(v.detach().numpy(), dtype=np.float64)
        arrays=tuple(array(v) for v in (c,h,x,ga,s0))
        waves,checks=[],[]
        cc,hh,xx,gg,ss=arrays
        for b in range(len(c)):
            y,check=module.forward(cc[b],hh,xx[b],gg[b],ss[b],hop,128)
            waves.append(y);checks.append(check)
        ctx.settings=module,arrays,checks,hop,c.dtype
        return torch.from_numpy(np.stack(waves)).to(c.dtype)

    @staticmethod
    def backward(ctx, gy):
        module,(c,h,x,ga,s0),checks,hop,dtype=ctx.settings
        derivatives=[[],[],[],[]]
        for b in range(len(c)):
            values=module.backward(c[b],h,x[b],ga[b],checks[b],np.ascontiguousarray(gy[b].numpy(),dtype=np.float64),hop,128)
            for rows,value in zip(derivatives,values):
                rows.append(value)
        gc,gx,gg,gs=[torch.from_numpy(np.stack(v)).to(dtype) for v in derivatives]
        return gc,None,gx,gg,gs,None,None


def render(net, c, ga, gain, seed, hop):
    if c.is_cuda:
        from .triton_backend import render as gpu_render
        return gpu_render(net,ga.shape[-1],coefficients=c,glottal_area=ga,noise_gain=gain,seed=seed,hop=hop)
    generator=torch.Generator(device='cpu').manual_seed(seed)
    white=torch.randn(len(c),ga.shape[-1],len(net.noise_sources),generator=generator,dtype=c.dtype)
    noise=F.pad(bandlimit_noise(white,net.sample_rate),(0,c.shape[-1]-white.shape[-1])) * gain
    h=fixed_parameters(net,c.shape[-1],c)
    s0=initial_state(net,len(c),c.shape[-1],c)
    return _CPUAcoustic.apply(c,h,noise,ga,s0,net,hop)
