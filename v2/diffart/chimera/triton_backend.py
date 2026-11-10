"""Persistent acoustic solver with block replay and generated exact adjoint.

Triton supports both ROCm and CUDA. No Python loop per audio sample, and no
96 kHz state tape: only block boundaries persist; a block is replayed backward.
The source algebra is in kernel_graph.py; generated files are disposable caches.
"""
from __future__ import annotations
import hashlib
import os
import importlib.util
from pathlib import Path
import numpy as np
import torch
from .kernel_graph import acoustic_graph
from .coefficients import pack_coefficients, fixed_parameters, initial_state


def _emit(graph,needed,aliases,indent='    '):
    """Emit only reachable nodes, with common subexpressions shared."""
    reachable=set()
    def visit(i):
        if i in reachable:return
        reachable.add(i)
        for j in graph.nodes[i][1]:visit(j)
    for v in needed:visit(v.index)
    lines=[];names={}
    for i,(op,ids,meta) in enumerate(graph.nodes):
        if i not in reachable:continue
        if op=='input':names[i]=aliases.get(meta,meta);continue
        if op=='constant':names[i]=repr(meta);continue
        x=[names[j] for j in ids];v=f'v{i}';names[i]=v
        if op in ('add','sub','mul','div','ge'):
            expr=f'({x[0]} {dict(add="+",sub="-",mul="*",div="/",ge=">=")[op]} {x[1]})'
        elif op=='exp':expr=f'tl.exp({x[0]})'
        elif op=='sqrt':expr=f'tl.sqrt({x[0]})'
        elif op=='abs':expr=f'tl.abs({x[0]})'
        elif op in ('min','max'):expr=f'tl.{"minimum" if op=="min" else "maximum"}({x[0]}, {x[1]})'
        elif op=='where':expr=f'tl.where({x[0]} != 0, {x[1]}, {x[2]})'
        elif op=='sum':expr=f'tl.sum({x[0]}, 0)'
        elif op=='broadcast':expr=f'({x[0]} + tl.full((D,), 0, tl.float32))'
        elif op=='gather':expr=f'tl.gather({x[0]}, idx_{meta}, 0)'
        elif op=='scatter':
            mapping=graph.maps[meta];count=max(mapping.count(j) for j in set(mapping))
            terms=[f'tl.where(valid_{meta}_{k} != 0, tl.gather({x[0]}, inv_{meta}_{k}, 0), 0.)' for k in range(count)]
            expr=' + '.join(terms)
        else:raise ValueError(op)
        lines.append(indent+v+' = '+expr)
    return lines,[names[v.index] for v in needed]


_CACHE={}
def compile_kernels(net,width,dynamic_glottis):
    import triton
    graph,states,coeff,signals,headers,outputs,out=acoustic_graph(net,width,dynamic_glottis=dynamic_glottis)
    nst,nc,nh=len(states),len(coeff),len(headers)
    interpolate=getattr(net,'interpolate_coefficients',False)
    signal_names=[graph.nodes[v.index][2] for v in signals]
    # Seeds are inputs, so reverse expressions are ordinary algebra nodes too.
    gs=[graph.input(f'g{i}') for i in range(nst)];gy=graph.input('gy')
    grads=graph.reverse(outputs+[out],gs+[gy],states+coeff+signals)
    # Maps include gathers and their exact sparse transposes. Never use atomics
    # in the recurrence: duplicated junction contributions are summed locally.
    maps=[];mapnames=[];idxlines=[]
    for key,ix in graph.maps.items():
        ix=np.array(ix);maps.append(ix);mapnames.append('idx_'+key)
        idxlines.append(f'    idx_{key} = tl.load(Map + {len(maps)-1}*D+i).to(tl.int32)')
        count=max(np.sum(ix==j) for j in set(ix))
        for k in range(count):
            inv=np.zeros(width,dtype=int);valid=np.zeros(width,dtype=int)
            for j in range(width):
                hits=np.flatnonzero(ix==j)
                if len(hits)>k:inv[j]=hits[k];valid[j]=1
            for name,array in [(f'inv_{key}_{k}',inv),(f'valid_{key}_{k}',valid)]:
                maps.append(array);mapnames.append(name)
                idxlines.append(f'    {name} = tl.load(Map + {len(maps)-1}*D+i).to(tl.int32)')
    args=[f's{i}' for i in range(nst)]+[f'c{i}' for i in range(nc)]+[f'h{i}' for i in range(nh)]+signal_names+mapnames
    source=['import triton','import triton.language as tl','', '@triton.jit',
            'def step('+', '.join(args)+', D:tl.constexpr):']
    lines,values=_emit(graph,outputs+[out],{})
    source+=lines+['    return '+', '.join(values),'','@triton.jit',
                  'def vjp('+', '.join(args+[f'g{i}' for i in range(nst)]+['gy'])+', D:tl.constexpr):']
    lines,values=_emit(graph,grads,{})
    # Preserve vector types even for state derivatives that are identically zero.
    for j in range(nst):values[j]=f'({values[j]} + tl.full((D,),0,s0.dtype))'
    # Scalar signals are shared over lanes, so their derivatives are summed.
    for j in range(len(values)-len(signals)+2,len(values)):
        values[j]=f'tl.sum({values[j]} + tl.full((D,), 0, tl.float32), 0)'
    source+=lines+['    return '+', '.join(values),'']
    header='S0, C, H, X, Inj, Glot, Map, Check, Y, Scratch, GC, GX, GI, GG, GY, S:tl.constexpr, F:tl.constexpr, HOP:tl.constexpr, K:tl.constexpr, D:tl.constexpr, NB:tl.constexpr'
    statevars=[f's{i}' for i in range(nst)];cvars=[f'c{i}' for i in range(nc)];hvars=[f'h{i}' for i in range(nh)]
    setup=['    b = tl.program_id(0)','    i = tl.arange(0,D)']+idxlines
    setup += [f'    h{j} = tl.load(H+{j}*D+i)' for j in range(nh)]
    def loadc(indent='        '):
        lines=[indent+'frame = tl.minimum(t//HOP, F-1)']
        if interpolate:
            lines += [indent+'next_frame = tl.minimum(frame+1,F-1)',
                      indent+'mix = tl.where(frame<F-1,tl.cast(t%HOP,C.dtype.element_ty)/HOP,0.)']
            lines += [indent+f'c{j} = (1-mix)*tl.load(C+((b*F+frame)*{nc}+{j})*D+i)+mix*tl.load(C+((b*F+next_frame)*{nc}+{j})*D+i)' for j in range(nc)]
        else:lines += [indent+f'c{j} = tl.load(C+((b*F+frame)*{nc}+{j})*D+i)' for j in range(nc)]
        return lines
    def loadsignals(indent='        '):
        lines=[indent+'noise = tl.load(X+(b*S+t)*D+i)',indent+'injection = tl.load(Inj+(b*S+t)*D+i)']
        if len(signals)==4:lines += [indent+'glot = tl.load(Glot+(b*S+t)*2)',indent+'f0 = tl.load(Glot+(b*S+t)*2+1)']
        else:lines += [indent+'glot = tl.load(Glot+b*S+t)']
        return lines
    stepargs=statevars+cvars+hvars+signal_names+mapnames+['D']
    source+=['@triton.jit',f'def forward_kernel({header}):']+setup
    source += [f'    s{j} = tl.load(S0+(b*{nst}+{j})*D+i)' for j in range(nst)]
    source+=['    for t in range(S):','        if t % K == 0:']
    source += [f'            tl.store(Check+((b*NB+t//K)*{nst}+{j})*D+i, s{j})' for j in range(nst)]
    source+=loadc()+loadsignals()
    source+=['        '+', '.join(statevars+['y'])+' = step('+', '.join(stepargs)+')',
             '        tl.store(Y+b*S+t,y)']
    if nst>=29:
        source += ['        fold_area = tl.minimum(tl.maximum(c32+2.8*s24,0.),tl.maximum(c32+2.8*s25,0.))',
                   '        tl.store(Scratch+b*S+t,tl.sum(fold_area*h0,0))']
    source += ['']
    source+=['@triton.jit',f'def backward_kernel({header}):']+setup
    source += [f'    g{j} = tl.full((D,),0,C.dtype.element_ty)' for j in range(nst)]
    source+=['    for block_rev in range(NB):','        block = NB-1-block_rev',
             '        start = block*K','        end = min(start+K,S)']
    source += [f'        s{j} = tl.load(Check+((b*NB+block)*{nst}+{j})*D+i)' for j in range(nst)]
    source += ['        for local_step in range(K):', '            t = start+local_step', '            if t < S:']
    source += [f'                tl.store(Scratch+((b*K+t-start)*{nst}+{j})*D+i,s{j})' for j in range(nst)]
    source+=loadc('                ')+loadsignals('                ')
    source+=['                '+', '.join(statevars+['y'])+' = step('+', '.join(stepargs)+')',
             '        tl.debug_barrier()',
             '        for rev in range(K):','            t = start+K-1-rev', '            if t < S:']
    source += [f'                s{j} = tl.load(Scratch+((b*K+t-start)*{nst}+{j})*D+i)' for j in range(nst)]
    source+=loadc('                ')+loadsignals('                ')+['                gy = tl.load(GY+b*S+t)']
    gradnames=[f'gn{j}' for j in range(nst)]+[f'gc{j}' for j in range(nc)]+['gx','gi','gg']+(['gf'] if len(signals)==4 else [])
    source+=['                '+', '.join(gradnames)+' = vjp('+', '.join(stepargs[:-1]+[f'g{j}' for j in range(nst)]+['gy','D'])+')']
    # One program owns b: deterministic stores aggregate time to the control
    # grid. No cross-program floating point atomics are needed.
    for j in range(nc):
        factor='*(1-mix)' if interpolate else ''
        source.append(f'                tl.store(GC+((b*F+frame)*{nc}+{j})*D+i,tl.load(GC+((b*F+frame)*{nc}+{j})*D+i)+gc{j}{factor})')
    if interpolate:
        source += ['                if frame != next_frame:']
        source += [f'                    tl.store(GC+((b*F+next_frame)*{nc}+{j})*D+i,tl.load(GC+((b*F+next_frame)*{nc}+{j})*D+i)+gc{j}*mix)' for j in range(nc)]
    source += ['                tl.store(GX+(b*S+t)*D+i,gx)','                tl.store(GI+(b*S+t)*D+i,gi)',
               ]
    if len(signals)==4:
        source += ['                tl.store(GG+(b*S+t)*2,gg)','                tl.store(GG+(b*S+t)*2+1,gf)']
    else:source += ['                tl.store(GG+b*S+t,gg)']
    source += [f'                g{j} = gn{j}' for j in range(nst)]
    source+=['        tl.debug_barrier()']
    source += [f'    tl.store(Y+(b*{nst}+{j})*D+i,g{j})' for j in range(nst)]
    content='\n'.join(source)+'\n';digest=hashlib.sha256(content.encode()).hexdigest()[:16]
    if digest not in _CACHE:
        target=Path(os.environ.get('CHIMERA_CACHE', str(Path(__import__('tempfile').gettempdir())/'throat-singer-kernels')))
        target.mkdir(parents=True,exist_ok=True);path=target/f'kernel_{digest}.py';path.write_text(content)
        spec=importlib.util.spec_from_file_location(f'chimera_kernel_{digest}',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        _CACHE[digest]=(module,np.stack(maps),graph)
    return _CACHE[digest]


class _Acoustic(torch.autograd.Function):
    @staticmethod
    def forward(ctx,c,h,x,inj,glot,s0,net,hop,block):
        b,f,_,d=c.shape;s=x.shape[1];nb=(s+block-1)//block
        module,maps,_=compile_kernels(net,d,hasattr(net,'glottis_edge'))
        maps=torch.tensor(maps,device=c.device,dtype=torch.int32)
        check=c.new_empty(b,nb,s0.shape[1],d);y=c.new_empty(b,s)
        dummy=c.new_empty(1)
        diag=c.new_empty(b,s) if s0.shape[1]>=29 else c.new_empty(0)
        args=[s0,c,h,x,inj,glot,maps,check,y,diag,dummy,dummy,dummy,dummy,dummy]
        module.forward_kernel[(b,)](*args,s,f,hop,block,d,nb,num_warps=int(os.environ.get('CHIMERA_NUM_WARPS','4')),enable_fp_fusion=False)
        ctx.save_for_backward(c,h,x,inj,glot,s0,maps,check)
        ctx.settings=(net,hop,block,module)
        ctx.mark_non_differentiable(diag)
        return y,diag
    @staticmethod
    def backward(ctx,gy,_diag):
        c,h,x,inj,glot,s0,maps,check=ctx.saved_tensors;net,hop,block,module=ctx.settings
        b,f,_,d=c.shape;s=x.shape[1];nb=(s+block-1)//block
        scratch=c.new_empty(b,block,s0.shape[1],d);gc=torch.zeros_like(c);gx=torch.empty_like(x)
        gi=torch.empty_like(inj);gg=torch.empty_like(glot);dummy=c.new_empty(1);gs0=torch.empty_like(s0)
        module.backward_kernel[(b,)](s0,c,h,x,inj,glot,maps,check,gs0,scratch,gc,gx,gi,gg,gy.contiguous(),
                                    s,f,hop,block,d,nb,num_warps=int(os.environ.get('CHIMERA_NUM_WARPS','4')),enable_fp_fusion=False)
        return gc,None,gx,gi,gg,gs0,None,None,None


def render(net, samples, *, coefficients=None, glottal_area=None, areas=None, dx=None,
           port_area=None, wall_scale=1., noise_gain=1., seed=0, hop=960, block=128,
           injection=None, noise=None,return_fold_area=False):
    if not net.areas.is_cuda and __import__('os').environ.get('TRITON_INTERPRET')!='1':raise ValueError('Triton backend requires a CUDA/ROCm GPU')
    c=pack_coefficients(net,areas,dx,port_area,wall_scale) if coefficients is None else coefficients
    b,_,_,d=c.shape
    if noise is None:
        from .acoustics import bandlimit_noise
        portable=os.environ.get('CHIMERA_PORTABLE_NOISE')=='1'
        random_device='cpu' if portable else c.device
        generator=torch.Generator(device=random_device).manual_seed(seed)
        count=1 if getattr(net,'repeat_noise_across_batch',False) else b
        noise=torch.randn(count,samples,len(net.noise_sources),device=random_device,dtype=c.dtype,generator=generator).to(c.device)
        if noise.shape[-1]:
            white=noise
            noise=bandlimit_noise(white,net.sample_rate)
    if isinstance(noise_gain,torch.Tensor) and noise_gain.ndim==3 and noise_gain.shape[-1]>1:
        noise_gain=torch.nn.functional.pad(noise_gain,(0,d-noise_gain.shape[-1]),value=1.)
    noise=torch.nn.functional.pad(noise,(0,d-noise.shape[-1])).expand(b,-1,-1).contiguous()*noise_gain
    if injection is None:injection=c.new_zeros(b,samples,d)
    else:injection=torch.nn.functional.pad(injection,(0,d-injection.shape[-1])).contiguous()
    if glottal_area is None:glottal_area=c.new_full((b,samples),.05)
    wave,diag = _Acoustic.apply(c,fixed_parameters(net,d,c),noise,injection,glottal_area.contiguous(),
                           initial_state(net,b,d,c),net,hop,block)

    return (wave,diag) if return_fold_area else wave
