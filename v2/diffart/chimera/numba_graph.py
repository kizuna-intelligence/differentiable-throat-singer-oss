"""CPU-only forward execution of the GPU algebra for long diagnostic runs.

This is not a training backend. The independent reference remains cpu.py.
"""
import hashlib,importlib.util
from pathlib import Path
import numpy as np
from .kernel_graph import acoustic_graph
from .coefficients import pack_coefficients,fixed_parameters,initial_state

_CACHE={}
def compile_forward(net,width):
    graph,states,coeff,signals,headers,outputs,out=acoustic_graph(net,width)
    needed=set()
    def visit(i):
        if i in needed:return
        needed.add(i)
        for j in graph.nodes[i][1]:visit(j)
    for v in outputs+[out]:visit(v.index)
    nst,nc=len(states),len(coeff);servo=len(signals)==4
    source=['import numpy as np','from numba import njit']
    for key,ix in graph.maps.items():source += [f'idx_{key} = np.array({ix!r},dtype=np.int64)']
    args=[f's{i}' for i in range(nst)]+[f'c{i}' for i in range(nc)]+[f'h{i}' for i in range(len(headers))]+['noise','injection','glot']+(['f0'] if servo else [])
    source += ['@njit(cache=False)','def step('+','.join(args)+'):']
    names={}
    for i,(op,ids,meta) in enumerate(graph.nodes):
        if i not in needed:continue
        if op=='input':names[i]=meta;continue
        if op=='constant':names[i]=repr(meta);continue
        x=[names[j] for j in ids];v=f'v{i}';names[i]=v
        if op in ('add','sub','mul','div','ge'):
            expr=f'({x[0]} {dict(add="+",sub="-",mul="*",div="/",ge=">=")[op]} {x[1]})'
        elif op=='exp':expr=f'np.exp({x[0]})'
        elif op=='gate':expr=f'({x[0]} >= {meta[0]!r}).astype(np.float64)'
        elif op=='sqrt':expr=f'np.sqrt({x[0]})'
        elif op=='abs':expr=f'np.abs({x[0]})'
        elif op in ('min','max'):expr=f'np.{"minimum" if op=="min" else "maximum"}({x[0]},{x[1]})'
        elif op=='where':expr=f'np.where({x[0]} != 0,{x[1]},{x[2]})'
        elif op=='sum':expr=f'np.sum({x[0]})'
        elif op=='gather':expr=f'{x[0]}[idx_{meta}]'
        elif op=='scatter':expr=f'np.bincount(idx_{meta},weights={x[0]},minlength={width})'
        else:raise ValueError(op)
        source += ['    '+v+' = '+expr]
    source += ['    return '+','.join(names[v.index] for v in outputs+[out]),
               '@njit(cache=False)','def forward(C,H,G,S0,hop):',
               '    samples=len(G)','    result=np.empty(samples)','    area=np.empty(samples)',
               f'    z=np.zeros({width})']
    source += [f'    s{j}=S0[{j}].copy()' for j in range(nst)]
    source += [f'    h{j}=H[{j}]' for j in range(len(headers))]
    source += ['    for t in range(samples):','        frame=min(t//hop,len(C)-1)']
    if getattr(net,'interpolate_coefficients',False):
        source += ['        next_frame=min(frame+1,len(C)-1)','        mix=(t%hop)/hop if frame<len(C)-1 else 0.']
        source += [f'        c{j}=(1-mix)*C[frame,{j}]+mix*C[next_frame,{j}]' for j in range(nc)]
    else:source += [f'        c{j}=C[frame,{j}]' for j in range(nc)]
    call=[f's{i}' for i in range(nst)]+[f'c{i}' for i in range(nc)]+[f'h{i}' for i in range(len(headers))]+['z','z','G[t,0]']+(['G[t,1]'] if servo else [])
    source += ['        '+','.join([f's{i}' for i in range(nst)]+['y'])+' = step('+','.join(call)+')',
               '        result[t]=y']
    if nst>=29:source += ['        area[t]=np.sum(np.minimum(np.maximum(c32+2.8*s24,0.),np.maximum(c32+2.8*s25,0.))*h0)']
    else:source += ['        area[t]=G[t,0]']
    source += ['    return result,area']
    content='\n'.join(source)+'\n';digest=hashlib.sha256(content.encode()).hexdigest()[:16]
    if digest not in _CACHE:
        directory=Path(__import__('os').environ.get('CHIMERA_CACHE',str(Path(__import__('tempfile').gettempdir())/'throat-singer-kernels')));directory.mkdir(parents=True,exist_ok=True)
        path=directory/f'numba_{digest}.py';path.write_text(content)
        spec=importlib.util.spec_from_file_location('chimera_numba_'+digest,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        _CACHE[digest]=module
    return _CACHE[digest]


def simulate(net,glottal,coefficients=None,hop=960):
    c=pack_coefficients(net) if coefficients is None else coefficients
    d=c.shape[-1];s0=initial_state(net,1,d,c);h=fixed_parameters(net,d,c)
    glottal=np.asarray(glottal)
    if glottal.ndim==1:glottal=glottal[:,None]
    module=compile_forward(net,d)
    def array(x):return np.ascontiguousarray(x.detach().cpu().numpy())
    return module.forward(array(c[0]),array(h),glottal,array(s0[0]),hop)
