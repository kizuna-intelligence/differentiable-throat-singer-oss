"""Algebra graph for parametric CHIMERA acoustics and exact differentiation.
Physics: CHIMERA airway/network.rs, MIT, (c) 2026 Jameson Hodge.
"""
from __future__ import annotations
import math
import numpy as np
from .constants import RHO, RHO_C2, C_SOUND, NU, RE_CRIT

class Value:

    def __init__(self, graph, index):
        self.graph, self.index = (graph, index)

    def __add__(self, x):
        return self.graph.op('add', self, x)
    __radd__ = __add__

    def __sub__(self, x):
        return self.graph.op('sub', self, x)

    def __rsub__(self, x):
        return self.graph.op('sub', x, self)

    def __mul__(self, x):
        return self.graph.op('mul', self, x)
    __rmul__ = __mul__

    def __truediv__(self, x):
        return self.graph.op('div', self, x)

    def __rtruediv__(self, x):
        return self.graph.op('div', x, self)

    def __neg__(self):
        return self * -1.0

    def exp(self):
        return self.graph.op('exp', self)

    def sqrt(self):
        return self.graph.op('sqrt', self)

    def abs(self):
        return self.graph.op('abs', self)

    def maximum(self, x):
        return self.graph.op('max', self, x)

    def minimum(self, x):
        return self.graph.op('min', self, x)

    def ge(self, x):
        return self.graph.op('ge', self, x)

    def where(self, yes, no):
        return self.graph.op('where', self, yes, no)

    def sum(self):
        return self.graph.op('sum', self)

    def gather(self, key):
        return self.graph.op('gather', self, meta=key)

    def scatter(self, key):
        return self.graph.op('scatter', self, meta=key)

class Graph:

    def __init__(self):
        self.nodes = []
        self.cache = {}
        self.maps = {}

    def op(self, op, *args, meta=None):
        args = tuple((x if isinstance(x, Value) else self.constant(x) for x in args))
        ids = tuple((x.index for x in args))
        key = (op, ids, meta)
        if key not in self.cache:
            self.cache[key] = len(self.nodes)
            self.nodes.append(key)
        return Value(self, self.cache[key])

    def constant(self, x):
        key = ('constant', (), float(x))
        if key not in self.cache:
            self.cache[key] = len(self.nodes)
            self.nodes.append(key)
        return Value(self, self.cache[key])

    def input(self, name):
        return self.op('input', meta=name)

    def reverse(self, outputs, seeds, inputs):
        grads={}
        def add(i,g): grads[i]=grads[i]+g if i in grads else g
        for out,seed in zip(outputs,seeds):add(out.index,seed)
        limit=len(self.nodes)
        for i in range(limit-1,-1,-1):
            if i not in grads:continue
            op,ids,meta=self.nodes[i];g=grads[i]
            xs=[Value(self,j) for j in ids]
            if op in ('input','constant','ge'):continue
            if op=='add':ds=[g,g]
            elif op=='sub':ds=[g,-g]
            elif op=='mul':ds=[g*xs[1],g*xs[0]]
            elif op=='div':ds=[g/xs[1],-g*xs[0]/(xs[1]*xs[1])]
            elif op=='exp':ds=[g*Value(self,i)]
            elif op=='sqrt':ds=[g/(2*Value(self,i))]
            elif op=='abs':ds=[g*xs[0].ge(0).where(1.,-1.)]
            elif op=='where':ds=[None,xs[0].where(g,0.),xs[0].where(0.,g)]
            elif op in ('min','max'):
                # Equality splits derivatives as torch.minimum/maximum do.
                eq=xs[0].ge(xs[1])*xs[1].ge(xs[0])
                w=xs[0].ge(xs[1]).where(1.,0.) if op=='max' else xs[1].ge(xs[0]).where(1.,0.)
                w=eq.where(.5,w);ds=[g*w,g*(1-w)]
            elif op=='sum':ds=[self.op('broadcast',g)]
            elif op=='broadcast':ds=[g.sum()]
            elif op=='gather':ds=[g.scatter(meta)]
            elif op=='scatter':ds=[g.gather(meta)]
            else:raise ValueError(op)
            for j,d in zip(ids,ds):
                if d is not None and self.nodes[j][0]!='constant':add(j,d)
        return [grads.get(v.index,self.constant(0.)) for v in inputs]


def acoustic_graph(net, width, *, dynamic_glottis=True):
    g = Graph()
    n, e = (len(net.areas), len(net.edge_a))

    def mapping(key, values):
        g.maps[key] = np.r_[np.asarray(values, dtype=int), np.arange(len(values), width)].tolist()
        return key
    ea = mapping('ea', net.edge_a.cpu().numpy())
    eb = mapping('eb', net.edge_b.cpu().numpy())
    nsources = len(net.noise_sources)
    src = mapping('src', [v[0] for v in net.noise_sources])
    fw = mapping('fw', [v[1] for v in net.noise_sources])
    rv = mapping('rv', [v[2] for v in net.noise_sources])
    state = [g.input(f's{i}') for i in range(25)]
    names = ['l', 'r', 'kf', 'kr', 'opened', 'compliance', 'wall_area', 'eff', 'mass', 'damping', 'stiffness', 'radmask', 'fixedmask', 'fixedp', 'wallrad', 'dx'] + [f'bg{i}' for i in range(8)] + [f'ng{i}' for i in range(8)]
    coefficients = [g.input(f'c{i}') for i in range(len(names))]
    c = dict(zip(names, coefficients))
    p, momentum, old_l, wv, wy, ul, old_lr, old_rad, *relax = state[:24]
    bx, nx = (relax[:8], relax[8:])
    l, r, kf, kr, opened, comp, wa, eff = [c[k] for k in names[:8]]
    noise = g.input('noise')
    injection = g.input('injection')
    glot = g.input('glot')
    signals = [noise, injection, glot]
    fixed = [g.input(f'h{i}') for i in range(10)]
    gmask, gl, gr, gent, gexit, inv_a, inv_b, source_gain, source_mask, edge_mask = fixed
    if getattr(net, 'interpolate_coefficients', False):
        opened = eff.ge(net.closure_area) * edge_mask
    fold_outputs = []
    if dynamic_glottis:
        ap = glot.maximum(1e-06)
        default = float(net.port_area[net.glottis_edge])
        l = gmask.where(l - gl / default + gl / ap, l)
        r = gmask.where(r - gr / default ** 3 + gr / (ap * ap * ap), r)
        node_inverse = c['dx'] / (comp * RHO_C2)
        dpb = (1 / ap - node_inverse.gather(eb)).maximum(0.0)
        dpa = (1 / ap - node_inverse.gather(ea)).maximum(0.0)
        kf = gmask.where(gent / (ap * ap) + gexit * dpb * dpb, kf)
        kr = gmask.where(gent / (ap * ap) + gexit * dpa * dpa, kr)
        gate = ap.ge(net.closure_area)
        opened = gmask.where(gate, opened)
        eff = gmask.where(ap, eff)
    fs = net.sample_rate
    dt = 1 / fs
    momentum = momentum * (l / old_l.maximum(1e-20)).minimum(1.0)
    rhs = p.gather(ea) - p.gather(eb) + momentum * fs
    if nsources:
        u = (momentum / l).gather(src)
        area = eff.gather(src).maximum(1e-06)
        re = u.abs() * 2 * (area / math.pi).sqrt() / (NU * area)
        excess = (re * re - RE_CRIT ** 2).maximum(0.0)
        amp = source_gain * excess / (1 + excess / (12000.0 ** 2 - RE_CRIT ** 2)) * math.sqrt(96000 / fs)
        signal = amp * noise * source_mask
        rhs = rhs + u.ge(0).where(signal, 0.0).scatter(fw) - u.ge(0).where(0.0, signal).scatter(rv)
    tg, ts = (0.0, 0.0)
    lbs = []
    las = []
    if net.frequency_dependent_loss and net.viscous:
        for j, pole in enumerate(net.loss_poles.tolist()):
            lb = pole * dt / (2 + pole * dt)
            la = 1 - 2 * lb
            lbs.append(lb)
            las.append(la)
            bg, ng = (c[f'bg{j}'], c[f'ng{j}'])
            rhs = rhs + bg * (la * bx[j] + lb * momentum / l)
            r = r + bg * (1 - lb)
            tg = tg + ng * (1 - lb) * 0.5
            ts = ts + ng * (1 - lb) * nx[j]
    k = rhs.ge(0.0).where(kf, kr)
    linear = l * fs + r
    u = 2 * rhs / (linear + (linear * linear + 4 * k * rhs.abs()).sqrt()) * opened
    newbx = [las[j] * bx[j] + lbs[j] * (u + momentum / l) for j in range(8)] if lbs else bx
    flux = u.scatter(eb) - u.scatter(ea) + injection
    if net.volume_source:
        flux = flux - (comp * RHO_C2 - state[-1]) * fs
    haswall = c['mass'].ge(1e-30)
    mass = haswall.where(c['mass'], 1.0)
    wvn = ((mass * fs - 0.5 * c['damping']) * wv + p - c['stiffness'] * wy) / (mass * fs + 0.5 * c['damping'])
    wvn = haswall.where(wvn, 0.0)
    wyn = wy + dt * wvn
    wallflow = wa * wvn
    flux = flux - wallflow
    pn = (p * (comp * fs - tg) + flux + ts) / (comp * fs + tg)
    ar = (comp * RHO_C2 / c['dx'].maximum(1e-12)).maximum(0.0001)
    lr = 8 * RHO / (3 * math.pi * (math.pi * ar).sqrt())
    rr = 128 * RHO * C_SOUND / (9 * math.pi ** 2 * ar)
    prev = ul * (old_lr / lr).minimum(1.0)
    h = dt / (2 * lr)
    rg = 0.5 / rr
    radp = (p * (comp * fs - 0.5 * h - rg - tg) + flux - prev + ts) / (comp * fs + 0.5 * h + rg + tg)
    uln = (prev + h * (p + radp)) * c['radmask']
    radflow = ((prev + uln) * 0.5 + rg * (p + radp)) * c['radmask']
    pn = c['radmask'].where(radp, pn)
    pn = c['fixedmask'].where(c['fixedp'], pn)
    newnx = [las[j] * nx[j] + lbs[j] * (p + pn) for j in range(8)] if lbs else nx
    radiated = radflow + wallflow * c['wallrad']
    out = (radiated - old_rad).sum() * fs
    outputs = [pn, l * u, l, wvn, wyn, uln, lr, radiated, *newbx, *newnx, *fold_outputs, comp * RHO_C2]
    return (g, state, coefficients, signals, fixed, outputs, out)
