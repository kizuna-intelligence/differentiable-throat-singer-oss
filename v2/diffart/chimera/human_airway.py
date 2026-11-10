"""Complete human airway topology. Reference: CHIMERA voice/human_airway.rs.
MIT, Copyright (c) 2026 Jameson Hodge. See THIRD_PARTY_NOTICES.md.
"""
import torch
from .network import AirwayNetwork,TubeSpec,PortSpec
from .constants import WALL_SOFT,WALL_TRACHEA,WALL_NASAL,RHO,C_SOUND,CM_H2O


def build_human_airway(areas=None,length=17.,scale=1.,velum=0.,sample_rate=96000,pressure=8*CM_H2O):
    net=AirwayNetwork(sample_rate)
    dx_min=C_SOUND/sample_rate/.8
    n=max(16,int(length/dx_min)) if areas is None else len(areas)
    split=n//2
    if areas is None:areas=[3.*scale*scale]*n
    tr=net.add_tube(TubeSpec('trachea',max(4,int(12*scale/max(.9*scale,dx_min))),12*scale,2.5*scale*scale,WALL_TRACHEA))
    ph=net.add_tube(TubeSpec('pharynx',split,length*split/n,areas[:split],WALL_SOFT,1.25,.25))
    oral=net.add_tube(TubeSpec('oral',n-split,length*(n-split)/n,areas[split:],WALL_SOFT,1.5,.25))
    nn=max(4,int(11.4*scale/max(.6*scale,dx_min)))
    nasal_area=[(2.6 if (i+.5)/nn<.18 else 2.-.6*((i+.5)/nn-.18) if (i+.5)/nn<.75 else 1.4 if (i+.5)/nn<.9 else 1.)*scale*scale for i in range(nn)]
    nose=net.add_tube(TubeSpec('nasal',nn,11.4*scale,nasal_area,WALL_NASAL,3.2))
    fossa=net.add_tube(TubeSpec('piriform',max(1,int(2*scale/dx_min)),2*scale,.9*scale*scale,WALL_SOFT,1.3))
    lung=net.add_pressure_node('lungs',pressure)
    net.connect(lung,tr[0],PortSpec(2.5*scale*scale,1.,0.,0.,resistance=.5*RHO*C_SOUND/(2.5*scale*scale)))
    net.glottis_edge=net.connect(tr[-1],ph[0],PortSpec(.05,.3*scale,slit_width=1.4*scale))
    net.connect(ph[-1],oral[0])
    net.velum_edge=net.connect(ph[-1],nose[0],PortSpec(velum,1.*scale,.3,.8))
    net.connect(ph[min(len(ph)-1,round(2.6/8.8*len(ph)))],fossa[0])
    sinus=net.add_cavity('sinus',14*scale**3)
    net.connect(nose[round(nn*.45)],sinus,PortSpec(.12*scale**2,.5*scale,0.,0.,half_sections=False))
    net.add_radiation(oral[-1]);net.add_radiation(nose[-1])
    net.finalize()
    # Jet sources about 1 cm downstream, with the teeth obstacle anteriorly.
    ph_edges=[j for j,(a,b) in enumerate(net._edges) if a in ph and b in ph]
    oral_edges=[j for j,(a,b) in enumerate(net._edges) if a in oral and b in oral]
    net.noise_sources=[(net.glottis_edge,ph_edges[min(1,len(ph_edges)-1)],max(0,len(tr)-2),3.2e-6)]
    for edges in (ph_edges,oral_edges):
        for i,edge in enumerate(edges):
            net.noise_sources.append((edge,edges[min(i+2,len(edges)-1)],edges[max(0,i-2)],2.2e-6))
    net.noise_sources.append((oral_edges[-3],oral_edges[-1],oral_edges[-4],3.e-6))
    return net
