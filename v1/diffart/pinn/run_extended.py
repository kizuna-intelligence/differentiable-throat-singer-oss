"""Reproducible CPU validation of steps, asymmetric transverse modes and bends.

Produces independent finite-volume grid convergence and spectral order studies,
including a deliberately retained failed strong-collocation step baseline.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy import special

from .finite_volume import solve
from .multidomain import Piece, TubePINN
from .uniform import C, plane_transfer

HERE = Path(__file__).resolve().parent


def complex_json(v):
    return dict(real=float(np.real(v)),imag=float(np.imag(v)),magnitude=float(abs(v)))


def relative_field(model,ref,seed=904):
    rng = np.random.default_rng(seed)
    ix = rng.choice(ref.cells,size=min(ref.cells,4096),replace=False)
    predicted = np.empty(len(ix),complex)
    for j in range(len(model.pieces)):
        used = ref.piece[ix] == j
        predicted[used] = model.pressure(j,ref.x[ix[used]],ref.y[ix[used]])
    truth,weight = ref.pressure[ix],ref.volumes[ix]
    return float(np.sqrt(np.sum(weight*abs(predicted-truth)**2)/np.sum(weight*abs(truth)**2)))


def one_dimensional(pieces,frequency,z,axisymmetric=False):
    matrix = np.eye(2,dtype=complex)
    areas = [(p.upper**2-p.lower**2)/2 if axisymmetric else p.width for p in pieces]
    for p,area in zip(pieces,areas):
        phase = 2*np.pi*frequency*p.length/C
        co,si = np.cos(phase),np.sin(phase)
        matrix = matrix@np.array([[co,1j*si/area],[1j*si*area,co]])
    return 1/(matrix[1,0]*z/areas[-1]+matrix[1,1])


def slab_reference(x,y,frequency,p,z,amplitudes=(1.,.5)):
    result = np.zeros(np.broadcast(x,y).shape,complex)
    kL = 2*np.pi*frequency*p.length/C
    h = z/(1j*kL)
    t = (np.asarray(x)-p.start)/p.length
    for m,amplitude in enumerate(amplitudes):
        beta = np.sqrt(complex(kL*kL-(m*np.pi*p.length/p.width)**2))
        if abs(beta)<1e-8:
            axial = t-1-h
        elif beta.imag>1:
            eta=beta.imag
            reflection=(h*eta-1)/(h*eta+1)
            a=1/(eta*(reflection*np.exp(-2*eta)-1))
            axial=a*(np.exp(-eta*t)+reflection*np.exp(-eta*(2-t)))
        else:
            a=-(np.sin(beta)/beta+h*np.cos(beta))/(np.cos(beta)-h*beta*np.sin(beta))
            axial=a*np.cos(beta*t)+np.sin(beta*t)/beta
        result+=amplitude*axial*np.cos(m*np.pi*(np.asarray(y)-p.lower)/p.width)
    return result


def modal_ratio(model,j=0,mode=1):
    p=model.pieces[j]
    q,w=np.polynomial.legendre.leggauss(90)
    y=p.lower+(q+1)*p.width/2
    x=np.linspace(p.start+.15*p.length,p.end-.15*p.length,100)
    field=model.pressure(j,x[:,None],y[None,:])
    mean=field@(w/2)
    modal=field@(w*np.cos(mode*np.pi*(y-p.lower)/p.width))
    return float(np.linalg.norm(modal)/max(np.linalg.norm(mean),1e-30))


def main(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out.mkdir(parents=True,exist_ok=True)
    start=time.monotonic()
    summary=dict(convention='exp(+i omega t); c=350 m/s; local constant Robin z=0.15',
        limitations=['Per-frequency fixed-feature physics solve, not amortized neural inference',
            '2D slab width is not a circular diameter acoustic equivalence',
            'No pyriform sinus or full external/modal radiation model',
            'Sharp-corner strong collocation is failed and excluded from usable results'],
        step={},slab_modes=[],bend=[],uniform_fv=[])
    z=.15
    # Verify independent FV phase/sign and expected second-order grid convergence.
    for spacing in (.001,.0005,.00025,.000125):
        ref=solve([Piece(0,.17,0,.015)],4500,z,spacing,axisymmetric=True)
        summary['uniform_fv'].append(dict(spacing_m=spacing,
            transfer_relative_error=float(abs(ref.transfer/plane_transfer(4500,.17,z)-1)),
            algebraic_residual=ref.algebraic_relative_residual,cells=ref.cells))

    geometries={'axisymmetric':[Piece(0,.08,0,.01),Piece(.08,.17,0,.015)],
                'asymmetric_slab':[Piece(0,.08,0,.02),Piece(.08,.17,0,.03)]}
    for name,pieces in geometries.items():
        axis=name=='axisymmetric'
        rows=[]
        for f in (500.,1500.,3000.,4500.,6000.,7500.,8000.):
            refs=[solve(pieces,f,z,spacing,axisymmetric=axis) for spacing in (.0005,.00025,.000125)]
            ref=refs[-1]
            model=TubePINN(pieces,axisymmetric=axis,nx=40,ny=18).train_variational(f,z)
            h=model.transfer()
            h1=one_dimensional(pieces,f,z,axis)
            row=dict(frequency_hz=f,transfer=complex_json(h),reference=complex_json(ref.transfer),
                one_dimensional=complex_json(h1),field_relative_l2=relative_field(model,ref),
                transfer_relative_error=float(abs(h/ref.transfer-1)),
                versus_1d_db=float(20*np.log10(abs(h/h1))),
                fv_grids=[dict(spacing_m=s,cells=r.cells,transfer=complex_json(r.transfer),
                    algebraic_residual=r.algebraic_relative_residual) for s,r in zip((.0005,.00025,.000125),refs)],
                fine_grid_transfer_change=float(abs(refs[-2].transfer/ref.transfer-1)),training=model.training)
            if not axis:row['first_transverse_to_plane_l2']=modal_ratio(model,1)
            rows.append(row)
            model.save(out/f'{name}_{int(f)}.npz')
            print(name,f,'field',row['field_relative_l2'],'H',row['transfer_relative_error'],flush=True)
        convergence=[]
        ref=solve(pieces,4500,z,.000125,axisymmetric=axis)
        for nx,ny in ((16,6),(24,10),(32,14),(40,18)):
            model=TubePINN(pieces,axisymmetric=axis,nx=nx,ny=ny).train_variational(4500,z)
            convergence.append(dict(nx=nx,ny=ny,field_relative_l2=relative_field(model,ref),
                transfer_relative_error=float(abs(model.transfer()/ref.transfer-1))))
        strong=TubePINN(pieces,axisymmetric=axis,nx=24,ny=10).train(4500,z)
        summary['step'][name]=dict(pieces=[asdict(p) for p in pieces],axisymmetric=axis,
            frequencies=rows,spectral_convergence=convergence,
            failed_strong_baseline=dict(training=strong.training,field_relative_l2=relative_field(strong,ref),
                transfer_relative_error=float(abs(strong.transfer()/ref.transfer-1)),
                diagnosis='Corner singularity and boundary measure/weighting; small residual is insufficient'))

    # An odd transverse mode is now representable: no rho² symmetry restriction.
    p=Piece(0,.17,0,.03)
    cutoff=C/(2*p.width)
    rng=np.random.default_rng(312)
    x,y=rng.random((2,2000));x*=p.length;y*=p.width
    drive=lambda y:1+.5*np.cos(np.pi*y/p.width)
    for f in (4000.,5500.,6000.,7000.,8000.):
        model=TubePINN([p],nx=48,ny=12).train(f,z,drive)
        truth=slab_reference(x,y,f,p,z)
        field_error=float(np.linalg.norm(model.pressure(0,x,y)-truth)/np.linalg.norm(truth))
        # Infer beta² from a projected transverse amplitude and its derivatives.
        q,w=np.polynomial.legendre.leggauss(80)
        yy=p.width*(q+1)/2
        xx=np.linspace(.02,.15,70)
        phi=w*np.cos(np.pi*yy/p.width)
        a=model.pressure(0,xx[:,None],yy[None,:])@phi
        axx=model.pressure(0,xx[:,None],yy[None,:],dx=2)@phi
        beta2=-np.vdot(a,axx)/np.vdot(a,a)
        inferred=C/(2*np.pi)*np.sqrt((2*np.pi*f/C)**2-beta2.real)
        row=dict(frequency_hz=f,field_relative_l2=field_error,
            beta_squared_m2=complex_json(beta2),cutoff_inferred_hz=float(inferred),
            cutoff_theory_hz=cutoff,first_transverse_to_plane_l2=modal_ratio(model),training=model.training)
        summary['slab_modes'].append(row)
        model.save(out/f'slab_mode_{int(f)}.npz')
    control=TubePINN([p],nx=48,ny=12).train(7000,z)
    summary['slab_uniform_drive_control']=dict(frequency_hz=7000,
        first_transverse_to_plane_l2=modal_ratio(control))
    summary['circular_cutoffs_at_radius_15mm_hz']=dict(
        axisymmetric=float(C*special.jn_zeros(1,1)[0]/(2*np.pi*.015)),
        first_nonaxisymmetric=float(C*special.jnp_zeros(1,1)[0]/(2*np.pi*.015)))

    # Circular-arc coordinates: h=1-kappa*y, centreline arc length is fixed.
    p=Piece(0,.17,-.015,.015)
    for curvature in (0.,6.):
        for f in (1000.,4500.,6000.,7500.):
            model=TubePINN([p],curvature=curvature,nx=48,ny=14).train(f,z)
            coarse=solve([p],f,z,.0005,curvature=curvature)
            ref=solve([p],f,z,.00025,curvature=curvature)
            row=dict(curvature_per_m=curvature,angle_deg=float(curvature*p.length*180/np.pi),
                frequency_hz=f,field_relative_l2=relative_field(model,ref),
                transfer_relative_error=float(abs(model.transfer()/ref.transfer-1)),
                fine_grid_transfer_change=float(abs(coarse.transfer/ref.transfer-1)),
                transfer=complex_json(model.transfer()),reference=complex_json(ref.transfer),
                versus_straight_db=float(20*np.log10(abs(model.transfer()/plane_transfer(f,.17,z)))),
                first_transverse_to_plane_l2=modal_ratio(model),training=model.training)
            summary['bend'].append(row)
            model.save(out/f'bend_{int(curvature)}_{int(f)}.npz')
            print('bend',curvature,f,'field',row['field_relative_l2'],flush=True)
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    for name,data in summary['step'].items():
        rows=data['frequencies'];f=[r['frequency_hz'] for r in rows]
        axes[0,0].plot(f,[r['versus_1d_db'] for r in rows],'o-',label=name)
        axes[0,1].semilogy(f,[r['field_relative_l2'] for r in rows],'o-',label=name)
        c=data['spectral_convergence']
        axes[1,0].semilogy([r['ny'] for r in c],[r['field_relative_l2'] for r in c],'o-',label=name)
    modes=summary['slab_modes']
    axes[1,1].plot([r['frequency_hz'] for r in modes],[r['beta_squared_m2']['real'] for r in modes],'o-',label='learned odd mode')
    axes[1,1].axhline(0,color='gray');axes[1,1].axvline(cutoff,color='gray',ls='--')
    for ax,title,xlabel,ylabel in zip(axes.ravel(),
        ['Stepped tube vs 1D','Independent FV comparison','Weak spectral order convergence','2D slab odd transverse mode'],
        ['Frequency (Hz)','Frequency (Hz)','Transverse degree','Frequency (Hz)'],
        ['Transfer ratio (dB)','Relative pressure L2','Relative pressure L2','Axial wavenumber squared (1/m2)']):
        ax.set(title=title,xlabel=xlabel,ylabel=ylabel);ax.grid(alpha=.3);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(out/'validation.png',dpi=150);plt.close(fig)
    # Save fields in actual bent Cartesian coordinates for visual inspection.
    model=TubePINN([p],curvature=6.,nx=48,ny=14).train(7500,z)
    s,y=np.meshgrid(np.linspace(0,.17,260),np.linspace(-.015,.015,70))
    theta=6*s
    X=(1/6-y)*np.sin(theta);Y=1/6-(1/6-y)*np.cos(theta)
    fig,ax=plt.subplots(figsize=(10,3))
    im=ax.pcolormesh(X*100,Y*100,abs(model.pressure(0,s,y)),shading='auto',cmap='magma')
    ax.set(aspect='equal',xlabel='x (cm)',ylabel='y (cm)',title='Curved 2D slab, 7500 Hz, |p| (normalized)')
    fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(out/'bent_field.png',dpi=150);plt.close(fig)
    summary['elapsed_seconds']=time.monotonic()-start
    summary['source_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')}
    (out/'metrics.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('Done in',summary['elapsed_seconds'],'seconds',flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,default=HERE/'results/extended')
    main(ap.parse_args().out)
