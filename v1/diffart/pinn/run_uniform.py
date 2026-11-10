"""Reproducible C milestone: 0--5 kHz gate, then 5--8 kHz and radial modes.

OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python -m diffart.pinn.run_uniform
All products remain under diffart/pinn/results/uniform.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize, special
import scipy

from .uniform import C, UniformPINN, modal_reference, piston_impedance, plane_transfer


HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def check_case(model, f, z, drive=None, amplitudes=(1.,), save=None):
    model.train(float(f), z, drive)
    rng = np.random.default_rng(9731)
    t, s = rng.random((2, 1024))
    prediction = model.pressure(t, s)
    reference = modal_reference(t, s, f, model.length, model.radius, z, amplitudes)
    error = np.linalg.norm(prediction - reference) / np.linalg.norm(reference)
    record = dict(frequency_hz=float(f), radius_m=model.radius,
                  field_relative_l2=float(error), **model.validate(drive), **model.training)
    if save:
        model.save(save)
    return record


def project_mode(model, t, alpha):
    # Cross-sectional area measure ds=2 rho/R² d rho.
    s, w = np.polynomial.legendre.leggauss(40)
    s, w = (s + 1) / 2, w / 2
    radial = special.j0(alpha * np.sqrt(s))
    field = model.pressure(np.asarray(t)[:, None], s[None, :])
    return (field * (w * radial)[None]).sum(-1) / np.sum(w * radial**2)


def main(out, quick=False):
    out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    protected = [HERE.parents[1] / 'artifacts/main/best.pt', HERE.parent / 'physics.py']
    protected_sha = {str(p): sha(p) for p in protected}
    length, radius, z = .17, .01, .03
    model = UniformPINN(length=length, radius=radius, axial_degree=40,
                        radial_degree=3, interior_points=700)
    records = []
    step = 250 if quick else 100
    for stage, frequencies in [('0_5k', np.arange(0, 5001, step)),
                                ('5_8k', np.arange(5000 + step, 8001, step))]:
        current = []
        for f in frequencies:
            r = check_case(model, f, z if f > 0 else 0.)
            r['stage'] = stage
            r['boundary'] = 'resistive_open_z=0.03' if f > 0 else 'static pressure gauge p(1)=0; DC transfer by limit'
            records.append(r)
            current.append(r)
        max_error = max(r['field_relative_l2'] for r in current)
        print(f'{stage}: {len(current)} frequencies, max field error {max_error:.3g}', flush=True)
        json_write(out / (stage + '.json'), current)
        if max_error > .01:
            raise RuntimeError('Uniform validation gate failed before extending frequency range')
    # Locate peaks through fresh PINN solves, rather than drawing exact peaks.
    peaks = []
    for n in range(5):
        expected = (2 * n + 1) * C / (4 * length)
        def gain(f):
            model.train(f, z)
            return abs(model.axial_derivative(1., .5))
        found = optimize.minimize_scalar(lambda f: -gain(f),
                                         bounds=(expected - 71, expected + 47),
                                         method='bounded', options={'xatol': .005})
        peaks.append(dict(order=n + 1, theory_hz=expected, pinn_hz=float(found.x),
                          absolute_error_hz=abs(float(found.x) - expected),
                          gain=float(-found.fun)))
    model.train(8000, z).save(out / 'plane_8000.npz')
    # A physical, frequency-dependent radiation approximation checked against
    # the analytic tube with EXACTLY THE SAME load. Not a free-field benchmark.
    radiation = []
    for f in (100, 300, 500, 1000, 2000, 4000, 5000, 6500, 8000):
        zz = piston_impedance(f, radius)
        r = check_case(model, f, zz)
        r['impedance_real'], r['impedance_imag'] = float(zz.real), float(zz.imag)
        r['transfer_relative_error'] = float(abs(model.axial_derivative(1., .5) /
                                               plane_transfer(f, length, zz) - 1))
        radiation.append(r)
    # Test additional radii independently; no inference about shape amortization.
    radii = []
    for rr in (.006, .015):
        mm = UniformPINN(radius=rr, axial_degree=40, radial_degree=3, interior_points=700)
        for f in (500., 4500., 8000.):
            radii.append(check_case(mm, f, piston_impedance(f, rr)))
    alpha = float(special.jn_zeros(1, 1)[0])
    wide_radius = .04  # deliberately non-anatomical; cutoff inside 0--8 kHz
    cutoff = C * alpha / (2 * np.pi * wide_radius)
    drive = lambda s: 1 + .5 * special.j0(alpha * np.sqrt(s))
    wide = UniformPINN(radius=wide_radius, axial_degree=48, radial_degree=8,
                       interior_points=1800)
    transverse = []
    snapshots = []
    for f in (4000., 5000., cutoff, 5500., 6000., 7000., 8000.):
        r = check_case(wide, f, .15, drive, (1., .5),
                       out / f'transverse_{round(f)}.npz')
        t = np.linspace(.1, .9, 401)
        mode = project_mode(wide, t, alpha)
        dt = t[1] - t[0]
        curvature = (mode[2:] - 2 * mode[1:-1] + mode[:-2]) / dt**2
        # Estimate beta²=-a''/a FROM the learned field, including below cutoff.
        beta2 = -np.vdot(mode[1:-1], curvature) / np.vdot(mode[1:-1], mode[1:-1])
        theory = (2 * np.pi * f * length / C)**2 - (alpha * length / wide_radius)**2
        r.update(beta2_estimate_real=float(beta2.real), beta2_estimate_imag=float(beta2.imag),
                 beta2_theory=float(theory), mode_at_midpoint_abs=float(abs(mode[len(t)//2])),
                 regime='propagating' if theory > 1e-8 else ('evanescent' if theory < -1e-8 else 'cutoff'))
        transverse.append(r)
        snapshots.append((f, wide.pressure(np.linspace(0, 1, 150)[None, :],
                                           np.linspace(0, 1, 80)[:, None]**2)))
        print(f'transverse {f:.2f} Hz: field error {r["field_relative_l2"]:.3g}, '
              f'beta² {beta2.real:.4g} vs {theory:.4g}', flush=True)
    # Control: no transverse mode with uniform drive even above cutoff.
    wide.train(7000, .15)
    plane = project_mode(wide, np.linspace(0, 1, 101), 0)
    radial = project_mode(wide, np.linspace(0, 1, 101), alpha)
    unexcited = float(np.linalg.norm(radial) / np.linalg.norm(plane))
    convergence = []
    for nx, ns, count, seed in [(24, 4, 800, 91), (36, 6, 1200, 92), (48, 8, 1800, 93)]:
        mm = UniformPINN(radius=wide_radius, axial_degree=nx, radial_degree=ns,
                         interior_points=count, seed=seed)
        r = check_case(mm, 7000., .15, drive, (1., .5))
        r.update(axial_degree=nx, radial_degree=ns, collocation_points=count)
        convergence.append(r)
    # Infer cutoff from the numerical axial dispersion, not the supplied root.
    fit = np.polyfit(np.array([r['frequency_hz'] for r in transverse])**2,
                     [r['beta2_estimate_real'] for r in transverse], 1)
    inferred_cutoff = float(np.sqrt(-fit[1] / fit[0]))
    dispersion_errors = [abs(r['beta2_estimate_real'] - r['beta2_theory']) /
                         max(1., abs(r['beta2_theory'])) for r in transverse]
    # Area extraction is optional for stand-alone uniform reproduction.
    shape_path = HERE / 'results/shapes/summary.json'
    shapes = json.loads(shape_path.read_text()) if shape_path.exists() else None
    anatomical_cutoff = C * alpha / (2 * np.pi * shapes['radius_m_range'][1]) if shapes else None
    changed = [str(p) for p in protected if sha(p) != protected_sha[str(p)]]
    if changed:
        raise RuntimeError('Protected inputs changed: ' + str(changed))
    validated = records + radiation + radii + transverse
    success = (max(r['field_relative_l2'] for r in validated) < 1e-5
               and max(r['pde_relative_l2'] for r in validated) < 1e-5
               and max(r[k] for r in validated for k in ('inlet_rms', 'wall_ds_rms', 'outlet_rms')) < 1e-6
               and max(p['absolute_error_hz'] for p in peaks) < 1.
               and unexcited < 1e-5 and max(dispersion_errors) < .001
               and abs(inferred_cutoff - cutoff) < 1.
               and convergence[-1]['field_relative_l2'] < convergence[0]['field_relative_l2'] / 1000)
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(), device='cpu',
                  elapsed_seconds=time.monotonic() - start, uniform_config=asdict(model),
                  source_sha256=protected_sha, method='fixed Chebyshev features; complex linear head; physics-only least squares',
                  experiment_source_sha256={p.name: sha(p) for p in HERE.glob('*.py')},
                  versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__),
                  parametric_geometry_surrogate='not trained or validated by this uniform milestone',
                  plane_cases=records, quarter_wave_peaks=peaks, piston_radiation_cases=radiation,
                  other_radii_cases=radii, transverse_cases=transverse, convergence=convergence,
                  first_axisymmetric_cutoff_hz=cutoff, wide_radius_m=wide_radius,
                  inferred_axisymmetric_cutoff_hz=inferred_cutoff,
                  max_dispersion_relative_error=max(dispersion_errors),
                  unexcited_transverse_to_plane_l2=unexcited,
                  extracted_shape_min_axisymmetric_local_cutoff_hz=anatomical_cutoff,
                  passed=bool(success))
    json_write(out / 'metrics.json', result)
    # Plots and tabular outputs use calculated PINN values alongside references.
    grid = np.arange(0, 8001, 5.)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
    axes[0].plot(grid, 20 * np.log10(abs(plane_transfer(grid, length, z))), label='1D same-load theory')
    sample_f, sample_h = [], []
    for f in np.arange(0, 8001, step):
        model.train(f, z if f > 0 else 0.)
        sample_f.append(f)
        sample_h.append(model.axial_derivative(1., .5))
    axes[0].scatter(sample_f, 20 * np.log10(np.abs(sample_h)), s=12, label='spectral PINN')
    axes[0].set(xlabel='Frequency (Hz)', ylabel='U_out / U_in (dB)', title='Uniform tube, L=17 cm, local resistive radiation z=0.03')
    axes[0].legend()
    axes[1].plot([r['frequency_hz'] for r in transverse], [r['beta2_theory'] for r in transverse], label='dispersion theory')
    axes[1].scatter([r['frequency_hz'] for r in transverse], [r['beta2_estimate_real'] for r in transverse], label='estimated from PINN field')
    axes[1].axhline(0, color='gray', linewidth=.7)
    axes[1].axvline(cutoff, color='gray', linestyle='--', label=f'cutoff={cutoff:.1f} Hz')
    axes[1].set(xlabel='Frequency (Hz)', ylabel='Axial beta squared', title='First axisymmetric transverse mode, R=4 cm (validation fixture)')
    axes[1].legend()
    fig.savefig(out / 'validation.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, idx in zip(axes, (0, 5)):
        f, p = snapshots[idx]
        image = ax.imshow(p.real, origin='lower', aspect='auto', extent=(0, length * 100, 0, 4), cmap='RdBu_r')
        ax.set(title=f'{f:.0f} Hz, nonuniform inlet', xlabel='x (cm)', ylabel='rho (cm)')
        fig.colorbar(image, ax=ax, label='Re normalized pressure')
    fig.savefig(out / 'transverse_fields.png', dpi=160)
    plt.close(fig)
    np.savez_compressed(out / 'plane_transfer.npz', frequency_hz=sample_f,
                        pinn_transfer=sample_h, theory_transfer=plane_transfer(np.asarray(sample_f), length, z))
    rows = ['f_hz,radius_m,field_relative_l2,pde_relative_l2,inlet_rms,wall_ds_rms,outlet_rms']
    for r in records + radiation + radii + transverse:
        rows.append(','.join(str(r[k]) for k in ('frequency_hz', 'radius_m', 'field_relative_l2',
                                                'pde_relative_l2', 'inlet_rms', 'wall_ds_rms', 'outlet_rms')))
    (out / 'validation.csv').write_text('\n'.join(rows) + '\n')
    result['elapsed_seconds'] = time.monotonic() - start
    json_write(out / 'metrics.json', result)
    print(json.dumps(dict(passed=success, elapsed_seconds=time.monotonic() - start,
                          max_plane_error=max(r['field_relative_l2'] for r in records),
                          max_transverse_error=max(r['field_relative_l2'] for r in transverse),
                          cutoff_hz=cutoff, unexcited_ratio=unexcited)), flush=True)
    if not success:
        raise SystemExit('Validation failed; inspect metrics.json')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=HERE / 'results/uniform')
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()
    main(args.out, args.quick)
