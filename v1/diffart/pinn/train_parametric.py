"""Physics-only parametric geometry PINN, isolated from the frozen throat.

The uniform validation baseline is run_uniform.py. This trainer is an initial
implementation of the future amortized surrogate, not a validated correction
teacher. A short CPU smoke run only verifies training and checkpointing.
"""
import argparse
import contextlib
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


HERE = Path(__file__).resolve().parent
ALLOWED_GPU = 'GPU-cd879302-07f2-435c-b24f-c3fdab3bbf7d'  # local RTX 3090 only


@contextlib.contextmanager
def device_lease(request):
    """Same shared UUID flock as diffart.runtime; acquire BEFORE importing torch.

    Require idle GPU, not merely a few GB of free memory during another job.
    CPU fallback for --device auto, fail visibly for explicit occupied GPU.
    """
    if request == 'cpu':
        yield 'cpu', {'device': 'cpu', 'reason': 'requested CPU'}
        return
    if request not in ('auto', ALLOWED_GPU):
        raise ValueError('Only cpu, auto, or the local 3090 UUID is allowed')
    lock = None
    try:
        lock = open(f'/tmp/cyborgy-gpu-{ALLOWED_GPU}.lock', 'a+')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        row = subprocess.check_output(['nvidia-smi', '-i', ALLOWED_GPU,
            '--query-gpu=uuid,name,memory.used,utilization.gpu',
            '--format=csv,noheader,nounits'], text=True).strip()
        uuid, name, used, utilization = (v.strip() for v in row.split(','))
        if uuid != ALLOWED_GPU or '3090' not in name or int(used) > 512 or int(utilization) > 5:
            raise RuntimeError('3090 not idle: ' + row)
        jobs = subprocess.check_output(['nvidia-smi', '-i', ALLOWED_GPU,
            '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip()
        if jobs:
            raise RuntimeError('3090 still has compute processes: ' + jobs)
    except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
        if lock is not None:
            lock.close()
        if request != 'auto':
            raise
        yield 'cpu', {'device': 'cpu', 'reason': str(exc), 'checked_utc': datetime.now(timezone.utc).isoformat()}
        return
    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = ALLOWED_GPU
        yield 'cuda:0', {'device': ALLOWED_GPU, 'idle_check': row, 'lock': lock.name}
    finally:
        lock.close()


def main(args):
    with device_lease(args.device) as (device, device_info):
        import numpy as np
        import torch
        from .field import (AreaGeometry, ParametricField, helmholtz_residual,
                            inlet_residual, radiation_residual, wall_residual)
        from .uniform import piston_impedance

        torch.set_num_threads(2)
        torch.manual_seed(args.seed)
        dtype = torch.float64
        with np.load(args.shapes) as d:
            geometry = AreaGeometry(d['log_area_mean_m2'], d['basis'], float(d['length_m'])).to(device)
            samples = torch.as_tensor(d['coefficients'], dtype=dtype, device=device)
        # Keep the final 20% of extracted frames out of the optimizer. PCA is
        # a shared representation fitted previously; this is not song CV.
        split = int(len(samples) * (1-args.holdout_fraction))
        train_samples, validation_samples = samples[:split], samples[split:]
        model = ParametricField(samples.shape[1], width=args.width).to(device=device, dtype=dtype)
        model.coefficient_scale.copy_(samples.std(0).clamp_min(1e-5))
        if args.resume is not None:
            checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
            model.load_state_dict(checkpoint['model'])
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        length = geometry.length

        def batch(count, max_frequency, population):
            coefficients = population[torch.randint(len(population), (count,), device=device)]
            x = torch.rand(count, 1, dtype=dtype, device=device) * length
            s = torch.rand_like(x)
            frequency = 10 + torch.rand_like(x) * (max_frequency - 10)
            with torch.no_grad():
                rho = geometry.radius(x, coefficients) * s.sqrt()
            # x and rho must be independent physical coordinates for derivatives.
            return x.requires_grad_(), rho.requires_grad_(), frequency, coefficients

        def loss_parts(count, max_frequency, population=train_samples):
            x, rho, f, coeff = batch(count, max_frequency, population)
            p = model(x, rho, f, coeff, geometry)
            normalizer = 1 + (2 * torch.pi * f * length / 350)**2
            pde = (length**2 * helmholtz_residual(p, x, rho, f) / normalizer).square().mean()
            boundary_count = max(12, count // 4)
            x, rho, f, coeff = batch(boundary_count, max_frequency, population)
            # Rigid sloping wall: p_r - r'(x)*p_x=0.
            with torch.no_grad():
                wall_rho = geometry.radius(x, coeff)
            wall_rho = wall_rho.detach().requires_grad_()
            p = model(x, wall_rho, f, coeff, geometry)
            wall = (wall_rho * wall_residual(p, x, wall_rho, coeff, geometry)).square().mean()
            inlet_x = torch.zeros_like(x).requires_grad_()
            # Independently sample each end's cross-sectional disk.
            with torch.no_grad():
                inlet_rho = geometry.radius(inlet_x, coeff) * torch.rand_like(x).sqrt()
            inlet_rho = inlet_rho.detach().requires_grad_()
            p = model(inlet_x, inlet_rho, f, coeff, geometry)
            target = torch.cat((torch.ones_like(x) / length, torch.zeros_like(x)), -1)
            inlet = (length * inlet_residual(p, inlet_x, target)).square().mean()
            outlet_x = torch.full_like(x, length).requires_grad_()
            with torch.no_grad():
                end_radius = geometry.radius(outlet_x, coeff)
                outlet_rho = end_radius * torch.rand_like(x).sqrt()
                z = piston_impedance(f.detach().cpu().numpy(), end_radius.detach().cpu().numpy())
                z = torch.as_tensor(z, dtype=torch.complex128, device=device)
            outlet_rho = outlet_rho.detach().requires_grad_()
            p = model(outlet_x, outlet_rho, f, coeff, geometry)
            outlet = radiation_residual(p, outlet_x, f, z).square().mean()
            return dict(pde=pde, inlet=inlet, wall=wall, radiation=outlet)

        start = time.monotonic()
        history = []
        for step in range(args.steps):
            # Explicit 0--5k curriculum; 8k is an optional later stage, gated by
            # independent validation, not automatically selected here.
            fraction = min(1., (step + 1) / max(1, args.steps * .7))
            maximum = 250 + (args.fmax - 250) * fraction
            parts = loss_parts(args.points, maximum)
            loss = parts['pde'] + args.boundary_weight * (parts['inlet'] + parts['wall'] + parts['radiation'])
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite PINN loss')
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
            optimizer.step()
            if step == 0 or (step + 1) % max(1, args.steps // 10) == 0:
                row = dict(step=step + 1, max_frequency_hz=maximum,
                           loss=float(loss.detach()), **{k: float(v.detach()) for k, v in parts.items()})
                history.append(row)
                print(json.dumps(row), flush=True)
        # New random collocation points, still a residual check, not reference validation.
        heldout = {k: float(v.detach()) for k, v in loss_parts(8 * args.points, args.fmax, validation_samples).items()}
        # Per-band checks expose the zero-field/inlet compromise hidden by total loss.
        by_band = {}
        for ceiling in (250., 1000., args.fmax):
            by_band[str(ceiling)] = {k: float(v.detach()) for k,v in
                loss_parts(8*args.points, ceiling, validation_samples).items()}
        args.out.mkdir(parents=True, exist_ok=True)
        torch.save(dict(model=model.state_dict(), geometry=geometry.state_dict(),
                        length_m=length, args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                        validation_status='UNVERIFIED: physics residual check only'), args.out / 'field.pt')
        summary = dict(device=device_info, seed=args.seed, steps=args.steps, fmax_hz=args.fmax,
                       elapsed_seconds=time.monotonic() - start, history=history,
                       heldout_residual_mse=heldout, heldout_by_frequency_ceiling=by_band,
                       shape_split=dict(train_frames=len(train_samples),heldout_frames=len(validation_samples),
                           description='Contiguous final frames withheld; shared pre-fitted PCA, not song CV'),
                       shape_pca_sha256=hashlib.sha256(args.shapes.read_bytes()).hexdigest(),
                       source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                      for p in HERE.glob('*.py')},
                       reference_validation='NOT DONE: not a usable correction teacher')
        (args.out / 'training.json').write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--shapes', type=Path, default=HERE / 'results/shapes/area_pca.npz')
    ap.add_argument('--out', type=Path, default=HERE / 'results/parametric_initial')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--points', type=int, default=256)
    ap.add_argument('--width', type=int, default=64)
    ap.add_argument('--fmax', type=float, default=5000)
    ap.add_argument('--holdout-fraction', type=float, default=.2)
    ap.add_argument('--boundary-weight', type=float, default=1.)
    ap.add_argument('--resume', type=Path)
    ap.add_argument('--seed', type=int, default=20261109)
    args = ap.parse_args()
    if args.fmax < 250 or args.fmax > 5000 or args.steps < 1 or args.points < 16:
        ap.error('Initial stage requires 250 <= fmax <= 5000, steps >= 1, points >= 16')
    if not 0 < args.holdout_fraction < 1 or args.boundary_weight <= 0:
        ap.error('Positive boundary weight and 0 < holdout fraction < 1 required')
    main(args)
