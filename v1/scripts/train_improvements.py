"""B1/B2 fine-tuning on the identical sampled hour and validation split.

CPU is supported; CUDA requires a UUID shared lock before importing torch.
"""
import argparse
import json
import random
import time
from pathlib import Path
from diffart.runtime import device_lease, save_json
from diffart.data import ROOT, SR, HOP, sha

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('variant', choices=['b1', 'b2', 'both'])
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--unfreeze-base', action='store_true', help='Also tune the legacy speaker and articulation; default isolates new components')
    ap.add_argument('--steps', type=int, default=72000)
    ap.add_argument('--threads', type=int, default=3)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--batch-size', type=int, default=1)
    ap.add_argument('--seconds', type=float, default=1.0)
    ap.add_argument('--patience', type=int, default=3)
    args = ap.parse_args()
    with device_lease(args.device) as device:
        import torch
        from diffart.dataset import Corpus
        from diffart.improved import ImprovedSynthesizer
        from diffart.losses import SpectralLoss
        torch.set_num_threads(args.threads)
        torch.manual_seed(20261109)
        if device.startswith('cuda'):
            torch.cuda.set_per_process_memory_fraction(0.15)
        corpus = Corpus()
        m = corpus.manifest
        model = ImprovedSynthesizer(len(m['vocab']), b1=args.variant in ('b1', 'both'), b2=args.variant in ('b2', 'both')).to(device)
        criterion = SpectralLoss().to(device)
        out = ROOT / 'artifacts/improvements' / args.variant
        out.mkdir(parents=True, exist_ok=True)
        rng = random.Random(20261109)
        if not args.unfreeze_base:
            for name, parameter in model.named_parameters():
                parameter.requires_grad_(name.startswith(('piriform.', 'glottis.')))
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=0.002)
        songs = [r for r in m['songs'] if r['split'] == 'train' and r.get('train_seconds', 0) > 0]
        phrases = []
        for song in songs:
            phrase = corpus.phrase(song)
            if phrase and (phrase['start_frame'] + phrase['frames']) * HOP / SR <= song['train_seconds']:
                phrases.append((song, phrase))
        song, phrase = phrases[0]
        save_json(out / 'phrase.json', phrase)
        model.load_legacy(torch.load(ROOT / 'artifacts/main/best.pt', map_location=device, weights_only=True)['model'])
        save_json(out / 'provenance.json', dict(base_checkpoint_sha256=sha(ROOT / 'artifacts/main/best.pt'), physics_sha256=sha(ROOT / 'diffart/physics.py'), improved_sha256=sha(ROOT / 'diffart/improved.py'), manifest_sha256=sha(ROOT / 'data/manifest.json'), args=vars(args)))
        first = 0
        best = float('inf')
        stale = 0
        initial = None
        ck = None
        if args.resume and (out / 'latest.pt').exists():
            ck = torch.load(out / 'latest.pt', map_location=device, weights_only=True)
            assert ck['manifest_sha256'] == sha(ROOT / 'data/manifest.json')
            model.load_state_dict(ck['model'])
            optimizer.load_state_dict(ck['optimizer'])
            first = ck['step']
            best = ck['best']
            stale = ck.get('stale', 0)
            initial = ck['initial']
            rng.setstate(ck['rng'])
            torch.set_rng_state(ck['torch_rng'].cpu())

        def validation():
            model.eval()
            values = []
            with torch.no_grad():
                for row in [r for r in m['songs'] if r['split'] == 'val']:
                    n = len(corpus.load(row)['f0'])
                    size = int(args.seconds * SR / HOP)
                    for a in range(0, n - size + 1, size):
                        data = corpus.excerpt(row, a, size, device)
                        if data[-1].square().mean() < 1e-07:
                            continue
                        y = model(*data[:3], seed=1234)
                        loss, _ = criterion(y, data[3])
                        values.append(float(loss))
            model.train()
            return sum(values) / len(values)
        frames = int(args.seconds * SR / HOP)
        blocks = [(r, a) for r in songs for a in range(0, int(r['train_seconds'] * SR / HOP) - frames + 1, frames)]
        if ck and 'block_order' in ck:
            by_id = {r['id']: r for r in songs}
            blocks = [(by_id[sid], a) for sid, a in ck['block_order']]
        if initial is None:
            # Exact-output regression and unchanged loss/grid permit reuse of the
            # original full validation result, avoiding three identical passes.
            base_result=json.loads((ROOT/'artifacts/main/result.json').read_text())
            assert base_result['checkpoint_sha256']==sha(ROOT/'artifacts/main/best.pt')
            initial=base_result['best'] if args.seconds==1.0 else validation()
        start = time.monotonic()
        log = (out / 'training.jsonl').open('a')
        history = []
        best_step = ck.get('best_step', 0) if ck else 0

        def checkpoint(step, val, baseline=False):
            nonlocal best, stale, best_step
            improved = val < best - 0.002
            if improved:
                best = val
                stale = 0
                best_step = step
            elif not baseline:
                stale += 1
            ck = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), step=step, best=best, initial=initial, manifest_sha256=sha(ROOT / 'data/manifest.json'), rng=rng.getstate(), torch_rng=torch.get_rng_state(), args=vars(args), stale=stale, best_step=best_step, block_order=[(r['id'], a) for r, a in blocks])
            torch.save(ck, out / 'latest.pt')
            if improved:
                torch.save(ck, out / 'best.pt')
            metric = dict(step=step, val_loss=val, best=best, initial=initial, relative=val / initial, elapsed=time.monotonic() - start)
            history.append(metric)
            log.write(json.dumps(metric) + '\n')
            log.flush()
            print(json.dumps(metric), flush=True)
            save_json(out / 'progress.json', metric)
        checkpoint(first, validation() if first else initial, baseline=bool(first))
        cursor = first
        while cursor < args.steps:
            if cursor % len(blocks) == 0:
                rng.shuffle(blocks)
            count = min(args.batch_size, len(blocks) - cursor % len(blocks), args.steps - cursor)
            examples = [corpus.excerpt(*blocks[(cursor + i) % len(blocks)], frames, device) for i in range(count)]
            data = tuple((torch.stack([x[j] for x in examples]) for j in range(4))) if count > 1 else examples[0]
            step = cursor + count
            model.train()
            optimizer.zero_grad(set_to_none=True)
            y, p = model(*data[:3], seed=step, return_controls=True)
            acoustic, parts = criterion(y, data[-1])
            reg = model.regularization(p)
            loss = acoustic + reg
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite loss at {step}')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.0)
            if not torch.isfinite(norm):
                raise RuntimeError(f'Nonfinite gradient at {step}')
            optimizer.step()
            model.project(warmup=step / len(blocks))
            if step // 10 != cursor // 10:
                log.write(json.dumps(dict(step=step, train_loss=float(loss.detach()), grad_norm=float(norm), parts={k: float(v.detach()) for k, v in parts.items()}, source_noise_gain=float((0.03 * p['amplitude'] * p['breath']).mean().detach()), constriction_noise_gain=float((0.03 * p['amplitude'] * p['turbulence']).mean().detach()))) + '\n')
                log.flush()
            cursor = step
            interval = len(blocks)
            if step % interval == 0 or step == args.steps:
                checkpoint(step, validation())
                if stale >= args.patience:
                    break
        log.close()
        result = dict(best_step=best_step, initial=initial, best=best, ratio=best / initial, step=step, elapsed=time.monotonic() - start, device=device, dataset_seconds=3600, epoch_blocks=len(blocks), validation_history=history, completed_epochs=step / len(blocks), batch_size=args.batch_size, stopping_reason='validation_patience' if stale >= args.patience else 'maximum_sampled_blocks', checkpoint_sha256=sha(out / 'best.pt'))
        save_json(out / 'result.json', result)
if __name__ == '__main__':
    main()
