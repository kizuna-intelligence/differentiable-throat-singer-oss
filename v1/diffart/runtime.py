"""Only cooperating, exclusive UUID locks; never stop another process."""
import contextlib
import fcntl
import json
import os
import subprocess
from pathlib import Path

@contextlib.contextmanager
def device_lease(device='cpu', min_free_mb=3500):
    if device == 'cpu':
        yield 'cpu'
        return
    if not device.startswith('GPU-'):
        raise ValueError('Pass an NVIDIA GPU UUID, or cpu')
    lock = open(f'/tmp/cyborgy-gpu-{device}.lock', 'a+')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        row = subprocess.check_output(['nvidia-smi', '-i', device,
            '--query-gpu=uuid,memory.free', '--format=csv,noheader,nounits'], text=True)
        uuid, free = row.strip().split(',')
        if uuid.strip() != device or int(free) < min_free_mb:
            raise RuntimeError(f'Insufficient free GPU memory: {row}')
        os.environ['CUDA_VISIBLE_DEVICES'] = device
        # Acquire before importing torch/creating any CUDA context.
        yield 'cuda:0'
    finally:
        lock.close()

def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    tmp.replace(path)
