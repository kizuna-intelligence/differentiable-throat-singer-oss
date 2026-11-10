#!/usr/bin/env python3
"""One specification builds a complete CHIMERA song and verified replay video."""
import argparse, os, sys
from pathlib import Path
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(name,'4')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from diffart.songkit.pipeline import execute

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('spec',type=Path);p.add_argument('--out',type=Path);p.add_argument('--downloads',type=Path)
    p.add_argument('--audio-only',action='store_true',help='Synthesize and mix without Chromium/video');args=p.parse_args();execute(args.spec.resolve(),(args.out or args.spec.parent/'render').resolve(),args.downloads,audio_only=args.audio_only)
if __name__=='__main__':main()
