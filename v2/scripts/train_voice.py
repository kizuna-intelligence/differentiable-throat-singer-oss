#!/usr/bin/env python3
"""Train on prepared singing or speech using the main_round2 method."""
import argparse
import os
import sys
from pathlib import Path
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(key,'2')
os.environ.setdefault('CHIMERA_PORTABLE_NOISE','1')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from diffart.chimera.training import Corpus, fit


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('data',type=Path);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--epochs',type=int,default=30);p.add_argument('--patience',type=int,default=3)
    p.add_argument('--batch',type=int,default=16);p.add_argument('--lr',type=float,default=.0005)
    p.add_argument('--frames',type=int,default=100)
    p.add_argument('--init',type=Path,help='Optional same-voice checkpoint; its model terms apply')
    a=p.parse_args()
    if min(a.epochs,a.patience,a.batch,a.frames,a.lr)<=0:
        p.error('Training limits must be positive')
    fit(Corpus(a.data,a.frames),a.out,a.epochs,a.patience,a.batch,a.lr,a.device,a.init)


if __name__=='__main__':
    main()
