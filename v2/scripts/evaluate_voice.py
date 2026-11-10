#!/usr/bin/env python3
"""Offline held-out spectral evaluation; no external speaker or ASR weights."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from diffart.chimera.training import Corpus, make_model, evaluate

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('data',type=Path);p.add_argument('checkpoint',type=Path)
    p.add_argument('--split',choices=['val','test'],default='test')
    p.add_argument('--frames',type=int,default=100)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    a=p.parse_args();ck=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    corpus=Corpus(a.data,a.frames)
    if ck['config']['vocab']!=corpus.vocab:
        raise ValueError('Checkpoint vocabulary differs')
    m=make_model(ck['config'],ck['model'],a.device)
    print(json.dumps(dict(split=a.split,spectral_loss=evaluate(m,corpus,a.split,device=a.device))))
