#!/usr/bin/env python3
"""Prepare singing or speech, without redistributing corpus material."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from diffart.chimera.prepare import prepare

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('manifest',type=Path)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--aligner',type=Path,help='User-obtained Narabas ONNX weights for speech alignment')
    p.add_argument('--aligner-symbols',type=Path,help='User-obtained Narabas symbols.py (read as literal data only)')
    a=p.parse_args();prepare(a.manifest,a.out,a.aligner,a.aligner_symbols)
