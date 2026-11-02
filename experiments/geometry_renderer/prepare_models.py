"""Explicit one-time download of pinned local model assets. Never accepts photos."""
from __future__ import annotations
import argparse
from pathlib import Path
from .prepare_hair_model import ensure_model
from .prepare_conditional_model import prepare


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-dir',type=Path,default=Path(__file__).parent/'.models')
    args=parser.parse_args()
    ensure_model(args.cache_dir/'face_parsing_resnet18.onnx')
    prepare(args.cache_dir/'fran_reaging.onnx')
    print('Both local models passed their pinned SHA-256 checks. Runtime has no downloads or photo uploads.')

if __name__=='__main__':main()
