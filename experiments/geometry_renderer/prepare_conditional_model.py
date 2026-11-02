"""Explicit download/verification of the optional local FRAN-style appearance model.

No photo is involved. The published MIT-model ONNX was reproducibly converted by
FaceStudio's public CI from timroelofs123's independent FRAN reimplementation.
No GPL FaceStudio application code is copied or used by this experiment.
"""
from __future__ import annotations
import argparse
from hashlib import sha256
from pathlib import Path
import os
import tempfile
import urllib.request

MODEL_URL='https://github.com/georgegach/facestudio/releases/download/models/fran.onnx'
MODEL_SHA256='5941608e21f5d218ec52862938898efb276b63048aa8586372ff046ef865e22b'
MODEL_BYTES=124230737


def prepare(output: Path) -> Path:
    output=Path(output).expanduser()
    if output.is_file() and output.stat().st_size==MODEL_BYTES and sha256(output.read_bytes()).hexdigest()==MODEL_SHA256:
        return output
    if output.exists():
        raise ValueError('Existing model does not match the pinned digest; it was not overwritten')
    output.parent.mkdir(parents=True,exist_ok=True)
    # HTTPS verification is left enabled. Stream and bound the official asset;
    # never publish or use a partial/unverified model after a failed download.
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(prefix='avatar-fran-',suffix='.partial',dir=output.parent,delete=False) as file:
            temporary=Path(file.name)
            digest=sha256();total=0
            with urllib.request.urlopen(MODEL_URL,timeout=120) as response:
                while True:
                    chunk=response.read(1024*1024)
                    if not chunk:break
                    total+=len(chunk)
                    if total>MODEL_BYTES:
                        raise RuntimeError('Optional skin model exceeded the published asset size')
                    file.write(chunk);digest.update(chunk)
            if total!=MODEL_BYTES or digest.hexdigest()!=MODEL_SHA256:
                raise RuntimeError('Optional skin model failed the published size/SHA-256 check')
        os.replace(temporary,output);temporary=None
    finally:
        if temporary is not None:temporary.unlink(missing_ok=True)
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path.home()/'.cache/avatar/fran_reaging.onnx')
    args=parser.parse_args()
    print(f'Verified optional local skin model prepared: {prepare(args.output)}')

if __name__=='__main__':main()
