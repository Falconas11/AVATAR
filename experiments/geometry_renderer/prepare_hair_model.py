"""Explicit setup command for the optional offline face parser; never used by render.

Downloads only the pinned upstream GitHub asset and verifies its published SHA-256
before atomically installing it. No photo is accepted, logged, written, or uploaded.
"""
from __future__ import annotations
import argparse
from hashlib import sha256
from pathlib import Path
import tempfile
import urllib.request
from .hair_layers import MODEL_FILENAME,MODEL_SHA256,MODEL_URL


def ensure_model(destination: Path | None = None) -> Path:
    destination=Path(destination) if destination is not None else Path.home()/'.cache/avatar'/MODEL_FILENAME
    if destination.is_file():
        if sha256(destination.read_bytes()).hexdigest()!=MODEL_SHA256:
            raise ValueError('Existing parser file does not match the pinned SHA-256; it was not overwritten')
        return destination
    destination.parent.mkdir(parents=True,exist_ok=True)
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent,suffix='.partial',delete=False) as output:
            temporary=Path(output.name)
            digest=sha256()
            with urllib.request.urlopen(MODEL_URL,timeout=60) as response:
                total=0
                while chunk:=response.read(1024*1024):
                    total+=len(chunk)
                    if total>60*1024*1024:
                        raise ValueError('Parser asset exceeds the expected size limit')
                    digest.update(chunk);output.write(chunk)
        if digest.hexdigest()!=MODEL_SHA256:
            raise ValueError('Downloaded parser failed the published SHA-256 verification')
        temporary.replace(destination)
        return destination
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=None,help='Local model cache destination')
    args=parser.parse_args()
    try:
        ensure_model(args.output)
    except Exception as exc:
        parser.exit(1,f'Face parser setup failed ({type(exc).__name__}). Existing files were preserved; verify permitted GitHub access and SHA-256.\n')
    print('Verified offline face parser model is installed. No runtime download is needed.')


if __name__=='__main__':
    main()
