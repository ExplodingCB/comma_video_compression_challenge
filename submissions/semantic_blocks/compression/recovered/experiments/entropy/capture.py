"""Decode the submitted archive once and cache rows for CPU entropy experiments."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

from cache_hook import Capture

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'submissions' / 'semantic_blocks'
OUT = ROOT / 'results' / 'phase2' / 'entropy' / 'cache'
sys.path.insert(0, str(SOURCE))
from runtime import residual_archive
from runtime.f26_inflate import _load_renderer


def main():
    if (OUT / 'manifest.json').exists():
        print('Completed cache already exists; refusing to replace it.')
        return
    OUT.mkdir(parents=True, exist_ok=True)
    os.environ['CPR1_RC64_LIBRARY'] = str(ROOT / 'artifacts' / 'pr135' / 'rc64_backend.so')
    residual_archive.configure_cuda_reproducibility()
    parts = residual_archive.read_residual_archive(SOURCE / 'archive.zip')
    renderer = _load_renderer(SOURCE / 'cpr1')
    capture = Capture(residual_archive, OUT)
    started = time.monotonic()
    tokens, report = residual_archive.decode_production_tokens(parts, renderer, SOURCE / 'cpr1', torch.device('cuda'))
    token_bytes = tokens.numpy().astype(np.uint8).tobytes()
    digest = hashlib.sha256(token_bytes).hexdigest()
    if digest != 'c5c7671d037b6912980c57929a5b6d789d250ee6a93e3b0a6018cf9f63e32ece':
        raise RuntimeError('Decoded tokens differ from baseline')
    (OUT / 'tokens.uint8').write_bytes(token_bytes)
    manifest = capture.finish()
    manifest.update(token_sha256=digest, decode_report=report, seconds=time.monotonic()-started)
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
