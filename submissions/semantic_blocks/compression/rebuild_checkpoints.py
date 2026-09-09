"""Replay saved state using the recovered CAP1 serializer and RC64 encoder.

This file is copied to the root of a freshly prepared experiment workbench.
The original experiment algorithms are imported, not reimplemented here.
"""
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'experiments/pose'))
import common


def run(*args):
    subprocess.run(args, check=True, cwd=ROOT)


def main():
    import numpy as np
    pose = ROOT / 'results/phase2/pose/full-control/pass3.zip'
    entropy = ROOT / 'results/phase2/entropy'
    if pose.exists() or (entropy / 'encoded.json').exists():
        raise ValueError('Checkpoint replay requires fresh outputs')
    ctx = common.context()
    codes = np.load(ROOT / 'checkpoints/pass3-codes.npy', allow_pickle=False)
    if codes.shape != (600, 12) or not np.issubdtype(codes.dtype, np.integer) or codes.min() < -2048 or codes.max() > 2047:
        raise ValueError('Expected 600 x 12 signed-int12 coefficient values')
    # audit_completed.py already established that the baseline basis is sufficient;
    # no recovered canonical carrier needs to be hidden in this source package.
    archive, _ = common.archive(ctx, ctx['canonical'], codes)
    expected = '0afd7f640ebdaac9826fc2c2f4f88a4c4ef3efa2297e35679bda501bbb49e3e6'
    if hashlib.sha256(archive).hexdigest() != expected:
        raise ValueError('Saved pose coefficients did not reproduce the immutable pass3 archive')
    pose.parent.mkdir(parents=True, exist_ok=True)
    with pose.open('xb') as stream:
        stream.write(archive)
    print('Pose checkpoint reserialized: 186152 bytes, exact pass3 SHA-256', flush=True)
    compiler = shlex.split(os.environ.get('CC', 'cc'))
    run(*compiler, '-O3', '-shared', '-fPIC',
        str(ROOT / 'references/pr135-experiments/src/cpr1_sub4/entropy/rc64_backend.c'),
        '-o', str(entropy / 'librc64.so'), '-lm')
    run(sys.executable, str(ROOT / 'experiments/entropy/replay_baseline.py'))
    # Retain the baseline and the selected calibrated row from the saved screen.
    # encode.py performs actual coding and cached-symbol replay for both.
    screen = json.loads((ROOT / 'checkpoints/screen.json').read_text())
    table = json.loads((ROOT / 'checkpoints/table.json').read_text())
    winner = next(r for r in screen['finalists'] if r['name'] == 'margin50_4_raw_1')
    if winner != table:
        raise ValueError('Saved finalist differs from its table checkpoint')
    screen['finalists'] = [r for r in screen['finalists'] if r['name'] in ('baseline', 'margin50_4_raw_1')]
    (entropy / 'screen.json').write_text(json.dumps(screen)+'\n')
    run(sys.executable, str(ROOT / 'experiments/entropy/encode.py'))
    candidate = entropy / 'candidates/margin50_4_raw_1/archive.zip'
    if common.file_sha(candidate) != '9d0a14db2620d788a6a5007dd4b5e09db9610f996710e26f9a6062760a1e3db4':
        raise ValueError('Re-encoded entropy archive differs from the target input')
    print('Entropy table and 117964800 symbols re-encoded: exact input SHA-256', flush=True)


if __name__ == '__main__':
    main()
