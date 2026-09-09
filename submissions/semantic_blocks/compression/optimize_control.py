"""Run just the historical winning control schedule in a prepared workbench.

Requires the original video/evaluator pose cache and the baseline inflated video.
Other rejected cases from run_joint_screen.py are deliberately not launched.
"""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'experiments/pose'))
import common


def main():
    import numpy as np
    out = common.OUT
    if any((out / name).exists() for name in ('control', 'rescue-control', 'full-control')):
        raise ValueError('Use a fresh workbench for the original optimization schedule')
    ctx = common.context()
    control = out / 'control'
    control.mkdir(parents=True)
    (control / 'archive.zip').write_bytes(ctx['data'])
    (control / 'canonical.bin').write_bytes(ctx['canonical'])
    np.save(control / 'codes.npy', ctx['codes'], allow_pickle=False)
    # Original run_joint_screen.py control: 3 passes, 96 pairs, batch 8.
    subprocess.run([sys.executable, str(ROOT / 'experiments/pose/rescue.py'),
                    '--candidate', str(control), '--output', str(out / 'rescue-control'),
                    '--pairs', '96', '--passes', '3', '--batch', '8'], check=True)
    # Original full recovery: 3 passes, 600 pairs, 2 neighbour dimensions, no coordinate sweep.
    subprocess.run([sys.executable, str(ROOT / 'experiments/pose/run_full_recovery.py'),
                    '--case', 'control', '--passes', '3'], check=True)


if __name__ == '__main__':
    main()
