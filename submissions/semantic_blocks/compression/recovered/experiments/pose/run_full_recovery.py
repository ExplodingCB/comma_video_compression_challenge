"""Run and resume full-600 recovery using previously validated local caches.

Outputs are research archives, not leaderboard submissions. The full decoder
and evaluator must still validate a promoted archive.
"""
from common import *
import argparse
import rescue


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', choices=('control', 'all_metadata_b2', 'atom0-linear', 'all-int4'), required=True)
    parser.add_argument('--passes', type=int, default=5)
    parser.add_argument('--coordinate', action='store_true')
    args = parser.parse_args()
    manifest = json.loads((OUT/'cache/manifest.json').read_text())
    for name, expected in manifest['files'].items():
        path = OUT/'cache'/(name+'.npy')
        assert file_sha(path) == expected['sha256'], f'Invalid cached target: {path}'
    name = args.case
    preferred = OUT/('refined-'+name)
    candidate = preferred if (preferred/'archive.zip').exists() else OUT/('rescue-'+name)
    if not candidate.exists():
        candidate = OUT/name
    report = candidate/'report.json'
    if report.exists():
        metadata = json.loads(report.read_text())
        assert sha((candidate/'archive.zip').read_bytes()) == metadata['history'][-1]['archive_sha256']
    decoded = context(candidate/'archive.zip')
    assert np.array_equal(decoded['codes'], np.load(candidate/'codes.npy', allow_pickle=False))
    assert decoded['canonical'] == (candidate/'canonical.bin').read_bytes()
    output = OUT/('full-'+name)
    extra = []
    if name == 'all_metadata_b2':
        meta = ROOT/'results/phase2/renderer'
        extra = ['--source-archive', str(meta/'archives/all_metadata_b2/archive.zip'),
                 '--master-frames', str(meta/'masters/all_metadata_b2/masters.uint8'),
                 '--seg-errors', str(OUT/'joint-cache/all_metadata_b2-seg.npy'),
                 '--warm-start', str(candidate/'codes.npy')]
    if args.coordinate:
        extra.append('--coordinate')
    sys.argv = ['rescue.py', '--candidate', str(candidate), '--output', str(output),
                '--pairs', '600', '--passes', str(args.passes), '--batch', '8',
                '--neighbours', '2', '--resume', *extra]
    rescue.main()
    from audit_completed import audit
    print(json.dumps(audit(output),indent=2),flush=True)


if __name__ == '__main__':
    main()
