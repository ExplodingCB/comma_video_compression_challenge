"""Restore the original relative layout in a NEW directory, without editing recovered code."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

PACKAGE = Path(__file__).resolve().parent
SUBMISSION = PACKAGE.parent


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def checked_copy(source, destination, expected=None):
    if expected and digest(source) != expected:
        raise ValueError(f'Input hash mismatch: {source}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def prepare(work, reference, inputs=None):
    work, reference = work.resolve(), reference.resolve()
    if work.exists():
        raise ValueError('Work directory must be new; existing experiments are never resumed implicitly')
    # Hash the files actually imported; accepts both a checkout and a source export.
    ref_manifest = json.loads((PACKAGE / 'reference-source.json').read_text())
    for name, expected in ref_manifest['files'].items():
        if digest(reference / name) != expected:
            raise ValueError(f'Expected pinned ExperimentBook source: {name}')
    manifest = json.loads((PACKAGE / 'inputs.json').read_text())
    if inputs:
        inputs = inputs.resolve()
        for record in manifest['files']:
            if record.get('stage') and digest(inputs / record['path']) != record['sha256']:
                raise ValueError(f'Wrong checkpoint or cache: {record["path"]}')
    work.mkdir(parents=True)
    shutil.copytree(PACKAGE / 'recovered/experiments', work / 'experiments')
    shutil.copytree(PACKAGE / 'recovered/scripts', work / 'scripts')
    for name, expected in ref_manifest['files'].items():
        checked_copy(reference / name, work / 'references/pr135-experiments' / name, expected)
    source = work / 'submissions/semantic_blocks'
    for top in ('runtime', 'cpr1'):
        shutil.copytree(SUBMISSION / top, source / top, ignore=shutil.ignore_patterns('__pycache__', '*.so', '*.pyc'))
    for name in ('inflate.py', 'inflate.sh', 'LICENSE', 'prepare_dependencies.sh'):
        checked_copy(SUBMISSION / name, source / name)
    # Only these v1 files differ from the deployed v2 model interpreter.
    shutil.copytree(PACKAGE / 'baseline', source, dirs_exist_ok=True)
    checked_copy(SUBMISSION / 'compress.sh', source / 'compress.sh')
    # The baseline compressor has its original flag interface (no subcommands).
    (source / 'compress.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\nHERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\npython "$HERE/compress.py" "$@"\n')
    checked_copy(PACKAGE / 'baseline-source-audit.json', work / 'results/source-audit.json')
    lossless = work / 'experiments/lossless/candidate'
    for top in ('runtime', 'cpr1'):
        shutil.copytree(SUBMISSION / top, lossless / top, ignore=shutil.ignore_patterns('__pycache__', '*.so', '*.pyc'))
    for name in ('inflate.py', 'inflate.sh', 'LICENSE', 'prepare_dependencies.sh'):
        checked_copy(SUBMISSION / name, lossless / name)
    checked_copy(PACKAGE / 'blk2.py', lossless / 'compress.py')
    checked_copy(PACKAGE / 'lossless-recipe.json', lossless / 'recipe.json')
    checked_copy(PACKAGE / 'baseline/LINEAGE.md', lossless / 'LINEAGE.md')
    checked_copy(source / 'compress.sh', lossless / 'compress.sh')
    for folder in ('results/lossless', 'results/phase2/lossless', 'results/phase2/pose',
                   'results/phase2/entropy', 'results/phase2/combinations', 'artifacts/pr135'):
        (work / folder).mkdir(parents=True, exist_ok=True)
    if inputs:
        for record in manifest['files']:
            if record.get('stage'):
                checked_copy(inputs / record['path'], work / record['stage'], record['sha256'])
    checked_copy(PACKAGE / 'rebuild_checkpoints.py', work / 'rebuild_checkpoints.py')
    checked_copy(PACKAGE / 'optimize_control.py', work / 'optimize_control.py')
    (work / 'workbench.json').write_text(json.dumps(dict(
        inputs=str(inputs) if inputs else None, reference=str(reference),
        reference_commit=ref_manifest['commit'],
        scope='Recovered phase-2 code and checkpoint replay, not upstream training'), indent=2)+'\n')
    return work


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--reference-source', type=Path, required=True)
    parser.add_argument('--inputs', type=Path)
    args = parser.parse_args()
    print(prepare(args.work_dir, args.reference_source, args.inputs))
