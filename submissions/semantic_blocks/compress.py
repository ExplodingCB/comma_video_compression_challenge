#!/usr/bin/env python3
"""Rebuild semantic_blocks v2 with the recovered phase-2 implementation.

compose consumes two intermediate archives; checkpoints also serializes the
saved pose coefficients and re-encodes the tokens from captured predictor rows.
Neither mode trains the inherited PR135 models from the original video.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import zipfile

from compression.blk2 import build_payload, zip_payload
from runtime.block_container import decode_model_blocks

HERE = Path(__file__).resolve().parent
TARGET_SHA256 = '640e9a8d618652ce4e0570cbcc92348d3f2457b8b737a13ab68cdbb9595fcf2f'
MODEL_SHA256 = '0afd7f640ebdaac9826fc2c2f4f88a4c4ef3efa2297e35679bda501bbb49e3e6'
ENTROPY_SHA256 = '9d0a14db2620d788a6a5007dd4b5e09db9610f996710e26f9a6062760a1e3db4'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_source(path, expected):
    data = path.read_bytes()
    if sha(data) != expected:
        raise ValueError(f'Wrong input {path}: expected SHA-256 {expected}, got {sha(data)}')
    with zipfile.ZipFile(path) as archive:
        if archive.namelist() != ['p']:
            raise ValueError('Expected exactly one ZIP member named p')
        models, tail = decode_model_blocks(archive.read('p'))
    return data, models, tail


def compose(models_path, entropy_path, output):
    import brotli
    output = output.resolve()
    report_path = output.with_name(output.name + '.rebuild.json')
    if output == HERE / 'archive.zip' or output.exists() or report_path.exists():
        raise ValueError('Choose a new output path; published and existing outputs are never overwritten')
    source, models, _ = read_source(models_path, MODEL_SHA256)
    entropy_source, entropy_models, tail = read_source(entropy_path, ENTROPY_SHA256)
    # The pinned F24S schema stores the entire IHS2 HPAC body in this span.
    if models[:16597] != entropy_models[:16597] or models[:4] != b'F24S':
        raise ValueError('Sources must carry byte-identical HPAC model data')
    recipe = json.loads((HERE / 'recipe.json').read_text())
    payload = build_payload(models, tail, recipe)
    result = zip_payload(payload)
    matched = len(result) == 185704 and sha(result) == TARGET_SHA256
    report = dict(
        status='matched_published_target' if matched else 'TARGET_MISMATCH',
        measured_at=datetime.now(timezone.utc).isoformat(),
        operation='composition_from_saved_intermediate_archives',
        archive_bytes=len(result), archive_sha256=sha(result),
        expected_bytes=185704, expected_sha256=TARGET_SHA256,
        model_source=str(models_path.resolve()), model_source_sha256=sha(source),
        entropy_source=str(entropy_path.resolve()), entropy_source_sha256=sha(entropy_source),
        models_bytes=len(models), models_sha256=sha(models),
        tail_bytes=len(tail), tail_sha256=sha(tail), hpac_bytes_identical=True,
        model_and_tail_roundtrip_exact=decode_model_blocks(payload) == (models, tail),
        recipe_sha256=sha((HERE / 'recipe.json').read_bytes()),
        python=sys.version, platform=platform.platform(), brotli=brotli.__version__,
        optimization_rerun=False, causal_inflation='not_run', evaluation='not_run')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as stream:
        stream.write(result)
    with report_path.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps(report, indent=2), flush=True)
    if not matched:
        raise ValueError(f'Rebuilt bytes differ from target; retained {output} and {report_path} for investigation')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest='mode', required=True)
    direct = modes.add_parser('compose', help='CPU BLK2 packing of the two saved input archives')
    direct.add_argument('--model-source', type=Path, required=True)
    direct.add_argument('--entropy-source', type=Path, required=True)
    direct.add_argument('--output', type=Path, required=True)
    saved = modes.add_parser('checkpoints', help='Rebuild pose and entropy archives before BLK2 packing')
    saved.add_argument('--inputs', type=Path, required=True, help='Recovered input bundle (see inputs.json)')
    saved.add_argument('--reference-source', type=Path, required=True, help='Pinned PR135 ExperimentBook checkout')
    saved.add_argument('--work-dir', type=Path, required=True, help='New directory; must not already exist')
    args = parser.parse_args()
    if args.mode == 'compose':
        compose(args.model_source, args.entropy_source, args.output)
    else:
        from compression.prepare_workbench import prepare
        work = prepare(args.work_dir, args.reference_source, args.inputs)
        subprocess.run([sys.executable, str(work / 'rebuild_checkpoints.py')], check=True, cwd=work)
        compose(work / 'results/phase2/pose/full-control/pass3.zip',
                work / 'results/phase2/entropy/candidates/margin50_4_raw_1/archive.zip',
                work / 'rebuilt/archive.zip')


if __name__ == '__main__':
    main()
