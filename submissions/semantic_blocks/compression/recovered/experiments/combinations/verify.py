"""Audit saved compositions after interruption without modifying their archives."""
import argparse
import json
from pathlib import Path
import platform
import sys
import zipfile

from compose import ROOT, OUTPUT, LOSSLESS, block_dispatch, build_payload, compatible_runtimes, decode_model_blocks, fingerprint, hybrid_fingerprints, inside, sha, zip_payload


def project_path(value):
    """Resolve provenance from either Windows or WSL inside this one project."""
    value=str(value).replace('\\','/')
    for prefix in ('D:/Projects/video-compression/', '/mnt/d/Projects/video-compression/'):
        if value.startswith(prefix):return inside(ROOT/value[len(prefix):])
    return inside(Path(value))


def unpack(path):
    with zipfile.ZipFile(path) as archive:
        if archive.namelist()!=['p']:raise ValueError('Expected one ZIP member p')
        return decode_model_blocks(archive.read('p'))


def audit(target,container_only):
    target=inside(target)
    report=json.loads((target/'provenance.json').read_text())
    source=project_path(report['source'])
    runtime=project_path(report['runtime_source'])
    models_runtime=project_path(report.get('models_runtime_source',report['runtime_source']))
    actual=(target/'archive.zip').read_bytes()
    original=source.read_bytes()
    if sha(actual)!=report['archive_sha256'] or len(actual)!=report['archive_bytes']:
        raise ValueError(f'{target.name}: saved output archive changed')
    if sha(original)!=report['source_archive_sha256'] or len(original)!=report['source_archive_bytes']:
        raise ValueError(f'{target.name}: saved input archive changed')
    models,tail=unpack(source)
    entropy_source=None
    if report.get('entropy_source'):
        entropy_source=project_path(report['entropy_source'])
        entropy_raw=entropy_source.read_bytes()
        if sha(entropy_raw)!=report['entropy_source_archive_sha256'] or len(entropy_raw)!=report['entropy_source_archive_bytes']:
            raise ValueError('Entropy input changed since composition')
        if runtime!=project_path(report['entropy_runtime_source']):
            raise ValueError('Hybrid must inherit the entropy reader')
        if compatible_runtimes(models_runtime,runtime)!=report['compatible_runtime_files']:
            raise ValueError('Hybrid runtime compatibility audit differs')
        _,tail=unpack(entropy_source)
    if unpack(target/'archive.zip')!=(models,tail):
        raise ValueError('Recovered models/tail differ from input')
    if sha(models)!=report['models_sha256'] or sha(tail)!=report['tail_sha256']:
        raise ValueError('Recovered model/tail hashes differ from provenance')
    recipe=json.loads((target/'recipe.json').read_text())
    rebuilt=zip_payload(build_payload(models,tail,recipe))
    if rebuilt!=actual:raise ValueError('Archive rebuild differs byte-for-byte')
    unchanged=[]
    allowed={'runtime/block_container.py','runtime/residual_archive.py'}
    changed=[]
    for top in ('runtime','cpr1'):
        for path in (runtime/top).rglob('*'):
            if path.is_file() and path.suffix in ('.py','.c') and '__pycache__' not in path.parts:
                relative=path.relative_to(runtime)
                if path.read_bytes()==(target/relative).read_bytes():unchanged.append(relative.as_posix())
                else:changed.append(relative.as_posix())
    if not set(changed)<=allowed:raise ValueError(f'Unexpected runtime change: {changed}')
    if set(changed)!=set(report['runtime_files_changed']):
        raise ValueError('Changed runtime file list differs from provenance')
    if (target/'runtime/block_container.py').read_bytes()!=(LOSSLESS/'runtime/block_container.py').read_bytes():
        raise ValueError('Unexpected lossless block decoder')
    if (target/'runtime/residual_archive.py').read_text()!=block_dispatch((runtime/'runtime/residual_archive.py').read_text()):
        raise ValueError('Source residual reader changed beyond block format dispatch')
    if not container_only:
        before=fingerprint(models_runtime,source,include_hpac=bool(entropy_source))
        if entropy_source:
            before=hybrid_fingerprints(before,fingerprint(runtime,entropy_source,include_hpac=True))
        after=fingerprint(target,target/'archive.zip')
        if before!=after or after!=report['decoded_state_fingerprints']:
            raise ValueError('Decoded states differ from input or saved provenance')
    return dict(candidate=target.name,archive_bytes=len(actual),archive_sha256=sha(actual),
                saved_vs_source=len(original)-len(actual),rebuild_identical=True,
                model_and_tail_identical=True,decoded_states_verified=not container_only,
                entropy_source=str(entropy_source) if entropy_source else None,
                hpac_byte_equality_verified=bool(entropy_source) and not container_only,
                source_reader_preserved_except_dispatch=True,
                inherited_runtime_files_unchanged=len(unchanged),allowed_runtime_files_changed=changed)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate',type=Path,action='append',required=True)
    parser.add_argument('--container-only',action='store_true',help='Skip Torch-dependent decoded-state readback')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=dict(status='passed_container_audit' if args.container_only else 'passed_full_cpu_audit',
                platform=platform.platform(),python=sys.version.split()[0],
                candidates=[audit(path,args.container_only) for path in args.candidate])
    output=inside(args.output)
    if OUTPUT not in output.parents:raise ValueError('Audit output must be inside results/phase2/combinations')
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
