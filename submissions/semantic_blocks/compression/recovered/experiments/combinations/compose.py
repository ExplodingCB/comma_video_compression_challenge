"""Losslessly compose model blocks with another candidate's exact learned bytes.

This repacks a candidate's exact model bytes and entropy tail. Optionally a
compatible entropy candidate supplies its complete tail and decoder. Hybrid
composition requires byte-identical HPAC models and does not inherit a score.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

ROOT=Path(__file__).resolve().parents[2]
LOSSLESS=ROOT/'experiments/lossless/candidate'
SUBMITTED=ROOT/'submissions/semantic_blocks'
OUTPUT=ROOT/'results/phase2/combinations'
sys.path.insert(0,str(ROOT/'experiments/lossless'/('deps' if sys.platform=='win32' else 'deps-linux')))
sys.path.insert(0,str(LOSSLESS))
from compress import build_payload,zip_payload
from runtime.block_container import decode_model_blocks
from lineage import write_lineage

def sha(data):return hashlib.sha256(data).hexdigest()

def inside(path):
    path=path.resolve()
    if path!=ROOT and ROOT not in path.parents:raise ValueError('All files must stay inside this project')
    return path

def fingerprint(runtime,archive,include_hpac=False):
    command=[sys.executable,str(Path(__file__).with_name('state_fingerprint.py')),
             '--runtime-source',str(runtime),'--archive',str(archive)]
    if include_hpac:command.append('--include-hpac-bytes')
    environment=os.environ.copy()
    dependency=ROOT/'experiments/lossless'/('deps' if sys.platform=='win32' else 'deps-linux')
    paths=[str(runtime/'.deps'),str(dependency)]
    if environment.get('PYTHONPATH'):paths.append(environment['PYTHONPATH'])
    environment['PYTHONPATH']=os.pathsep.join(paths)
    result=subprocess.run(command,check=True,capture_output=True,text=True,env=environment)
    return json.loads(result.stdout)


def unpack(path):
    with zipfile.ZipFile(path) as archive:
        if archive.namelist()!=['p']:raise ValueError('Expected exactly ZIP member p')
        return decode_model_blocks(archive.read('p'))


def source_runtime(source,requested=None):
    if requested:return inside(requested)
    return source.parent if (source.parent/'runtime/residual_archive.py').exists() else SUBMITTED


def compatible_runtimes(models_runtime,entropy_runtime):
    """All model interpretation/rendering code must agree, not just HPAC data."""
    def code(runtime):
        return {p.relative_to(runtime).as_posix():p.read_text()
                for top in ('runtime','cpr1') for p in (runtime/top).rglob('*')
                if p.is_file() and p.suffix in ('.py','.c') and '__pycache__' not in p.parts
                and p.relative_to(runtime).as_posix() not in
                ('runtime/block_container.py','runtime/residual_archive.py')}
    models=code(models_runtime);entropy=code(entropy_runtime)
    if models!=entropy:
        changed=[p for p in sorted(models.keys()|entropy.keys()) if models.get(p)!=entropy.get(p)]
        raise ValueError(f'Hybrid source runtime algorithms differ: {changed}')
    return sorted(models)


def hybrid_fingerprints(models,entropy):
    if models.get('hpac_bytes_hex') is None or models['hpac_bytes_hex']!=entropy.get('hpac_bytes_hex'):
        raise ValueError('Hybrid requires byte-identical HPAC model blobs')
    if models['hpac_blob']!=entropy['hpac_blob']:
        raise ValueError('HPAC fingerprint mismatch')
    result=copy.deepcopy(entropy)
    result.pop('hpac_bytes_hex')
    for key in ('semantic_blob','carrier_blob','renderer_tensors'):result[key]=copy.deepcopy(models[key])
    return result

def adapt_recipe(models,recipe):
    result=copy.deepcopy(recipe)
    # Preserve proven earlier boundaries, truncate blocks beyond a shorter input,
    # and make the final nonempty block end at the new model length.
    blocks=[]
    for block in result['blocks']:
        if block['start']>=len(models):break
        block['end']=min(block['end'],len(models))
        blocks.append(block)
    if not blocks:raise ValueError('Empty models')
    blocks[-1]['end']=len(models)
    for block in blocks:
        for key in ('stored_bytes','cost'):block.pop(key,None)
    result['blocks']=blocks
    return result

def refine(models,tail,recipe,radius):
    tried=0
    if not radius:return recipe,0
    for index in range(len(recipe['blocks'])-1):
        boundary=recipe['blocks'][index]['end']
        current=len(build_payload(models,tail,recipe))
        chosen=recipe
        low=recipe['blocks'][index]['start']+1
        high=recipe['blocks'][index+1]['end']-1
        for split in sorted(set([boundary]+list(range(max(low,boundary-radius),min(high,boundary+radius)+1,4)))):
            trial=copy.deepcopy(recipe)
            trial['blocks'][index]['end']=split
            trial['blocks'][index+1]['start']=split
            size=len(build_payload(models,tail,trial));tried+=1
            if size<current:chosen=trial;current=size
        recipe=chosen
    return recipe,tried

def block_dispatch(text):
    if "outer.startswith((b'BLK1', b'BLK2'))" in text:return text
    for old in ('outer.startswith(b"BLK1")',"outer.startswith(b'BLK1')"):
        if old in text:return text.replace(old,"outer.startswith((b'BLK1', b'BLK2'))",1)
    raise ValueError('Cannot find BLK1 format dispatch in source runtime')


def copy_decoder(runtime,target):
    inherited={}
    for top in ('runtime','cpr1'):
        for path in (runtime/top).rglob('*'):
            if path.is_file() and path.suffix in ('.py','.c') and '__pycache__' not in path.parts:
                rel=path.relative_to(runtime); dest=target/rel
                dest.parent.mkdir(parents=True,exist_ok=True)
                dest.write_bytes(path.read_bytes());inherited[str(rel)]=sha(path.read_bytes())
    for name in ('inflate.py','inflate.sh','LICENSE'):
        source=runtime/name
        if not source.exists():source=SUBMITTED/name
        (target/name).write_bytes(source.read_bytes())
    shutil.copyfile(LOSSLESS/'runtime/block_container.py',target/'runtime/block_container.py')
    reader=target/'runtime/residual_archive.py'
    reader.write_text(block_dispatch(reader.read_text()),newline='\n')
    # Runtime algorithms are inherited except container dispatch. The shell
    # bootstrap adds the general-purpose Brotli library in a contained target.
    shell=target/'inflate.sh';text=shell.read_text()
    if 'prepare_dependencies.sh' not in text:
        marker='DATA_DIR="$1"'
        if marker not in text:raise ValueError('Cannot locate decoder shell setup')
        text=text.replace(marker,'export PYTHONPATH="$HERE/.deps${PYTHONPATH:+:$PYTHONPATH}"\nif ! python -c "import brotli" >/dev/null 2>&1; then\n  bash "$HERE/prepare_dependencies.sh"\nfi\n'+marker,1)
    shell.write_text(text,newline='\n')
    shutil.copyfile(LOSSLESS/'prepare_dependencies.sh',target/'prepare_dependencies.sh')
    changed=[]
    for rel,digest in inherited.items():
        if sha((target/rel).read_bytes())!=digest:changed.append(rel.replace('\\','/'))
    allowed={'runtime/block_container.py','runtime/residual_archive.py'}
    if not set(changed)<=allowed:raise ValueError(f'Unexpected runtime changes: {changed}')
    return changed

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--name',required=True)
    ap.add_argument('--runtime-source',type=Path)
    ap.add_argument('--entropy-source',type=Path,help='Use this compatible candidate\'s complete entropy tail')
    ap.add_argument('--entropy-runtime-source',type=Path,help='Actual entropy decoder, default: its archive directory')
    ap.add_argument('--recipe',type=Path,default=LOSSLESS/'recipe.json')
    ap.add_argument('--refine-radius',type=int,default=0)
    args=ap.parse_args()
    if not args.name or any(x not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for x in args.name):
        raise ValueError('Use a simple unique candidate name')
    if not 0<=args.refine_radius<=128:raise ValueError('Refinement radius must be 0..128')
    if args.entropy_runtime_source and not args.entropy_source:
        raise ValueError('--entropy-runtime-source requires --entropy-source')
    source=inside(args.source);target=inside(OUTPUT/args.name)
    if target.exists():raise ValueError('Output already exists; choose another name')
    models_runtime=source_runtime(source,args.runtime_source)
    runtime=models_runtime
    started=time.monotonic();raw=source.read_bytes()
    models,tail=unpack(source)
    if runtime==SUBMITTED and (not models.startswith(b'F24S') or tail.startswith(b'ETP1')):
        raise ValueError('Input needs a custom reader; supply --runtime-source with its actual runtime')
    entropy_source=inside(args.entropy_source) if args.entropy_source else None
    before=fingerprint(models_runtime,source,include_hpac=bool(entropy_source))
    hybrid={}
    if entropy_source:
        entropy_raw=entropy_source.read_bytes()
        runtime=source_runtime(entropy_source,args.entropy_runtime_source)
        compatible=compatible_runtimes(models_runtime,runtime)
        _,tail=unpack(entropy_source)
        if runtime==SUBMITTED and tail.startswith(b'ETP1'):
            raise ValueError('Entropy source needs its actual ETP1 reader')
        entropy_before=fingerprint(runtime,entropy_source,include_hpac=True)
        before=hybrid_fingerprints(before,entropy_before)
        hybrid=dict(composition_kind='model_source_plus_entropy_source',
                    models_runtime_source=str(models_runtime),entropy_source=str(entropy_source),
                    entropy_runtime_source=str(runtime),entropy_source_archive_bytes=len(entropy_raw),
                    entropy_source_archive_sha256=sha(entropy_raw),hpac_blobs_byte_identical=True,
                    compatible_runtime_files=compatible,
                    entropy_causal_replay='Not verified by this CPU composition tool')
    recipe=adapt_recipe(models,json.loads(inside(args.recipe).read_text()))
    recipe,trials=refine(models,tail,recipe,args.refine_radius)
    packed=build_payload(models,tail,recipe);archive=zip_payload(packed)
    assert decode_model_blocks(packed)==(models,tail)
    assert archive==zip_payload(build_payload(models,tail,recipe))
    target.mkdir(parents=True)
    changed=copy_decoder(runtime,target)
    (target/'archive.zip').write_bytes(archive)
    (target/'recipe.json').write_text(json.dumps(recipe,indent=2)+'\n')
    after=fingerprint(target,target/'archive.zip')
    if before!=after:raise ValueError('Decoded learned states or entropy tail changed')
    if source.read_bytes()!=raw:raise ValueError('Input changed during composition; rerun with a stable source')
    if entropy_source and entropy_source.read_bytes()!=entropy_raw:
        raise ValueError('Entropy input changed during composition; rerun with a stable source')
    report=dict(status='exact_hybrid_composition_not_scored' if entropy_source else 'exact_storage_composition_not_scored',source=str(source),runtime_source=str(runtime),
                source_archive_bytes=len(raw),archive_bytes=len(archive),saved_vs_input=len(raw)-len(archive),
                source_archive_sha256=sha(raw),archive_sha256=sha(archive),
                models_bytes=len(models),models_sha256=sha(models),tail_bytes=len(tail),tail_sha256=sha(tail),
                byte_roundtrip_exact=True,decoded_states_equal=True,decoded_state_fingerprints=after,
                runtime_files_changed=changed,refinement_trials=trials,elapsed_seconds=time.monotonic()-started,
                pixels_and_score='Not evaluated by this composition tool',
                recipe=str(target/'recipe.json'),decoder_dependency='brotli==1.2.0')
    report.update(hybrid)
    (target/'provenance.json').write_text(json.dumps(report,indent=2)+'\n')
    write_lineage(target,report)
    print(json.dumps({k:v for k,v in report.items() if k!='decoded_state_fingerprints'},indent=2))

if __name__=='__main__':main()
