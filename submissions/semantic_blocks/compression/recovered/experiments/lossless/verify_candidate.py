"""Verify archive determinism, exact model/tail recovery and format rejection."""
from pathlib import Path
import hashlib, io, json, random, sys, zipfile
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE/('deps' if sys.platform=='win32' else 'deps-linux')))
sys.path.insert(0,str(HERE/'candidate'))
from runtime.block_container import decode_model_blocks, transpose
from compress import build, build_payload, zip_payload

def payload(path):
    with zipfile.ZipFile(path) as z:
        assert z.namelist()==['p']
        return z.read('p')

def main():
    candidate=HERE/'candidate'
    packed=payload(candidate/'archive.zip')
    models,tail=decode_model_blocks(packed)
    assert models==(ROOT/'results/lossless/models.bin').read_bytes()
    assert tail==(ROOT/'results/lossless/tail.bin').read_bytes()
    assert decode_model_blocks(payload(ROOT/'submissions/semantic_blocks/archive.zip'))==(models,tail)
    recipe=json.loads((candidate/'recipe.json').read_text())
    encoded=build((ROOT/'artifacts/pr135/archive.zip').read_bytes(),recipe)
    assert encoded==(candidate/'archive.zip').read_bytes()
    assert zip_payload(build_payload(models,tail,recipe))==encoded
    rng=random.Random(12345)
    for stride in (1,2,3,4,12,255):
        for size in (0,1,2,3,7,8,15,16,31,32,255,256,1025):
            data=rng.randbytes(size)
            assert transpose(transpose(data,stride),stride,True)==data
    malformed=[b'',b'BLK2',packed[:8],packed[:10],packed[:-len(tail)],
               b'NOPE'+packed[4:],packed[:4]+bytes(4)+packed[8:],
               packed[:9]+b'\x7f'+packed[10:]]
    for bad in malformed:
        try: decode_model_blocks(bad)
        except (ValueError, RuntimeError): pass
        else: raise AssertionError('Malformed container accepted')
    report=dict(platform=sys.platform,archive_bytes=len(encoded),
                archive_sha256=hashlib.sha256(encoded).hexdigest(),
                models_sha256=hashlib.sha256(models).hexdigest(),
                tail_sha256=hashlib.sha256(tail).hexdigest(),
                exact_roundtrip=True,baseline_models_tail_equal=True,
                reproducible_archive=True,transpose_cases=78,rejected_malformed_cases=len(malformed))
    target=ROOT/'results/phase2/lossless'/f'verify-{sys.platform}.json'
    target.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
