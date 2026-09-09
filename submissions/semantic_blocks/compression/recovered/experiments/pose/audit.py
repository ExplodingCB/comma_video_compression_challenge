"""CPU-only evidence audit: zip integrity, immutable tokens and model prefixes."""
from pathlib import Path
import hashlib,importlib.util,json,zipfile
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/phase2/pose'
spec=importlib.util.spec_from_file_location('pose_blocks',ROOT/'submissions/semantic_blocks/runtime/block_container.py')
blocks=importlib.util.module_from_spec(spec);spec.loader.exec_module(blocks)
def unpack(path):
    with zipfile.ZipFile(path) as z:
        assert z.namelist()==['p'] and z.testzip() is None
        return blocks.decode_model_blocks(z.read('p'))
def main():
    baseline,tail=unpack(ROOT/'submissions/semantic_blocks/archive.zip');results=[]
    for path in sorted(OUT.glob('**/archive.zip')):
        models,newtail=unpack(path);assert newtail==tail,'semantic token/residual bytes changed'
        if 'all_metadata_b2' in path.parent.name:
            reference,_=unpack(ROOT/'results/phase2/renderer/archives/all_metadata_b2/archive.zip')
        else:reference=baseline
        assert models[:52637]==reference[:52637],'pose experiment changed renderer or HPAC'
        cap=models[52637:]
        size=6+36+96+32+12+(int.from_bytes(cap[:3],'little')+7)//8+(int.from_bytes(cap[3:6],'little')+7)//8
        before=reference[52637:]
        oldsize=6+36+96+32+12+(int.from_bytes(before[:3],'little')+7)//8+(int.from_bytes(before[3:6],'little')+7)//8
        assert cap[size:]==before[oldsize:],'stored selector changed'
        results.append(dict(path=str(path.relative_to(ROOT)),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    report=dict(status='passed_component_integrity',count=len(results),archives=results)
    (OUT/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(f'Passed {len(results)} complete archive component audits.')
if __name__=='__main__':main()
