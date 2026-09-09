"""Build an isolated candidate without changing reviewed/public submission files."""
from pathlib import Path
import hashlib, importlib.util, json, shutil, sys
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'deps'))
candidate=HERE/'candidate'
source=ROOT/'submissions/semantic_blocks'
for path in source.rglob('*'):
    rel=path.relative_to(source)
    if any(part in ('archive','inflated','__pycache__') for part in rel.parts): continue
    if path.is_file() and (path.suffix in ('.py','.sh','.c') or str(rel) in ('LICENSE','LINEAGE.md')):
        dest=candidate/rel; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(path,dest)
shutil.copy2(HERE/'block_container.py',candidate/'runtime/block_container.py')
shutil.copy2(HERE/'compress.py',candidate/'compress.py')
residual=candidate/'runtime/residual_archive.py'
text=residual.read_text()
old="outer.startswith(b'BLK1')"
if old not in text:
    old='outer.startswith(b"BLK1")'
if old not in text:
    raise ValueError('Could not locate model format detection')
text=text.replace(old,"outer.startswith((b'BLK1', b'BLK2'))")
residual.write_text(text,newline='\n')
report=json.loads((ROOT/'results/phase2/lossless/refine-report.json').read_text())
(candidate/'recipe.json').write_text(json.dumps(dict(blocks=report['blocks']),indent=2)+'\n')
lineage=candidate/'LINEAGE.md'
lineage.write_text(lineage.read_text()+'\nThis experimental BLK2/Brotli candidate is a subsequent lossless storage experiment.\nIt is not the archive submitted in PR #141. Its model and tail bytes are identical\nto the inherited baseline; the container, codec choices and byte-lane layout differ.\n',newline='\n')
inflate=candidate/'inflate.sh'
text=inflate.read_text().replace('DATA_DIR="$1"','export PYTHONPATH="$HERE/.deps${PYTHONPATH:+:$PYTHONPATH}"\nif ! python -c "import brotli" >/dev/null 2>&1; then\n  bash "$HERE/prepare_dependencies.sh"\nfi\nDATA_DIR="$1"')
inflate.write_text(text,newline='\n')
compress_shell=candidate/'compress.sh'
text=compress_shell.read_text().replace('python "$HERE/compress.py"',
    'export PYTHONPATH="$HERE/.deps${PYTHONPATH:+:$PYTHONPATH}"\nif ! python -c "import brotli" >/dev/null 2>&1; then\n  bash "$HERE/prepare_dependencies.sh"\nfi\npython "$HERE/compress.py"')
compress_shell.write_text(text,newline='\n')
(candidate/'prepare_dependencies.sh').write_text('''#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$HERE/.deps" "$HERE/.cache" "$HERE/.tmp"
export TMPDIR="$HERE/.tmp" TMP="$HERE/.tmp" TEMP="$HERE/.tmp"
export UV_CACHE_DIR="$HERE/.cache/uv" PIP_CACHE_DIR="$HERE/.cache/pip"
if command -v uv >/dev/null 2>&1; then
  uv pip install --python "$(command -v python)" --target "$HERE/.deps" brotli==1.2.0
else
  python -m pip install --target "$HERE/.deps" brotli==1.2.0
fi
''',newline='\n')
sys.path.insert(0,str(candidate))
spec=importlib.util.spec_from_file_location('candidate_compress',candidate/'compress.py')
mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
raw=(ROOT/'artifacts/pr135/archive.zip').read_bytes()
recipe=json.loads((candidate/'recipe.json').read_text())
archive=mod.build(raw,recipe); assert archive==mod.build(raw,recipe)
assert len(archive)==report['archive_bytes']
(candidate/'archive.zip').write_bytes(archive)
audit=dict(archive_bytes=len(archive),saved_vs_current=186151-len(archive),
           archive_sha256=hashlib.sha256(archive).hexdigest(),
           predicted_local_score=0.1618888066881454+25*(len(archive)-186151)/37545489,
           distortion_status='Unchanged model and tail bytes, full GPU evaluation not yet repeated',
           decoder_dependency='brotli==1.2.0',
           source_sha256=hashlib.sha256(raw).hexdigest(),
           files={str(p.relative_to(candidate)):hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in candidate.rglob('*') if p.is_file() and not any(x.startswith('.') for x in p.relative_to(candidate).parts)})
(ROOT/'results/phase2/lossless/candidate-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps(audit,indent=2))
