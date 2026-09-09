"""Re-encode promising calibration tables, charge archives, and package decoders.

Native arithmetic encoder is attributed PR135 experiment code. Full independent
causal GPU inflation is still required after the cached probability replay check.
"""
import hashlib
import io
import json
from pathlib import Path
import shutil
import struct
import sys
import time
import zipfile

from fit import ROOT, OUT, SOURCE, CHUNK, ROWS, PRECISION, arrays, features
import numpy as np
from runtime.residual_archive import read_residual_archive, _probability_table
from runtime.block_container import decode_model_blocks

sys.path.insert(0, str(ROOT / 'references/pr135-experiments/src'))
from cpr1_sub4.bits import pack_signed
from cpr1_sub4.entropy.rc64 import NativeEncoder, NativeDecoder


def table_bytes(row):
    codes = np.asarray(row['codes'], dtype=np.int8)
    packed = pack_signed(codes.ravel(), 6)
    scale = np.asarray([row['scale']], dtype='<f2').tobytes()
    bins = row['bins']
    if bins:
        return b'ETP1' + bytes([len(bins)]) + bytes(bins) + struct.pack('<H', len(codes)) + scale + packed
    return scale + packed


def make_zip(payload):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_STORED) as z:
        info = zipfile.ZipInfo('p', date_time=(1980, 1, 1, 0, 0, 0))
        info.create_system = 3
        info.external_attr = 0o644 << 16
        z.writestr(info, payload)
    return buffer.getvalue()


def package(row, archive):
    target = OUT / 'candidates' / row['name']
    target.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT/'results/source-audit.json').read_text())
    for name in manifest['candidate_source_hashes']:
        name = name.replace('\\', '/')
        if name in ('compress.py', 'compress.sh', 'recipe.json'):
            continue
        destination = target / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((SOURCE/name).read_bytes().replace(b'\r\n', b'\n'))
    shutil.copyfile(SOURCE/'LICENSE', target/'LICENSE')
    lineage = (SOURCE/'LINEAGE.md').read_text()
    lineage += '\nExperimental causal probability calibration; not the archive submitted in PR #141.\n'
    (target/'LINEAGE.md').write_text(lineage, newline='\n')
    (target/'archive.zip').write_bytes(archive)
    (target/'table.json').write_text(json.dumps(row, indent=2)+'\n')
    if row['bins']:
        patch_margin_reader(target)
    return target


def patch_margin_reader(target):
    path = target/'runtime/residual_archive.py'
    text = path.read_text()
    text = text.replace('    values: np.ndarray\n', '    values: np.ndarray\n    margins: tuple[int, ...] = ()\n', 1)
    old = '''    compact_size = fixed_size - len(FIXED_MAGIC)
    if len(section) <= compact_size:
        raise ResidualArchiveError("truncated F24S residual or token section")
    residual = FIXED_MAGIC + section[:compact_size]
    table = _decode_fixed_table(residual)
'''
    new = '''    if section.startswith(b"ETP1"):
        if len(section) < 10:
            raise ResidualArchiveError("truncated ETP1 table")
        count = section[4]
        if not 1 <= count <= 3:
            raise ResidualArchiveError("invalid ETP1 margin count")
        margins = tuple(section[5:5 + count])
        if tuple(sorted(set(margins))) != margins or not all(0 < v < 128 for v in margins):
            raise ResidualArchiveError("invalid ETP1 margins")
        states = int.from_bytes(section[5 + count:7 + count], "little")
        if states != FIXED_STATES * (count + 1):
            raise ResidualArchiveError("invalid ETP1 state count")
        scale = float(np.frombuffer(section[7 + count:9 + count], dtype="<f2")[0])
        if not np.isfinite(scale) or scale <= 0:
            raise ResidualArchiveError("invalid ETP1 scale")
        compact_size = 9 + count + packed_length(states * NUM_CLASSES, FIXED_BITS)
        if len(section) <= compact_size:
            raise ResidualArchiveError("truncated ETP1 codes or token stream")
        codes = np.asarray(unpack_signed(section[9 + count:compact_size], states * NUM_CLASSES, FIXED_BITS), dtype=np.int8).reshape(states, NUM_CLASSES)
        table = QuantizedTable("boundary_margin", FIXED_BITS, codes, scale,
                               codes.astype(np.float32) * scale, margins)
        residual = section[:compact_size]
    else:
        compact_size = fixed_size - len(FIXED_MAGIC)
        if len(section) <= compact_size:
            raise ResidualArchiveError("truncated F24S residual or token section")
        residual = FIXED_MAGIC + section[:compact_size]
        table = _decode_fixed_table(residual)
'''
    if old not in text:
        raise RuntimeError('Residual parser patch no longer matches the reviewed source')
    text = text.replace(old, new, 1)
    needle = '                corrected = base_logits + parts.table.values[feature]\n'
    replacement = '''                if parts.table.margins:
                    top = np.partition(base_logits, -2, axis=1)[:, -2:]
                    margin = top[:, 1] - top[:, 0]
                    bucket = np.searchsorted(np.asarray(parts.table.margins), margin, side="right")
                    feature = feature * (len(parts.table.margins) + 1) + bucket
                corrected = base_logits + parts.table.values[feature]
'''
    if needle not in text:
        raise RuntimeError('Probability feature patch no longer matches reviewed source')
    text = text.replace(needle, replacement, 1)
    path.write_text(text, newline='\n')


def main():
    report = json.loads((OUT/'screen.json').read_text())
    rows = report['finalists']
    baseline_row = next(r for r in rows if r['name'] == 'baseline')
    candidates = [r for r in rows if r['name'] == 'baseline' or
                  r['full_charged_ideal_bytes'] < baseline_row['full_charged_ideal_bytes'] - 1]
    candidates = sorted(candidates, key=lambda r:r['name'] != 'baseline')
    base, original, symbols = arrays()
    parts = read_residual_archive(SOURCE/'archive.zip')
    with zipfile.ZipFile(SOURCE/'archive.zip') as z:
        payload = z.read('p')
    models, tail = decode_model_blocks(payload)
    prefix = payload[:-len(tail)]
    result = []
    for row in candidates:
        table = np.asarray(row['codes'], dtype=np.float32) * row['scale']
        encoder = NativeEncoder(OUT/'librc64.so')
        started = time.monotonic()
        for start in range(0, ROWS, CHUNK):
            b = np.asarray(base[start:start+CHUNK])
            f = features(b, original[start:start+CHUNK], row['bins'])
            s = np.asarray(symbols[start:start+CHUNK], dtype=np.int32)
            p = _probability_table(b.astype(np.float32)/8 + table[f], PRECISION)
            encoder.encode(s, p)
        stream = encoder.finish()
        encoder.close()
        if row['name'] == 'baseline' and stream != parts.token_stream:
            raise RuntimeError('Baseline encoder does not exactly reproduce submitted token bytes')
        decoder = NativeDecoder(OUT/'librc64.so', stream)
        digest = hashlib.sha256()
        for start in range(0, ROWS, CHUNK):
            b = np.asarray(base[start:start+CHUNK])
            f = features(b, original[start:start+CHUNK], row['bins'])
            s = np.asarray(symbols[start:start+CHUNK])
            p = _probability_table(b.astype(np.float32)/8 + table[f], PRECISION)
            decoded = decoder.decode(p).astype(np.uint8)
            if not np.array_equal(decoded, s):
                raise RuntimeError('Cached probability round trip did not recover exact tokens')
            digest.update(decoded.tobytes())
        decoder.close()
        table_payload = table_bytes(row)
        assert len(table_payload) == row['table_bytes']
        archive = make_zip(prefix + table_payload + stream)
        if row['name'] == 'baseline' and archive != (SOURCE/'archive.zip').read_bytes():
            raise RuntimeError('Baseline archive reconstruction differs')
        target = package(row, archive) if row['name'] != 'baseline' else None
        record = dict(name=row['name'], archive_bytes=len(archive), saved_bytes=186151-len(archive),
                      table_bytes=len(table_payload), token_bytes=len(stream),
                      archive_sha256=hashlib.sha256(archive).hexdigest(),
                      model_sha256=hashlib.sha256(models).hexdigest(),
                      cached_coding_order_token_sha256=digest.hexdigest(),
                      cached_decode_exact=True, independent_causal_decode='pending',
                      candidate=str(target) if target else None, seconds=time.monotonic()-started)
        result.append(record)
        print(json.dumps(record), flush=True)
        (OUT/'encoded.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
