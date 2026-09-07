#!/usr/bin/env python3
"""Repack the attributed PR135 archive; this does not train from the source video."""
import argparse
import hashlib
import io
import json
import lzma
from pathlib import Path
import struct
import zipfile

from runtime.block_container import decode_model_blocks

SOURCE_SHA256 = '12cf5d71a94065184f097c3e40dfe9f1db8402a1a76a80efc76a6956fe1e4004'


def build(source: bytes, recipe: dict) -> bytes:
    if hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
        raise ValueError('Expected the original PR135 archive; see LINEAGE.md')
    with zipfile.ZipFile(io.BytesIO(source)) as z:
        if z.namelist() != ['p']:
            raise ValueError('Unexpected source ZIP members')
        outer = z.read('p')
    decoder = lzma.LZMADecompressor(format=lzma.FORMAT_RAW,
                                  filters=[dict(id=lzma.FILTER_LZMA2, dict_size=1 << 16)])
    models = decoder.decompress(outer)
    if not decoder.eof or not decoder.unused_data:
        raise ValueError('Incomplete source stream')
    tail = decoder.unused_data
    blocks = []
    offset = 0
    for span in recipe['blocks']:
        if span['start'] != offset or span['end'] <= offset:
            raise ValueError('Noncontiguous block recipe')
        raw = models[offset:span['end']]
        codec = span['codec']
        if codec == 1:
            raw = lzma.compress(raw, format=lzma.FORMAT_RAW, filters=[span['filter']])
        elif codec != 0:
            raise ValueError('Unknown codec')
        blocks.append(bytes([codec]) + len(raw).to_bytes(3, 'little') + raw)
        offset = span['end']
    if offset != len(models):
        raise ValueError('Recipe does not cover the source models')
    payload = b'BLK1' + struct.pack('<IB', len(models), len(blocks)) + b''.join(blocks) + tail
    if decode_model_blocks(payload) != (models, tail):
        raise ValueError('Lossless round trip failed')
    output = io.BytesIO()
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_STORED) as z:
        info = zipfile.ZipInfo('p', date_time=(1980, 1, 1, 0, 0, 0))
        info.create_system = 3
        info.external_attr = 0o644 << 16
        z.writestr(info, payload)
    return output.getvalue()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--output', type=Path, default=Path(__file__).with_name('archive.zip'))
    args = ap.parse_args()
    if args.source.resolve() == args.output.resolve():
        raise ValueError('Source and output must be different files')
    recipe = json.loads(Path(__file__).with_name('recipe.json').read_text())
    source = args.source.read_bytes()
    result = build(source, recipe)
    if result != build(source, recipe):
        raise ValueError('Repeated builds differ')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(result)
    print(json.dumps(dict(archive_bytes=len(result), saved_bytes=len(source)-len(result),
                         sha256=hashlib.sha256(result).hexdigest()), indent=2))


if __name__ == '__main__':
    main()
