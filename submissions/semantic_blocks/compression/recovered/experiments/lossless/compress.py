#!/usr/bin/env python3
"""Repack PR135's trained archive with byte-exact Brotli/LZMA2 model blocks."""
import argparse
import hashlib
import io
import json
import lzma
from pathlib import Path
import struct
import zipfile

from runtime.block_container import decode_model_blocks, transpose

SOURCE_SHA256 = '12cf5d71a94065184f097c3e40dfe9f1db8402a1a76a80efc76a6956fe1e4004'


def build_payload(models: bytes, tail: bytes, recipe: dict) -> bytes:
    blocks = []
    offset = 0
    codecs = {'raw': 0, 'lzma': 1, 'brotli': 2}
    for span in recipe['blocks']:
        if span['start'] != offset or not offset < span['end'] <= len(models):
            raise ValueError('Noncontiguous or out-of-range recipe')
        data = models[offset:span['end']]
        transform = b''
        codec = codecs[span['codec']]
        kind = span.get('kind', 'identity')
        if kind != 'identity':
            name, stride = kind.split(':')
            if name != 'transpose':
                raise ValueError('Unsupported transform')
            stride = int(stride)
            data = transpose(data, stride)
            transform = bytes((1, stride))
        if codec == 1:
            data = lzma.compress(data, format=lzma.FORMAT_RAW, filters=[span['params']])
        elif codec == 2:
            import brotli
            data = brotli.compress(data, **span['params'])
        blocks.append(bytes((codec | (128 if transform else 0),)) + len(data).to_bytes(3, 'little') + transform + data)
        offset = span['end']
    if offset != len(models):
        raise ValueError('Recipe does not cover all model data')
    payload = b'BLK2' + struct.pack('<IB', len(models), len(blocks)) + b''.join(blocks) + tail
    if decode_model_blocks(payload) != (models, tail):
        raise ValueError('Lossless round trip failed')
    return payload


def zip_payload(payload: bytes) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_STORED) as z:
        info = zipfile.ZipInfo('p', date_time=(1980, 1, 1, 0, 0, 0))
        info.create_system = 3
        info.external_attr = 0o644 << 16
        z.writestr(info, payload)
    return output.getvalue()


def build(source: bytes, recipe: dict) -> bytes:
    if hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
        raise ValueError('Expected original PR135 archive; see LINEAGE.md')
    with zipfile.ZipFile(io.BytesIO(source)) as z:
        if z.namelist() != ['p']:
            raise ValueError('Unexpected source ZIP members')
        outer = z.read('p')
    decoder = lzma.LZMADecompressor(format=lzma.FORMAT_RAW,
        filters=[dict(id=lzma.FILTER_LZMA2, dict_size=1 << 16)])
    models = decoder.decompress(outer)
    if not decoder.eof or not decoder.unused_data:
        raise ValueError('Incomplete source stream')
    return zip_payload(build_payload(models, decoder.unused_data, recipe))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--output', type=Path, default=Path(__file__).with_name('archive.zip'))
    args = ap.parse_args()
    if args.source.resolve() == args.output.resolve():
        raise ValueError('Source and output must differ')
    source = args.source.read_bytes()
    recipe = json.loads(Path(__file__).with_name('recipe.json').read_text())
    result = build(source, recipe)
    if result != build(source, recipe):
        raise ValueError('Repeated builds differ')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(result)
    print(json.dumps(dict(archive_bytes=len(result), saved_bytes=len(source)-len(result),
                         sha256=hashlib.sha256(result).hexdigest()), indent=2))


if __name__ == '__main__':
    main()
