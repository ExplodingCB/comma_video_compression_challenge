"""Lossless, length-delimited blocks around the inherited F24S model bytes.

BLK1 stores a u32 total output size, u8 block count, then blocks with a u8
codec and u24 stored size. Codec 0 is raw; codec 1 is raw LZMA2 with a 1 MiB
dictionary. LZMA2 carries its own lc/lp/pb properties. Remaining bytes are
the unchanged residual table and RC64 token stream. All lengths are charged.
"""
import lzma
import struct

MAGIC = b'BLK1'
MAX_MODELS = 1 << 20


def decode_model_blocks(payload: bytes) -> tuple[bytes, bytes]:
    if len(payload) < 9 or payload[:4] != MAGIC:
        raise ValueError('Invalid BLK1 header')
    expected, count = struct.unpack_from('<IB', payload, 4)
    if not 0 < expected <= MAX_MODELS or not count:
        raise ValueError('Invalid BLK1 model size or block count')
    offset = 9
    output = bytearray()
    for _ in range(count):
        if offset + 4 > len(payload):
            raise ValueError('Truncated BLK1 block header')
        codec = payload[offset]
        size = int.from_bytes(payload[offset + 1:offset + 4], 'little')
        offset += 4
        if not size or offset + size > len(payload):
            raise ValueError('Truncated or empty BLK1 block')
        data = payload[offset:offset + size]
        offset += size
        if codec == 1:
            decoder = lzma.LZMADecompressor(
                format=lzma.FORMAT_RAW,
                filters=[{'id': lzma.FILTER_LZMA2, 'dict_size': 1 << 20}],
            )
            data = decoder.decompress(data, max_length=expected - len(output) + 1)
            if not decoder.eof or decoder.unused_data:
                raise ValueError('Invalid BLK1 compressed stream boundary')
        elif codec != 0:
            raise ValueError('Unknown BLK1 codec')
        output.extend(data)
        if len(output) > expected:
            raise ValueError('BLK1 output exceeds its declared size')
    if len(output) != expected or offset == len(payload):
        raise ValueError('Incomplete BLK1 models or missing residual/token stream')
    return bytes(output), payload[offset:]
