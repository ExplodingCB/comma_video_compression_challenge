"""Bounded CPU codec and reversible layout search; every candidate round-trips."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).with_name('deps')))
import argparse, bz2, hashlib, io, itertools, json, lzma, struct, time, zipfile
import brotli, pyppmd, zstandard as zstd
import numpy as np

OUT = ROOT / 'results/phase2/lossless'
MODELS = (ROOT / 'results/lossless/models.bin').read_bytes()
TAIL = (ROOT / 'results/lossless/tail.bin').read_bytes()
BASE = json.loads((ROOT / 'submissions/semantic_blocks/recipe.json').read_text())['blocks']
LZ = dict(id=lzma.FILTER_LZMA2, dict_size=1<<20, lc=0, lp=0, pb=0,
          mode=lzma.MODE_NORMAL, nice_len=32, mf=lzma.MF_BT2, depth=0)

def transform(b, kind, inverse=False):
    if kind == 'identity': return b
    a = np.frombuffer(b, np.uint8)
    name, sn = kind.split(':'); n = int(sn)
    if name == 'transpose':
        lim = len(a)//n*n
        shape = (n, lim//n) if inverse else (lim//n,n)
        return a[:lim].reshape(shape).T.copy().tobytes() + b[lim:]
    if name == 'xor':
        out = a.copy()
        if inverse:
            for i in range(n): out[i::n] = np.bitwise_xor.accumulate(out[i::n])
        else: out[n:] ^= a[:-n]
        return out.tobytes()
    if name == 'delta':
        out = a.copy()
        if inverse:
            for i in range(n): out[i::n] = np.cumsum(out[i::n],dtype=np.uint64).astype(np.uint8)
        else: out[n:] -= a[:-n]
        return out.tobytes()
    if name == 'bits':
        lim = len(a)//8*8
        bits = np.unpackbits(a[:lim])
        shape = (8,lim) if inverse else (lim,8)
        return np.packbits(bits.reshape(shape).T.copy()).tobytes()+b[lim:]
    if name == 'nibbles':
        # Separate high and low nibbles, retaining the final odd byte directly.
        lim=len(a)//2*2
        if inverse:
            unpack=np.empty(2*lim,dtype=np.uint8)
            unpack[::2]=a[:lim]>>4; unpack[1::2]=a[:lim]&15
            out=(unpack[:lim]<<4)|unpack[lim:]
        else:
            vals=np.concatenate((a[:lim]>>4,a[:lim]&15))
            out=(vals[::2]<<4)|vals[1::2]
        return out.tobytes()+b[lim:]
    raise ValueError(kind)

def codec_configs(full=True):
    yield 'raw',{}
    for variant,order in itertools.product(('I','H'),(2,3,4,5,6,8,12,16)):
        yield 'ppmd',dict(max_order=order,mem_size=1<<24,variant=variant)
    for quality,mode,window in itertools.product((9,10,11) if full else (10,11), (0,1,2), (16,18,22) if full else (18,)):
        yield 'brotli',dict(quality=quality,mode=mode,lgwin=window)
    for level in ((3,9,15,19,22) if full else (19,22)):
        yield 'zstd',dict(level=level)
    for lc,lp,pb in ((0,0,0),(0,1,0),(3,0,2),(2,0,0)):
        yield 'lzma',dict(LZ,lc=lc,lp=lp,pb=pb)

def encode(raw,codec,p):
    if codec=='raw': return raw
    if codec=='ppmd': return pyppmd.compress(raw,**p)
    if codec=='brotli': return brotli.compress(raw,**p)
    if codec=='zstd': return zstd.ZstdCompressor(**p,write_content_size=False,write_checksum=False,write_dict_id=False).compress(raw)
    if codec=='lzma': return lzma.compress(raw,format=lzma.FORMAT_RAW,filters=[p])
    raise ValueError(codec)

def decode(raw,codec,p,expected=None):
    if codec=='raw': return raw
    if codec=='ppmd':
        if p['variant']=='H':
            return pyppmd.Ppmd7Decoder(p['max_order'],p['mem_size']).decode(raw,expected)
        return pyppmd.decompress(raw,**p)
    if codec=='brotli': return brotli.decompress(raw)
    if codec=='zstd': return zstd.ZstdDecompressor().decompress(raw,max_output_size=1<<20)
    if codec=='lzma': return lzma.decompress(raw,format=lzma.FORMAT_RAW,filters=[dict(id=lzma.FILTER_LZMA2,dict_size=1<<20)])
    raise ValueError(codec)

def extra(codec,kind):
    # BLK1's codec+u24 length is always charged. New codec ID includes PPMd variant;
    # 2 extra bytes carry its order and memory exponent. Transforms use 2 bytes.
    return (5 if codec=='ppmd' else 0) + (2 if kind!='identity' else 0)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--layouts',action='store_true'); args=ap.parse_args()
    started=time.monotonic(); rows=[]; best={}
    spans=[(r['start'],r['end']) for r in BASE]
    spans += [(0,len(MODELS)),(0,16597),(16597,52637),(52637,74860),(0,24895)]
    kinds=['identity']
    if args.layouts:
        kinds += [f'{x}:{n}' for x in ('transpose','delta','xor') for n in (1,2,3,4,8,12,16,32,64)] + ['bits:8','nibbles:2']
    for start,end in spans:
        raw=MODELS[start:end]; key=f'{start}:{end}'; best[key]=None
        for kind in kinds:
            altered=transform(raw,kind); assert transform(altered,kind,True)==raw
            for codec,p in codec_configs(full=not args.layouts):
                packed=encode(altered,codec,p)
                try:
                    restored=decode(packed,codec,p,len(altered))
                except (ValueError,RuntimeError) as error:
                    rows.append(dict(start=start,end=end,kind=kind,codec=codec,params=p,
                                     rejected='decoder exception',error=str(error)))
                    continue
                if transform(restored,kind,True)!=raw:
                    rows.append(dict(start=start,end=end,kind=kind,codec=codec,params=p,
                                     rejected='roundtrip mismatch',decoded_bytes=len(restored)))
                    continue
                cost=len(packed)+4+extra(codec,kind)
                row=dict(start=start,end=end,kind=kind,codec=codec,params=p,compressed_bytes=len(packed),charged_block_bytes=cost)
                rows.append(row)
                if best[key] is None or cost<best[key]['charged_block_bytes']:
                    best[key]=row
                    print(json.dumps(row),flush=True)
        label='layouts' if args.layouts else 'codecs'
        (OUT/f'{label}-ledger.json').write_text(json.dumps(rows,indent=2)+'\n')
        (OUT/f'{label}-best.json').write_text(json.dumps(best,indent=2)+'\n')
    report=dict(candidates=len(rows),elapsed_seconds=time.monotonic()-started,best=best,
                versions={m:m.__version__ for m in ()}, models_sha256=hashlib.sha256(MODELS).hexdigest())
    (OUT/f'{label}-report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
