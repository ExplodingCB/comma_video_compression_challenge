"""Read candidate learned-state blobs using its own decoder, without GPU work."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--runtime-source',type=Path,required=True)
    ap.add_argument('--archive',type=Path,required=True)
    ap.add_argument('--include-hpac-bytes',action='store_true')
    args=ap.parse_args()
    sys.path.insert(0,str(args.runtime_source.resolve()))
    from runtime.residual_archive import read_residual_archive
    from runtime.entropy.renderer_weight_codec import decode_wans1
    parts=read_residual_archive(args.archive)
    sha=lambda data:hashlib.sha256(data).hexdigest()
    tensors={r.schema.name:sha(r.values.tobytes()) for r in decode_wans1(parts.semantic_blob)}
    result={key:sha(getattr(parts,key)) for key in
            ('semantic_blob','carrier_blob','hpac_blob','token_stream','residual_payload')}
    result.update(renderer_tensors=tensors,table_values_sha256=sha(parts.table.values.tobytes()),
                  table_shape=list(parts.table.values.shape),table_margins=list(getattr(parts.table,'margins',())))
    if args.include_hpac_bytes:result['hpac_bytes_hex']=parts.hpac_blob.hex()
    print(json.dumps(result,sort_keys=True))

if __name__=='__main__':main()
