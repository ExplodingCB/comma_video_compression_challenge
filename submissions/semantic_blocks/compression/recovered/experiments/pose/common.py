"""Exact carrier experiments, with frozen sources and charged BLK1 archives."""
from pathlib import Path
import hashlib, io, json, lzma, struct, sys, zipfile
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/phase2/pose'
SOURCE = ROOT / 'submissions/semantic_blocks'
REF = ROOT / 'references/pr135-experiments'
for path in (REF/'src', REF/'scripts', SOURCE, ROOT/'challenge'):
    sys.path.insert(0, str(path))
from runtime.residual_archive import read_residual_archive, _cap1_body_bytes, _restore_cap1
from runtime.block_container import decode_model_blocks
from runtime.carrier_repack import split_frame0_selector_carrier
from runtime.frame0_selector import decode_selector, apply_pixel_mode
from runtime.f26_inflate import _load_renderer
from cpr1_sub4.carrier_cbq import requantize_cpr1_basis
from cpr1_sub4.entropy.coefficient_ar1_codec import decode_cap1, encode_cap1
from cpr1_sub4.carrier_repack import decode_cpr1_coefficients, encode_cps3_coefficients, decode_cps3
from cpr1_sub4.residual_archive import _encode_f12_cap1_body
from polish_carrier_coefficients import ExactCarrierRenderer

def sha(x): return hashlib.sha256(x).hexdigest()
def file_sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def context(source_archive=None):
    data = (SOURCE/'archive.zip').read_bytes()
    assert sha(data) == 'd8ae5a9f4c3205ff2f2d823943afd3d25f8eb5cf9fd1e148c1159175e8f80c48'
    path=SOURCE/'archive.zip' if source_archive is None else Path(source_archive)
    data=path.read_bytes()
    with zipfile.ZipFile(io.BytesIO(data)) as z: models, tail = decode_model_blocks(z.read('p'))
    parts = read_residual_archive(path)
    cap, selector = split_frame0_selector_carrier(parts.carrier_blob)
    canonical = decode_cap1(cap, frames=600, dimensions=12)
    codes = decode_cpr1_coefficients(canonical, frames=600, dimensions=12)
    return dict(data=data,models=models,tail=tail,parts=parts,canonical=canonical,codes=codes,selector=selector,source_archive=str(path),
                recipe=json.loads((SOURCE/'recipe.json').read_text()))

def archive(ctx, canonical, codes):
    canonical = decode_cps3(encode_cps3_coefficients(canonical,codes,frames=600,dimensions=12,mode=3,threshold=16),frames=600,dimensions=12)
    cap,_ = encode_cap1(canonical,frames=600,dimensions=12)
    # Fixed CAP order 1 is scales,predictor,lengths,ks,basis,rice.
    capbody = _encode_f12_cap1_body(cap, 1)
    assert _restore_cap1(capbody)==cap
    original_models=ctx['models']; capstart=4+16593+36040
    oldcaplen=_cap1_body_bytes(original_models[capstart:])
    models=original_models[:capstart]+capbody+original_models[capstart+oldcaplen:]
    # Retain the submitted partition's internal offsets; shortening happens in
    # the last carrier block, and is fully charged including all block headers.
    blocks=[]
    for index,span in enumerate(ctx['recipe']['blocks']):
        end=len(models) if index==4 else span['end']
        chunk=models[span['start']:end]
        if span['codec']==1: chunk=lzma.compress(chunk,format=lzma.FORMAT_RAW,filters=[span['filter']])
        blocks.append(bytes([span['codec']])+len(chunk).to_bytes(3,'little')+chunk)
    payload=b'BLK1'+struct.pack('<IB',len(models),len(blocks))+b''.join(blocks)+ctx['tail']
    assert decode_model_blocks(payload)==(models,ctx['tail'])
    result=io.BytesIO()
    with zipfile.ZipFile(result,'w',compression=zipfile.ZIP_STORED) as z:
        info=zipfile.ZipInfo('p',date_time=(1980,1,1,0,0,0));info.create_system=3;info.external_attr=0o644<<16
        z.writestr(info,payload)
    return result.getvalue(),canonical

def setup_gpu():
    import torch
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    torch.cuda.set_device(0)
    return torch.device('cuda')

def load_pose(device):
    from modules import PoseNet,posenet_sd_path
    from safetensors.torch import load_file
    model=PoseNet().eval().to(device);model.load_state_dict(load_file(posenet_sd_path,device=str(device)))
    for p in model.parameters(): p.requires_grad_(False)
    return model

def renderer(canonical,device):
    return ExactCarrierRenderer.create(_load_renderer(SOURCE/'cpr1'),canonical,device)

def raw_stream(master_frames=None):
    if master_frames is not None:
        path=Path(master_frames);assert path.stat().st_size==600*874*1164*3
        class Masters:
            def __init__(self):self.data=np.memmap(path,dtype=np.uint8,mode='r',shape=(600,874,1164,3))
            def __getitem__(self,key):
                keys=np.asarray(key);assert np.all(keys%2==1)
                return self.data[keys//2]
        return Masters()
    path=SOURCE/'inflated/0.raw'
    assert path.stat().st_size==1200*874*1164*3
    return np.memmap(path,dtype=np.uint8,mode='r',shape=(1200,874,1164,3))

def render_selected(render,codes,ids,selector):
    import torch
    import torch.nn.functional as F
    # The deployed carrier matrix product uses64 rows, with24 in the final
    # chunk. Preserve that GEMM shape: smaller matrices can round differently
    # at pixel thresholds on this GPU. Upscale only the requested rows.
    frames=np.empty((len(codes),874,1164,3),dtype=np.uint8)
    with torch.inference_mode():
        for batch_rows in (64,24):
            chosen=np.flatnonzero((np.asarray(ids)<576)==(batch_rows==64))
            for offset in range(0,len(chosen),batch_rows):
                ix=chosen[offset:offset+batch_rows];chunk=codes[ix]
                if not len(ix):continue
                padded=np.concatenate([chunk,np.repeat(chunk[-1:],batch_rows-len(chunk),axis=0)])
                coeff=torch.from_numpy(np.ascontiguousarray(padded.astype(np.float32)*render.coefficient_scales[None])).to(render.device)
                carrier=torch.einsum('bk,kchw->bchw',coeff,render.basis)[:len(ix)]/np.sqrt(12)
                low=(127.5+render.runtime.CARRIER_AMPLITUDE*carrier).clamp(0,255).round()
                high=F.interpolate(low,size=(874,1164),mode='bicubic',align_corners=False).clamp(0,255).round()
                frames[ix]=high.to(torch.uint8).permute(0,2,3,1).cpu().numpy()
    modes,choices=decode_selector(selector)
    for idx,mode in enumerate(modes):
        chosen=np.flatnonzero(choices[ids]==idx)
        if idx and chosen.size:frames[chosen]=apply_pixel_mode(frames[chosen],mode)
    return frames

def errors(render,pose,raw,targets,codes,ids,selector,batch=16):
    import torch
    result=[]
    with torch.inference_mode():
        for start in range(0,len(ids),batch):
            ix=ids[start:start+batch]
            slaves=render_selected(render,codes[start:start+batch],ix,selector)
            pairs=np.stack([slaves,np.asarray(raw[2*ix+1]).copy()],axis=1)
            inputs=torch.from_numpy(pairs).permute(0,1,4,2,3).float().to(render.device)
            output=pose(pose.preprocess_input(inputs))['pose'][...,:6]
            values=(output-torch.from_numpy(targets[ix]).to(render.device)).square().mean(dim=1)
            result.extend(values.cpu().numpy().tolist())
    return np.asarray(result)

def score(pose,bytes_,seg=0.0002964019950013608):return 100*seg+np.sqrt(10*pose)+25*bytes_/37545489

def write_json(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(data,indent=2)+'\n')
    temporary.replace(path)


def save_array(path,value):
    """Keep the preceding valid coefficient checkpoint if interrupted mid-save."""
    temporary=path.with_name(path.name+'.tmp')
    with temporary.open('wb') as stream:
        np.save(stream,value,allow_pickle=False)
    temporary.replace(path)
