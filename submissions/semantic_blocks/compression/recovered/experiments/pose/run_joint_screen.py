"""Recover the renderer metadata candidate, with a source-basis control."""
from common import *
import sys
import rescue

def prepare_seg(masters,archive_path,output):
    import torch
    from modules import SegNet,segnet_sd_path
    from safetensors.torch import load_file
    if output.exists():return
    device=setup_gpu();model=SegNet().eval().to(device)
    model.load_state_dict(load_file(segnet_sd_path,device=str(device)))
    stream=np.memmap(masters,dtype=np.uint8,mode='r',shape=(600,874,1164,3))
    targets=np.load(OUT/'cache/seg_targets.npy',mmap_mode='r');values=[]
    with torch.inference_mode():
        for start in range(0,600,16):
            rgb=torch.from_numpy(np.array(stream[start:start+16])).permute(0,3,1,2).float().to(device)
            logits=model(model.preprocess_input(rgb[:,None]))
            labels=torch.from_numpy(np.array(targets[start:start+16])).to(device)
            values.extend((logits.argmax(1)!=labels).float().mean((1,2)).cpu().numpy())
    errors=np.asarray(values,dtype=np.float32);output.parent.mkdir(parents=True,exist_ok=True);np.save(output,errors)
    write_json(output.with_suffix('.json'),dict(pairs=600,archive_sha256=sha(archive_path.read_bytes()),
               masters_sha256=file_sha(masters),segmentation_distortion=float(errors.mean()),
               error_sha256=sha(output.read_bytes()),target_manifest=str(OUT/'cache/manifest.json')))
    print(f'candidate SegNet mean: {errors.mean():.12g}',flush=True)

def main():
    ctx=context();control=OUT/'control';control.mkdir(exist_ok=True)
    (control/'archive.zip').write_bytes(ctx['data']);(control/'canonical.bin').write_bytes(ctx['canonical'])
    np.save(control/'codes.npy',ctx['codes'])
    name='all_metadata_b2';meta=ROOT/'results/phase2/renderer'
    masters=meta/'masters'/name/'masters.uint8';archive_path=meta/'archives'/name/'archive.zip'
    seg=OUT/'joint-cache'/f'{name}-seg.npy';prepare_seg(masters,archive_path,seg)
    for candidate,extra in [
        ('control',[]),
        (name,['--source-archive',str(archive_path),'--master-frames',str(masters),'--seg-errors',str(seg)]),
        ('atom0-linear',[]),
    ]:
        sys.argv=['rescue.py','--candidate',str(OUT/candidate),'--output',str(OUT/('rescue-'+candidate)),
                  '--pairs','96','--passes','3','--batch','8',*extra]
        rescue.main()
if __name__=='__main__':main()
