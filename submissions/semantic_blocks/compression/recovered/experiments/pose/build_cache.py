"""Produce local exact DALI targets and baseline arrays, with provenance."""
from common import *
import time
def main():
    import torch
    from modules import DistortionNet,posenet_sd_path,segnet_sd_path
    from frame_utils import DaliVideoDataset,TensorVideoDataset
    out=OUT/'cache';out.mkdir(parents=True,exist_ok=True)
    device=setup_gpu();net=DistortionNet().eval().to(device)
    net.load_state_dicts(posenet_sd_path,segnet_sd_path,device)
    kw=dict(batch_size=16,device=device,num_threads=2,seed=1234,prefetch_queue_depth=4)
    gt=DaliVideoDataset(['0.mkv'],data_dir=ROOT/'challenge/videos',**kw)
    base=TensorVideoDataset(['0.mkv'],data_dir=SOURCE/'inflated',**kw)
    gt.prepare_data();base.prepare_data();targets=[];outputs=[];pes=[];ses=[];labels=[];count=0
    pd=torch.zeros([],device=device);sd=torch.zeros([],device=device)
    start=time.monotonic()
    with torch.inference_mode():
        for a,b in zip(gt,base):
            p,s=net(a[2].to(device));q,t=net(b[2].to(device));pe=net.posenet.compute_distortion(p,q);se=net.segnet.compute_distortion(s,t)
            pd+=pe.sum();sd+=se.sum();count+=len(pe)
            targets.append(p['pose'][...,:6].cpu().numpy());outputs.append(q['pose'][...,:6].cpu().numpy())
            labels.append(s.argmax(1).to(torch.uint8).cpu().numpy());pes.append(pe.cpu().numpy());ses.append(se.cpu().numpy())
            print(f'cache {count}/600',flush=True)
    assert count==600
    values={'pose_targets':np.concatenate(targets),'pose_baseline':np.concatenate(outputs),
            'pose_errors':np.concatenate(pes),'seg_errors':np.concatenate(ses),'seg_targets':np.concatenate(labels)}
    for name,value in values.items():np.save(out/(name+'.npy'),value,allow_pickle=False)
    write_json(out/'manifest.json',dict(status='complete_local_DALI_cache',pairs=count,batch_size=16,seed=1234,
               num_threads=2,prefetch_queue_depth=4,pose=(pd/600).item(),seg=(sd/600).item(),
               archive_sha256=sha((SOURCE/'archive.zip').read_bytes()),numeric_policy='IEEE TF32 disabled',
               torch=torch.__version__,gpu=torch.cuda.get_device_name(),elapsed_seconds=time.monotonic()-start,
               files={name:dict(shape=list(value.shape),sha256=sha((out/(name+'.npy')).read_bytes())) for name,value in values.items()}))
    print((out/'manifest.json').read_text(),flush=True)
if __name__=='__main__':main()
