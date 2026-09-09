#!/usr/bin/env python3
"""Local full-data evaluator using the unchanged challenge models and DALI loader.

The metric loop and float32 accumulation follow challenge/evaluate.py.
This adds full-precision JSON, complete-output checks, and input provenance.
The organizer's T4 run remains authoritative for leaderboard placement.
"""
import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
CHALLENGE = ROOT / 'challenge'
sys.path.insert(0,str(CHALLENGE))


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    os.environ.setdefault('DALI_DISABLE_NVML','1')
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--submission-dir',type=Path,required=True)
    ap.add_argument('--report',type=Path,required=True)
    args=ap.parse_args()
    import torch
    from tqdm import tqdm
    from frame_utils import DaliVideoDataset,TensorVideoDataset,camera_size,seq_len
    from modules import DistortionNet,posenet_sd_path,segnet_sd_path
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    device=torch.device('cuda',0); torch.cuda.set_device(device)
    names=(CHALLENGE/'public_test_video_names.txt').read_text().splitlines()
    assert names==['0.mkv'], 'This experiment targets the public 600-pair fixture'
    raw=args.submission_dir/'inflated/0.raw'
    assert raw.stat().st_size==1200*camera_size[0]*camera_size[1]*3, 'Incomplete raw output'
    assert {p.name for p in (CHALLENGE/'videos').iterdir()}=={'0.mkv'}
    net=DistortionNet().eval().to(device)
    net.load_state_dicts(posenet_sd_path,segnet_sd_path,device)
    kwargs=dict(batch_size=16,device=device,num_threads=2,seed=1234,prefetch_queue_depth=4)
    gt=DaliVideoDataset(names,data_dir=CHALLENGE/'videos',**kwargs)
    comp=TensorVideoDataset(names,data_dir=args.submission_dir/'inflated',**kwargs)
    gt.prepare_data(); comp.prepare_data()
    dlgt=torch.utils.data.DataLoader(gt,batch_size=None,num_workers=0)
    dlcomp=torch.utils.data.DataLoader(comp,batch_size=None,num_workers=0)
    pd,sd,count=(torch.zeros([],device=device) for _ in range(3))
    started=time.monotonic()
    with torch.inference_mode():
        for a,b in tqdm(itertools.zip_longest(dlgt,dlcomp),total=38):
            assert a is not None and b is not None,'Unequal dataset lengths'
            x,y=a[2].to(device),b[2].to(device)
            assert x.shape==y.shape and list(y.shape)[1:]==[seq_len,camera_size[1],camera_size[0],3]
            p,s=net.compute_distortion(x,y)
            assert p.shape==s.shape==(x.shape[0],)
            pd+=p.sum(); sd+=s.sum(); count+=x.shape[0]
    assert int(count.item())==600
    pose=(pd/count).item(); seg=(sd/count).item()
    assert math.isfinite(pose) and math.isfinite(seg)
    archive=args.submission_dir/'archive.zip'
    original=(CHALLENGE/'videos/0.mkv').stat().st_size
    rate=archive.stat().st_size/original
    score=100*seg+math.sqrt(10*pose)+25*rate
    result=dict(status='full_local_evaluation',samples=600,archive_bytes=archive.stat().st_size,
                original_bytes=original,posenet_distortion=pose,segnet_distortion=seg,rate=rate,score=score,
                terms=dict(segmentation=100*seg,pose=math.sqrt(10*pose),rate=25*rate),
                evaluator_commit=subprocess.check_output(['git','-C',str(CHALLENGE),'rev-parse','HEAD'],text=True).strip(),
                archive_sha256=sha(archive),raw_sha256=sha(raw),
                evaluator_files={f:sha(CHALLENGE/f) for f in ('evaluate.py','frame_utils.py','modules.py','uv.lock')},
                input_files={str(p.relative_to(CHALLENGE)):sha(p) for p in (posenet_sd_path,segnet_sd_path,CHALLENGE/'videos/0.mkv')},
                torch=torch.__version__,gpu=torch.cuda.get_device_name(),numeric_policy='IEEE / TF32 disabled',
                dali_disable_nvml=os.environ.get('DALI_DISABLE_NVML'),
                elapsed_seconds=time.monotonic()-started)
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(result,indent=2)+'\n')
    lines=['=== Evaluation results over 600 samples ===',
           f'  Average PoseNet Distortion: {pose:.10f}',f'  Average SegNet Distortion: {seg:.10f}',
           f'  Submission file size: {archive.stat().st_size:,} bytes',f'  Original uncompressed size: {original:,} bytes',
           f'  Compression Rate: {rate:.10f}',f'  Final score: {score:.12f}',
           '  Local RTX 5080 / IEEE run; official leaderboard placement is pending.']
    (args.submission_dir/'report.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__': main()
