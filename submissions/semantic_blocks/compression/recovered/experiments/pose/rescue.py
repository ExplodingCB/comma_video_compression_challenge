"""Batched exact-evaluated Gauss-Newton recovery of quantized pose atoms.

Differentiable STE proposals are never scores: acceptance uses the unmodified
PoseNet preprocessing and uint8 camera frames including stored selectors.
"""
from common import *
import argparse,time,itertools
from cpr1_sub4.differentiable_metrics import posenet_preprocess_differentiable
from cpr1_sub4.joint_pose_solve import solve_damped_least_squares,quantize_int12_update

def jacobians(render,pose,masters,codes):
    import torch
    import torch.nn.functional as F
    def ste(x):return x+(x.round()-x).detach()
    c=torch.tensor(codes,dtype=torch.float32,device=render.device,requires_grad=True)
    coeff=c*torch.as_tensor(render.coefficient_scales,device=render.device)[None]
    carrier=torch.einsum('bk,kchw->bchw',coeff,render.basis)/np.sqrt(12)
    low=ste((127.5+render.runtime.CARRIER_AMPLITUDE*carrier).clamp(0,255))
    slaves=ste(F.interpolate(low,size=(874,1164),mode='bicubic',align_corners=False).clamp(0,255))
    master=torch.from_numpy(masters).permute(0,3,1,2).float().to(render.device)
    inputs=torch.stack([slaves,master],dim=1)
    outputs=pose(posenet_preprocess_differentiable(inputs))['pose'][...,:6]
    jac=[]
    for k in range(6):
        jac.append(torch.autograd.grad(outputs[:,k].sum(),c,retain_graph=k<5)[0].detach().cpu().numpy())
    return outputs.detach().cpu().numpy(),np.stack(jac,axis=1)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--candidate',required=True);ap.add_argument('--pairs',type=int,default=96)
    ap.add_argument('--passes',type=int,default=3);ap.add_argument('--batch',type=int,default=4)
    ap.add_argument('--output',required=True);ap.add_argument('--resume',action='store_true')
    ap.add_argument('--source-archive',help='Renderer-changedF24S archive to preserve while solving its pose coefficients')
    ap.add_argument('--master-frames',help='600x874x1164x3uint8 frame1 cache for source-archive')
    ap.add_argument('--seg-errors',help='600perpairSegNet errors of the changed frame1 cache')
    ap.add_argument('--warm-start',help='Signed-int12 coefficient checkpoint for all600 pairs')
    ap.add_argument('--coordinate',action='store_true',help='also evaluate all24 unit coordinate moves')
    ap.add_argument('--neighbours',type=int,default=0,choices=(0,1,2,3),help='integer cube dimensions around the Gauss-Newton centre')
    args=ap.parse_args()
    ctx=context(args.source_archive);folder=Path(args.candidate);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    assert archive(ctx,ctx['canonical'],ctx['codes'])[0]==ctx['data'],'source archive must reconstruct byte for byte'
    canonical=ctx['canonical'] if args.source_archive else (folder/'canonical.bin').read_bytes()
    codes=ctx['codes'].copy() if args.source_archive else np.load(folder/'codes.npy')
    if args.warm_start:codes=np.load(args.warm_start,allow_pickle=False)
    if args.resume and (out/'codes.npy').exists():codes=np.load(out/'codes.npy')
    assert codes.shape==(600,12) and np.issubdtype(codes.dtype,np.integer)
    assert codes.min()>=-2048 and codes.max()<=2047
    targets=np.load(OUT/'cache/pose_targets.npy');base_errors=np.load(OUT/'cache/pose_errors.npy')
    ids=np.arange(600) if args.pairs==600 else np.linspace(0,599,args.pairs,dtype=int)
    _,choices=decode_selector(ctx['selector']);eligible=ids[choices[ids]==0]
    if bool(args.source_archive)!=bool(args.master_frames) or bool(args.master_frames)!=bool(args.seg_errors):
        raise ValueError('changedrenderer requires archive,masters,andcompleteSegNeterrors together')
    base_seg=float(np.load(OUT/'cache/seg_errors.npy')[ids].mean())
    candidate_seg=float(np.load(args.seg_errors)[ids].mean()) if args.seg_errors else base_seg
    def delta(pose_value,rate_bytes):return float(score(pose_value,rate_bytes,candidate_seg)-score(base_errors[ids].mean(),186151,base_seg))
    device=setup_gpu();pose=load_pose(device);raw=raw_stream(args.master_frames);render=renderer(canonical,device)
    before=errors(render,pose,raw,targets,codes[ids],ids,ctx['selector'])
    selected_errors=before.copy();lookup={int(v):i for i,v in enumerate(ids)}
    data,_=archive(ctx,canonical,codes)
    initial=dict(pass_index=0,pairs=len(ids),pose=float(before.mean()),archive_bytes=len(data),
                 score_delta=delta(before.mean(),len(data)))
    history=[initial];pass_offset=0
    if args.resume and (out/'report.json').exists():
        previous=json.loads((out/'report.json').read_text())
        assert previous['source_archive_sha256']==sha(ctx['data'])
        assert previous['indices']==ids.tolist()
        assert previous.get('master_frames')==args.master_frames
        history=previous['history'];pass_offset=history[-1]['pass_index']
        initial['resume_after_pass']=pass_offset
    print(json.dumps(initial),flush=True);started=time.monotonic()
    for passindex in range(pass_offset+1,pass_offset+args.passes+1):
        accepted=0
        for start in range(0,len(eligible),args.batch):
            ix=eligible[start:start+args.batch];current=codes[ix].copy()
            outputs,jac=jacobians(render,pose,np.asarray(raw[2*ix+1]).copy(),current)
            updates=[]
            for j,frame in enumerate(ix):
                updates.append(solve_damped_least_squares(jac[j],targets[frame]-outputs[j],damping=.01,max_code_step=128).update)
            proposals=[]
            for factor in [.25,.5,1.,2.]:
                proposals.append(np.stack([quantize_int12_update(c,u*factor) for c,u in zip(current,updates)]))
            if args.neighbours:
                centres=proposals[2]
                rank=np.argsort(-np.abs(np.asarray(updates))*np.linalg.norm(jac,axis=1),axis=1)[:,:args.neighbours]
                for offsets in itertools.product((-1,0,1),repeat=args.neighbours):
                    if not any(offsets):continue
                    proposed=centres.copy()
                    for row in range(len(ix)):
                        proposed[row,rank[row]]+=np.asarray(offsets)
                    proposals.append(np.clip(proposed,-2048,2047))
            # All 12 +/-1 controls provide exact local fallback around current;
            # the larger GN proposals handle quantization-induced shifts.
            for dim in range(12 if args.coordinate else 0):
                for sign in [-1,1]:
                    p=current.copy();p[:,dim]=np.clip(p[:,dim]+sign,-2048,2047);proposals.append(p)
            best=np.asarray([selected_errors[lookup[int(v)]] for v in ix]);bestcodes=current.copy()
            # Pack proposals into the evaluator's 16-row CNN batches. Rendering
            # still retains the deployed 64/24-row carrier matrix dimensions.
            proposal_errors=errors(render,pose,raw,targets,np.concatenate(proposals),
                                   np.tile(ix,len(proposals)),ctx['selector'],batch=16)
            for proposed,e in zip(proposals,proposal_errors.reshape(len(proposals),len(ix))):
                better=e<best;best[better]=e[better];bestcodes[better]=proposed[better]
            accepted+=int(np.any(bestcodes!=current,axis=1).sum());codes[ix]=bestcodes
            for frame,e in zip(ix,best):selected_errors[lookup[int(frame)]]=e
            if start%32==0:print(f'pass {passindex} pairs {start+len(ix)}/{len(eligible)} accepted {accepted}',flush=True)
            save_array(out/'codes.npy',codes);save_array(out/'screen_errors.npy',selected_errors)
        # Recheck at canonical evaluator-sized batches before reporting.
        selected_errors=errors(render,pose,raw,targets,codes[ids],ids,ctx['selector'])
        data,rebuilt=archive(ctx,canonical,codes)
        (out/'archive.zip').write_bytes(data);(out/'canonical.bin').write_bytes(rebuilt)
        (out/f'pass{passindex}.zip').write_bytes(data)
        save_array(out/f'pass{passindex}-codes.npy',codes)
        save_array(out/f'pass{passindex}-errors.npy',selected_errors)
        row=dict(pass_index=passindex,pairs=len(ids),pose=float(selected_errors.mean()),archive_bytes=len(data),
                 score_delta=delta(selected_errors.mean(),len(data)),
                 accepted_rows=accepted,archive_sha256=sha(data),elapsed_seconds=time.monotonic()-started)
        row.update(coordinate=args.coordinate,neighbours=args.neighbours)
        history.append(row);print(json.dumps(row),flush=True)
        write_json(out/'report.json',dict(status='cached_full_pose_requires_full_inflate_evaluation' if len(ids)==600 else 'subset_screen_not_full_score',
                   source_candidate=str(folder),source_archive=ctx['source_archive'],source_archive_sha256=sha(ctx['data']),
                   indices=ids.tolist(),history=history,segmentation_distortion=candidate_seg,
                   master_frames=args.master_frames,seg_errors=args.seg_errors,
                   provenance_cache=str(OUT/'cache/manifest.json'),all_learned_data_charged=True))
        if accepted==0:break
    print('rescue complete',flush=True)
if __name__=='__main__':main()
