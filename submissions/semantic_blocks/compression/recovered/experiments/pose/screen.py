"""Screen each extra int4 atom and linear coefficient recovery on fixed pairs."""
from common import *
import time

def main():
    import torch
    ctx=context();rebuilt,_=archive(ctx,ctx['canonical'],ctx['codes'])
    assert rebuilt==ctx['data'],'baseline rebuild must be byte identical'
    device=setup_gpu();pose=load_pose(device);raw=raw_stream()
    targets=np.load(OUT/'cache/pose_targets.npy');base_errors=np.load(OUT/'cache/pose_errors.npy')
    ids=np.linspace(0,599,96,dtype=int)
    original=renderer(ctx['canonical'],device)
    # Verify inherited direct carrier math including sparse selectors against
    # the submitted decoder's existing output on the entire screening set.
    for offset in range(0,len(ids),16):
        ix=ids[offset:offset+16]
        got=render_selected(original,ctx['codes'][ix],ix,ctx['selector'])
        want=np.asarray(raw[2*ix])
        assert np.array_equal(got,want),f'carrier render parity failure ids={ix.tolist()} different={int((got!=want).sum())}'
    check=errors(original,pose,raw,targets,ctx['codes'][ids],ids,ctx['selector'])
    original_mean=float(check.mean())
    print(f'baseline96 {original_mean:.12g} cached {base_errors[ids].mean():.12g}',flush=True)
    rows=[];started=time.monotonic()
    for atom in [0,1,3,4,6,7,8,10,11]:
        can=requantize_cpr1_basis(ctx['canonical'],atoms=(atom,)).carrier
        render=renderer(can,device)
        # Least squares fit in normalized expanded-basis space, including
        # exact per-dimension scale conversion back to signed int12 codes.
        with torch.no_grad():
            old=original.basis.reshape(12,-1).double();new=render.basis.reshape(12,-1).double()
            mapping=torch.linalg.solve(new@new.T,new@old.T).cpu().numpy()
        old_values=ctx['codes']*original.coefficient_scales
        linear=np.clip(np.rint((old_values@mapping.T)/render.coefficient_scales),-2048,2047).astype(np.int32)
        for seed,codes in [('unchanged',ctx['codes']),('linear',linear)]:
            data,can2=archive(ctx,can,codes)
            err=errors(render,pose,raw,targets,codes[ids],ids,ctx['selector'])
            folder=OUT/f'atom{atom}-{seed}';folder.mkdir(exist_ok=True)
            (folder/'archive.zip').write_bytes(data);(folder/'canonical.bin').write_bytes(can2)
            np.save(folder/'codes.npy',codes);np.save(folder/'screen_errors.npy',err)
            row=dict(atom=atom,seed=seed,archive_bytes=len(data),saved_bytes=len(ctx['data'])-len(data),pairs=len(ids),
                     pose=float(err.mean()),baseline_pose=original_mean,score_delta=float(score(err.mean(),len(data))-score(original_mean,len(ctx['data']))),
                     archive_sha256=sha(data),folder=str(folder))
            rows.append(row);print(json.dumps(row),flush=True)
            write_json(OUT/'screen.json',dict(status='subset_screen_not_full_score',indices=ids.tolist(),rows=rows,elapsed_seconds=time.monotonic()-started))
        del render
    print('screen complete',flush=True)
if __name__=='__main__':main()
