"""Audit one immutable completed pose checkpoint before downstream packaging."""
from common import *
import argparse


def audit(folder, pass_index=None):
    folder=Path(folder).resolve()
    if ROOT not in folder.parents:
        raise ValueError('Checkpoint must be inside this project')
    report=json.loads((folder/'report.json').read_text())
    if pass_index is None:
        pass_index=report['history'][-1]['pass_index']
    rows=[row for row in report['history'] if row['pass_index']==pass_index]
    assert len(rows)==1 and pass_index>0, 'Select exactly one completed pass'
    row=rows[0]
    assert report['indices']==list(range(600)) and row['pairs']==600
    archive_path=folder/f'pass{pass_index}.zip'
    codes_path=folder/f'pass{pass_index}-codes.npy'
    errors_path=folder/f'pass{pass_index}-errors.npy'
    data=archive_path.read_bytes()
    assert sha(data)==row['archive_sha256']
    assert len(data)==row['archive_bytes']
    codes=np.load(codes_path,allow_pickle=False)
    errors_=np.load(errors_path,allow_pickle=False)
    assert codes.shape==(600,12) and np.issubdtype(codes.dtype,np.integer)
    assert errors_.shape==(600,) and np.all(np.isfinite(errors_)) and np.all(errors_>=0)
    assert float(errors_.mean())==row['pose']
    source=context(report['source_archive'])
    assert sha(source['data'])==report['source_archive_sha256']
    decoded=context(archive_path)
    assert np.array_equal(decoded['codes'],codes), 'Serialized coefficient mismatch'
    seed=Path(report['source_candidate'])/'canonical.bin'
    assert ROOT in seed.resolve().parents
    rebuilt,canonical=archive(source,seed.read_bytes(),codes)
    assert rebuilt==data, 'Completed archive differs from declared source and coefficients'
    assert canonical==decoded['canonical'], 'Canonical carrier mismatch'
    assert source['tail']==decoded['tail'] and source['selector']==decoded['selector']
    coefficients_only=archive(source,source['canonical'],codes)[0]==data
    master_provenance=None
    if report.get('master_frames'):
        masters=Path(report['master_frames']).resolve()
        seg_errors=Path(report['seg_errors']).resolve()
        assert ROOT in masters.parents and ROOT in seg_errors.parents
        metadata=json.loads(seg_errors.with_suffix('.json').read_text())
        assert metadata['archive_sha256']==sha(source['data'])
        assert masters.stat().st_size==600*874*1164*3
        actual_master_sha=file_sha(masters)
        actual_seg_sha=file_sha(seg_errors)
        assert actual_master_sha==metadata['masters_sha256']
        assert actual_seg_sha==metadata['error_sha256']
        assert float(np.load(seg_errors,allow_pickle=False).mean())==report['segmentation_distortion']
        master_provenance=dict(path=str(masters),sha256=actual_master_sha,
                               segmentation_errors=str(seg_errors),segmentation_errors_sha256=actual_seg_sha,
                               source_archive_sha256=metadata['archive_sha256'])
    result=dict(status='passed_completed_600_checkpoint_audit_requires_full_inflate',
                pass_index=pass_index,pairs=600,archive=str(archive_path),
                archive_sha256=sha(data),archive_bytes=len(data),
                codes_path=str(codes_path),codes_sha256=file_sha(codes_path),
                errors_sha256=file_sha(errors_path),canonical_sha256=sha(canonical),
                source_archive=source['source_archive'],source_archive_sha256=sha(source['data']),
                seed_canonical=str(seed),seed_canonical_sha256=file_sha(seed),
                cached_pose_distortion=row['pose'],
                cached_segmentation_distortion=report['segmentation_distortion'],
                cached_score_delta=row['score_delta'],
                learned_source_preserved_outside_declared_seed_carrier=True,
                source_basis_preserved=coefficients_only,
                source_changes_are_coefficient_only=coefficients_only,
                entropy_tail_identical=True,selector_identical=True,
                changed_master_cache=master_provenance,
                final_decoder_and_evaluator_status='pending')
    write_json(folder/f'pass{pass_index}-audit.json',result)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--folder',required=True)
    parser.add_argument('--pass-index',type=int)
    args=parser.parse_args()
    print(json.dumps(audit(args.folder,args.pass_index),indent=2))


if __name__=='__main__':main()
