"""Write phase-2 provenance without inheriting the earlier submission's status."""
import argparse
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]


def project_relative(value):
    value=str(value).replace('\\','/')
    for prefix in ('D:/Projects/video-compression/','/mnt/d/Projects/video-compression/'):
        if value.startswith(prefix):return value[len(prefix):]
    return value


def write_lineage(target,report):
    source=project_relative(report['source'])
    is_b2=any(part in source for part in (
        'renderer/archives/all_metadata_b2/', 'pose/refined-all_metadata_b2/',
        'pose/rescue-all_metadata_b2/', 'pose/full-all_metadata_b2/'))
    has_pose='/pose/' in '/'+source
    is_control=any(part in source for part in ('full-control/','rescue-control/'))
    if is_b2:
        model_changes=(
            'The renderer architecture and four-bit weight codes are inherited. Its\n'
            'binary16 metadata was rounded to a coarser grid with two low mantissa\n'
            'bits removed; this changes decoded renderer values. The HPAC model,\n'
            'pose basis, and frame-zero selector are inherited unchanged.\n')
        if has_pose:model_changes+='The model-source experiment also optimized and serialized pose coefficients.\n'
    elif is_control:
        model_changes=(
            'The original renderer, HPAC model, pose basis, and frame-zero selector\n'
            'are inherited unchanged. The model-source experiment optimized and\n'
            'serialized pose coefficients; the complete pose carrier is therefore\n'
            'not byte-identical to the earlier submission.\n')
    elif 'submissions/semantic_blocks/' in source:
        model_changes='The model section is inherited byte-for-byte from the earlier submission.\n'
    else:
        model_changes=(
            'The model section is preserved byte-for-byte from the stated model\n'
            'source, including that experiment\'s learned-state changes. Refer to\n'
            'its experiment report for the scope of those changes.\n')
    hybrid=bool(report.get('entropy_source'))
    if hybrid:
        entropy_source=project_relative(report['entropy_source'])
        entropy_changes=(
            f'The entropy tail comes from `{entropy_source}`.\n'
            f'Its input archive SHA-256 is `{report["entropy_source_archive_sha256"]}`.\n'
            'This replaces the earlier residual correction table and encoded token\n'
            'stream with a serialized margin-conditioned correction table and its\n'
            'newly arithmetic-coded stream. The encoded bytes change; the intended\n'
            'decoded semantic token symbols are retained. Cached symbol replay and\n'
            'independent causal GPU inflation are separate validation gates. The\n'
            'CPU composition tool does not certify causal replay or a video score.\n'
            'New table values, scale, margin descriptors, pose coefficients, and\n'
            'the encoded stream are carried in the charged archive.\n\n'
            'The two source archives must decode byte-identical HPAC model blobs.\n'
            'Composition also checks compatible model/runtime code and preserves\n'
            'the entropy source\'s ETP1 reader, adding only BLK2 format dispatch.\n')
        workflow=('The phase-2 workflow includes pose-coefficient optimization, entropy\n'
                  'calibration/re-encoding, and lossless block packing. The final composition\n'
                  'step combines the exact supplied sections; it is not the entire\n'
                  'optimization pipeline. Reproduction uses the prior trained artifacts\n'
                  'and experiment checkpoints identified in provenance.json.\n')
    else:
        entropy_changes=('The entire entropy tail is preserved from the model source, including\n'
                         'its correction table and encoded token stream. This statement concerns\n'
                         'the immediate source, not necessarily the earlier PR #141 archive.\n')
        workflow=('The composition step repacks exact input bytes; it does not rerun the\n'
                  'model-source experiment. Reproduction requires the prior trained archive\n'
                  'or checkpoint identified in provenance.json.\n')
    text=(
        '# Phase-2 provenance and review status\n\n'
        'This local research candidate descends from codexblack\'s\n'
        '[PR #135](https://github.com/commaai/comma_video_compression_challenge/pull/135),\n'
        '`semantic-pose-HPAC_CPR1_polished`, source commit\n'
        '`6dcf77164ccbdcc1e0e41c99312e65ace4bc1fb4`. The original trained archive\n'
        'was 186,724 bytes, SHA-256\n'
        '`12cf5d71a94065184f097c3e40dfe9f1db8402a1a76a80efc76a6956fe1e4004`.\n'
        'The original architecture and much of its learned state are prior work.\n'
        'The upstream MIT license and copyright notice are included in LICENSE.\n\n'
        'That work builds on [PR #130](https://github.com/commaai/comma_video_compression_challenge/pull/130),\n'
        '[PR #133](https://github.com/commaai/comma_video_compression_challenge/pull/133),\n'
        'jas0xf\'s [PR #86](https://github.com/commaai/comma_video_compression_challenge/pull/86),\n'
        'and EthanYangTW\'s [PR #67](https://github.com/commaai/comma_video_compression_challenge/pull/67)\n'
        'and [PR #79](https://github.com/commaai/comma_video_compression_challenge/pull/79).\n\n'
        f'The model section comes from `{source}`.\n'
        f'Its input archive SHA-256 is `{report["source_archive_sha256"]}`.\n\n'
        +model_changes+'\n'+entropy_changes+'\n'
        'BLK2 stores independent blocks using raw bytes, Brotli or LZMA2 as\n'
        'specified in recipe.json, with optional reversible byte-lane transposes.\n'
        'All block boundaries and\n'
        'transform descriptors are serialized. The model bytes are recovered\n'
        'exactly; Brotli is a general-purpose library installed in a contained\n'
        'dependency directory by the supplied bootstrap.\n\n'
        +workflow+'\n'
        'These phase-2 changes were made with Codex assistance and have not yet\n'
        'been reviewed by the participant or submitted. The participant\'s earlier\n'
        'review and public explanation applied separately to the 186,151-byte\n'
        '[PR #141](https://github.com/commaai/comma_video_compression_challenge/pull/141)\n'
        'artifact. This package does not inherit that review or submission status.\n'
        'This file is an internal technical record, not a public PR description.\n'
        'The challenge requires participants to read and write most submitted\n'
        'code and provide their own public explanation.\n')
    (target/'LINEAGE.md').write_text(text,newline='\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate',type=Path,action='append',required=True)
    args=parser.parse_args()
    for target in args.candidate:
        target=target.resolve()
        if ROOT not in target.parents:raise ValueError('Candidate must be inside this project')
        write_lineage(target,json.loads((target/'provenance.json').read_text()))
        print(str(target/'LINEAGE.md'))


if __name__=='__main__':main()
