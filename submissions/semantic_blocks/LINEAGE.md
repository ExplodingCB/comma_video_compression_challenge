# Phase-2 provenance and review status

This local research candidate descends from codexblack's
[PR #135](https://github.com/commaai/comma_video_compression_challenge/pull/135),
`semantic-pose-HPAC_CPR1_polished`, source commit
`6dcf77164ccbdcc1e0e41c99312e65ace4bc1fb4`. The original trained archive
was 186,724 bytes, SHA-256
`12cf5d71a94065184f097c3e40dfe9f1db8402a1a76a80efc76a6956fe1e4004`.
The original architecture and much of its learned state are prior work.
The upstream MIT license and copyright notice are included in LICENSE.

That work builds on [PR #130](https://github.com/commaai/comma_video_compression_challenge/pull/130),
[PR #133](https://github.com/commaai/comma_video_compression_challenge/pull/133),
jas0xf's [PR #86](https://github.com/commaai/comma_video_compression_challenge/pull/86),
and EthanYangTW's [PR #67](https://github.com/commaai/comma_video_compression_challenge/pull/67)
and [PR #79](https://github.com/commaai/comma_video_compression_challenge/pull/79).

The model section comes from `results/phase2/pose/full-control/pass3.zip`.
Its input archive SHA-256 is `0afd7f640ebdaac9826fc2c2f4f88a4c4ef3efa2297e35679bda501bbb49e3e6`.

The original renderer, HPAC model, pose basis, and frame-zero selector
are inherited unchanged. The model-source experiment optimized and
serialized pose coefficients; the complete pose carrier is therefore
not byte-identical to the earlier submission.

The entropy tail comes from `results/phase2/entropy/candidates/margin50_4_raw_1/archive.zip`.
Its input archive SHA-256 is `9d0a14db2620d788a6a5007dd4b5e09db9610f996710e26f9a6062760a1e3db4`.
This replaces the earlier residual correction table and encoded token
stream with a serialized margin-conditioned correction table and its
newly arithmetic-coded stream. The encoded bytes change; the intended
decoded semantic token symbols are retained. Cached symbol replay and
independent causal GPU inflation are separate validation gates. The
CPU composition tool does not certify causal replay or a video score.
New table values, scale, margin descriptors, pose coefficients, and
the encoded stream are carried in the charged archive.

The two source archives must decode byte-identical HPAC model blobs.
Composition also checks compatible model/runtime code and preserves
the entropy source's ETP1 reader, adding only BLK2 format dispatch.

BLK2 stores independent blocks using raw bytes, Brotli or LZMA2 as
specified in recipe.json, with optional reversible byte-lane transposes.
All block boundaries and
transform descriptors are serialized. The model bytes are recovered
exactly; Brotli is a general-purpose library installed in a contained
dependency directory by the supplied bootstrap.

The phase-2 workflow includes pose-coefficient optimization, entropy
calibration/re-encoding, and lossless block packing. The final composition
step combines the exact supplied sections; it is not the entire
optimization pipeline. Reproduction uses the prior trained artifacts
and experiment checkpoints identified in provenance.json.

These phase-2 changes were prepared with Codex assistance. The earlier
186,151-byte PR #141 artifact and its participant explanation describe
the first version. This update retains the original attribution and
license; its new pose coefficients and entropy calibration are distinct
from that first version. This file records technical provenance.
