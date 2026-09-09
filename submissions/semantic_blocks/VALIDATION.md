# Recovery validation — 2026-09-08

The new recovery run passed. Both direct composition and rebuilding the saved
pose/entropy checkpoints produced **185,704 bytes**, SHA-256
`640e9a8d618652ce4e0570cbcc92348d3f2457b8b737a13ab68cdbb9595fcf2f`. The rebuilt bytes also match the previously verified public
download at `results/submission-v2/download-verified.zip` byte for byte.
No archive discrepancy was found. The published archive and both original
phase-2 intermediate archives remain unchanged.

## Newly performed

- Rebuilt v1 from the attributed PR135 archive; exact 186151-byte v1 hash.
- Serialized `pass3-codes.npy` using the recovered CPS3/CAP1 implementation;
  exact `full-control/pass3.zip` hash.
- Replayed all 117964800 baseline probability rows with deployed precision 8;
  exact original RC64 stream and probability hash.
- Re-encoded the saved calibrated table and all 117964800 symbols using the
  recovered native encoder, then checked cached-symbol replay; exact
  `margin50_4_raw_1/archive.zip` hash.
- Ran the original radius-32 composition refinement; 68 trials recovered
  the published five-block recipe. Fixed-recipe composition also matched.
- Ran fresh independent causal GPU inflation with the existing decoder,
  producing 3,662,409,600 bytes in
  239.6 seconds. Tokens SHA-256:
  `c5c7671d037b6912980c57929a5b6d789d250ee6a93e3b0a6018cf9f63e32ece`.
- Evaluated all 600 pairs with the original challenge metric
  code, weights and video. Evaluator/input hashes equal the earlier record.
- Four integration tests passed: target and decoded-state equality, incorrect
  input rejection, protection of existing/published outputs, original packing
  search. Source hash, Python parsing and Bash syntax checks also passed.

| Measurement | New recovery run | September 7 record |
| --- | --- | --- |
| Raw SHA-256 | `d6f2a492b2ca6df4f6a8f29d0f1ccd6a3872e4a0ed2547970286adecd0504e36` | Identical |
| PoseNet distortion | 6.709069111821009e-06 | Identical |
| SegNet distortion | 0.0002964019950013608 | Identical |
| Score | 0.16148376127095507 | Identical |
| Evaluation seconds, including hashing | 39.9 | 39.8 |

The GPU was an RTX 5080 with Torch 2.9.0+cu128, DALI 1.52.0 and
IEEE float32/TF32 disabled. These are new **local** measurements, not an official
leaderboard result. The scratch directories were new; the existing project
Python/dependency environment was reused. Predictor rows were copied from saved
state and hash-checked. Pose optimization, calibration fitting and upstream
training from the original video were **not rerun**.

## Receipts and harness correction

New receipts are under `compression/validation/`; the earlier measurements are
under `compression/recorded/`. Original `provenance.json` remains the historical
composition receipt. `baseline-replay.json` and `encoded.json` document fresh
CPU work; the latter's causal status is from before the separate fresh GPU gate.
`validation-2026-09-08.json` records that later gate and the comparison.

The first inflation attempt exposed a harness layout error: the existing decoder
expects `archive.zip` beside its code and an extracted `archive/p`. It failed
before decoding. `compression/prepare_inflation.py` now prepares that layout.
The first Linux `/tmp` run directory also disappeared between commands, so the
complete run was repeated in a fresh temporary directory inside the project.
Neither issue changed compressed bytes or decoder algorithms.

Retained local run: `/mnt/d/Projects/video-compression/results/pipeline-recovery/clean-SfQlGg`. The compact rebuilt archive and logs are also in
the original workspace's `results/pipeline-recovery/`. `REPRODUCING.md` gives
portable commands and the limits of the published recovered checkpoint bundle.
