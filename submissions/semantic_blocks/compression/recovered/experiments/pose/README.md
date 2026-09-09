# Pose precision and coefficient experiments

These scripts keep the submitted archive and all upstream sources unchanged.
They import the attributed PR135 experiment helpers from
`references/pr135-experiments`, then adapt them to the submitted BLK1 container
and this machine's exact local CUDA/DALI outputs. Learned changes remain in the
complete, charged archive. No optimized coefficients are embedded in source.

The full 600 local DALI cache reproduces the submitted baseline metrics exactly:
PoseNet 6.8861736508551985e-06 and SegNet 0.0002964019950013608. Its manifest records
hashes, versions, hardware, batch size, seed, and IEEE precision policy.

`screen.py` tests each of the nine remaining five-bit basis atoms at int4, both
with unchanged coefficients and with a least-squares fit of the old normalized
basis in the new basis. It evaluates 96 evenly spaced pairs and serializes every
complete candidate archive. These subset scores are screening evidence, never
leaderboard claims. All 18 initial candidates lose despite 288–395 byte savings.

`rescue.py` computes six-output, twelve-coefficient Jacobians through a
straight-through rounded renderer and differentiable PoseNet preprocessing.
Damped Gauss–Newton steps propose signed-int12 coefficient updates. Four line
search factors are evaluated with the unmodified challenge preprocessing and
actual uint8 rendered camera frames. Optional coordinate descent evaluates all
24 single-coordinate +/-1 moves. Every completed pass is charged by rebuilding
the full archive, and rechecked at the evaluator's batch size.

An exactness guard caught a real inherited-helper limitation: deployed carrier
rendering uses matrix batches 64, with 24 in its final batch. Small arbitrary
matrix batches can round differently at pixel thresholds on this GPU. The
local renderer preserves those matrix dimensions, applies the stored sparse
selector, and passes byte-for-byte parity against the submitted raw video on
the screening pairs. Proposal Jacobians need not have exact forward parity;
only the deployed uint8 path is used for candidate acceptance and scoring.

`run_joint_screen.py` additionally tests coefficient recovery after the renderer
agent's all_metadata_b2 quantization. It uses that candidate's newly rendered
frame1 cache, computes all 600 SegNet errors, and preserves its charged metadata
while changing only the carrier coefficients. It also runs an unchanged-basis
control and a recovered extra-atom candidate, separating the two effects.

Run from WSL in this project after sourcing `scripts/env.sh`:

```bash
python experiments/pose/build_cache.py
python experiments/pose/screen.py
python experiments/pose/run_joint_screen.py
python experiments/pose/refine_screen.py
python experiments/pose/run_full_recovery.py --case control --passes 3
python experiments/pose/run_full_recovery.py --case all_metadata_b2 --passes 3
```

`run_full_recovery.py` checks all cached target hashes, verifies that the seed
archive decodes to its saved canonical carrier and coefficients, and recovers
all 600 pairs. Repeating it resumes from the most recent coefficient checkpoint
and adds the requested number of passes. Pass numbering and earlier history are
preserved. Coefficient arrays and JSON reports are replaced atomically so a
process interruption during a write leaves the preceding complete file usable.
`--coordinate` additionally tests the 24 single-coordinate integer neighbours.

Audit a completed pass before consuming it in another experiment:

```bash
python experiments/pose/audit_completed.py --folder results/phase2/pose/full-control --pass-index 3
```

The audit uses immutable `passN.zip`, `passN-codes.npy`, and `passN-errors.npy`,
rather than the potentially mid-pass `codes.npy`. It checks report hashes,
decoded coefficients, the canonical carrier, reconstruction from the declared
source and seed basis, and preservation of the entropy tail and pixel selector.

The resumed 96-pair refinement screens produced the following results before
any additional lossless container composition:

| Candidate | Complete archive bytes | Pose error | Score change vs submitted baseline |
| --- | ---: | ---: | ---: |
| Extra int4 atom 0, six recovery passes total | 185770 | 0.0000091330093 | +0.000123854 |
| B2 renderer metadata, six recovery passes total | 185423 | 0.0000098733126 | +0.000309661 |
| All 12 atoms int4, four recovery passes | 182899 | 0.0000203678105 | +0.002927079 |

Lower is better. These screens reject the aggressively quantized basis variants
at this stage. The metadata candidate receives a separate full-600 recovery run
because its larger additional lossless container savings can offset the smaller
remaining distortion penalty. These are fitted screening pairs, not held-out
generalization claims or official challenge scores.

Three full-600 recovery passes after the screening warm starts completed after
the restart. Both immutable pass-3 archives passed `audit_completed.py`:

| Candidate | Archive bytes before later composition | Full cached PoseNet error | Cached score change |
| --- | ---: | ---: | ---: |
| Submitted baseline | 186151 | 0.0000068861737 | 0 |
| Original basis and renderer, coefficient recovery | 186152 | 0.0000067090702 | -0.000106744 |
| B2 metadata renderer, coefficient recovery | 185426 | 0.0000087515787 | +0.000582419 |

The control changes 1101 of the 7200 integer coefficients, spanning 103 of 600
pairs. Those 103 pairs improve; no pair's cached PoseNet error regresses. Its
basis, renderer, token stream and pixel selector remain unchanged. The complete
coefficient stream costs one additional byte in the original BLK1 container.

The B2 full-pair error progresses from 0.0000280157205 to 0.0000107183239,
0.0000089536103 and 0.0000087515787. Its improvement shrinks by almost ninefold
between the last two steps. It remains about 0.0000009 above the approximate
pose error needed to beat the fully composed control, so this bounded recovery
branch stops here. This is evidence against this specific candidate and solver
budget, not a claim that every B2 recovery method must fail.

Use `results/phase2/pose/full-control/pass3.zip` or
`results/phase2/pose/full-all_metadata_b2/pass3.zip` for downstream work, with the
adjacent `pass3-audit.json` receipts. The B2 receipt also hashes its complete
master-frame cache and verifies the SegNet error cache's source-archive binding.
Full decoder runs and the original evaluator remain necessary before treating
the cached figures as validated end-to-end scores. Additional container and
entropy changes have their own composition receipts outside this directory.

Evidence lives under `results/phase2/pose/`. Each recovery pass saves its archive,
coefficients, per-pair errors, and report. A successful subset result must still
be optimized/evaluated across all 600 pairs, inflated with the submission
decoder, and evaluated by the complete local evaluator before promotion.
