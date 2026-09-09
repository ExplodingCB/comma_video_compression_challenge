# Reproducing semantic_blocks v2

This packages the **existing phase-2 implementation** recovered from the original
workbench. It does not download the finished v2 submission. The target is
**185,704 bytes**, SHA-256
`640e9a8d618652ce4e0570cbcc92348d3f2457b8b737a13ab68cdbb9595fcf2f`.

## Scope

| Operation | Inputs | What runs | What it does not establish |
| --- | --- | --- | --- |
| `compress.sh compose` | Immutable pose and entropy intermediate archives | Unpack sections, check source hashes and HPAC equality, encode the fixed BLK2 recipe, round-trip, deterministic ZIP | Pose optimization, table fitting, causal decoding, or a video score |
| `compress.sh checkpoints` | v1 trained archive, saved pose coefficients, fitted table/screen, captured predictor rows, pinned ExperimentBook source | Serialize coefficients through CPS3/CAP1, reproduce the baseline RC64 stream, re-encode the selected table's 117,964,800 tokens and replay them, then BLK2 composition | Rerunning coefficient optimization or table fitting; training inherited models from video |
| Original optimization commands below | v1 or PR135 trained archive, original video and evaluator weights | Pose warm start and full recovery, predictor capture, calibration fit/screen/encode, optional packing search | Complete upstream training starting from only the original video; identical floating-point search trajectories on arbitrary hardware |

All learned coefficients, table entries, margin threshold and scale used by the
decoder are serialized in the charged archive. Checkpoints and predictor caches
are compression inputs only. The deployed decoder needs neither those caches nor
the original video or evaluator weights.

## Recovered files and attribution

`compression/recovered-source.json` maps each recovered file to its original
workbench path and SHA-256. Recovered files are byte-for-byte copies. The small
new wrappers restore paths, select the historical winning case, validate inputs,
and orchestrate the old algorithms; they do not implement a replacement codec.

| Existing implementation | Packaged location / role |
| --- | --- |
| `experiments/pose/common.py` | `compression/recovered/experiments/pose/common.py`: charged CPS3/CAP1 serialization and exact carrier rendering |
| `experiments/pose/rescue.py` | Same relative path under `compression/recovered/`: Gauss–Newton proposals, exact uint8 acceptance and coefficient checkpoints |
| `run_joint_screen.py`, `run_full_recovery.py`, `audit_completed.py` | Same pose directory: 96-pair control warm start, three full-600 passes and immutable checkpoint audit |
| `experiments/entropy/{capture,cache_hook,replay_baseline,fit,encode}.py` | Same relative paths: causal predictor capture, precision-8 replay gate, weighted L-BFGS-B fit and native arithmetic re-encoding |
| `experiments/lossless/{sweep,block_dp,refine,prepare_candidate}.py` | Same relative paths: original codec/layout/partition search and package preparation |
| `experiments/lossless/compress.py` | Also copied unchanged to `compression/blk2.py`; its `build_payload` and `zip_payload` are the active packer |
| `experiments/combinations/{compose,state_fingerprint,verify}.py` | Same relative paths: hybrid section selection, radius-32 refinement, decoded-state audits |

The actual final composition command used `full-control/pass3.zip` plus
`margin50_4_raw_1/archive.zip`, starting from the standalone lossless recipe and
`--refine-radius 32` (68 trials). The resulting fixed `recipe.json` is used by
the portable entry point, so ordinary rebuilding does not repeat the search.
`compression/lossless-recipe.json` retains the pre-composition recipe.

The renderer, HPAC model, pose basis and selector descend from **codexblack's
[PR135](https://github.com/commaai/comma_video_compression_challenge/pull/135)**,
source commit `6dcf77164ccbdcc1e0e41c99312e65ace4bc1fb4`. The earlier attribution
chain and comma.ai MIT copyright/license remain in [LINEAGE.md](LINEAGE.md) and
[LICENSE](LICENSE). The v1 interpreter overrides in `compression/baseline/`
preserve the original source needed by the experiment scripts.

Pose helpers and the **encoder** come from codexblack's separate
[ExperimentBook at f229b26735dffc53fdf1ac9987ac7c303298d028](https://github.com/codexblack/CommaVideoCompressionChallenge_ExperimentBook/tree/f229b26735dffc53fdf1ac9987ac7c303298d028).
That pinned checkout contains no LICENSE/COPYING file. It remains an external
source dependency; this package does not redistribute it or assert that the
decoder's MIT license covers it. `compression/reference-source.json` records the
commit, immutable source download and hashes of every helper file copied into a
temporary workbench. Brotli, SciPy, PyTorch and other dependencies retain their
own licenses and are installed as packages, not vendored binaries.

## Input locations and availability

Paths in this table are relative to the original research workspace or the
[published reproduction input bundle](https://github.com/ExplodingCB/comma_video_compression_challenge/releases/download/semantic-blocks-v2/reproduction-inputs.zip).
The checkpoints were recovered, not recreated from the finished v2 archive.

| Input | Bytes | SHA-256 |
| --- | ---: | --- |
| `results/phase2/pose/full-control/pass3.zip` | 186152 | `0afd7f640ebdaac9826fc2c2f4f88a4c4ef3efa2297e35679bda501bbb49e3e6` |
| `results/phase2/entropy/candidates/margin50_4_raw_1/archive.zip` | 185986 | `9d0a14db2620d788a6a5007dd4b5e09db9610f996710e26f9a6062760a1e3db4` |
| `results/phase2/pose/full-control/pass3-codes.npy` | 28928 | `32094037b87cae7010ff1fb95766490593085eb599403ce24645afd1b4e34838` |
| `results/phase2/entropy/candidates/margin50_4_raw_1/table.json` | 3663 | `ffa6d1e0862851475178b206f1379ff388f70fd82ce6f4e41f17f5b4b62e4bde` |

`compression/inputs.json` lists **all** input paths, sizes, hashes, public URLs
where known, and staging destinations. This includes `screen.json` and the
1,415,577,600 bytes of `base_logits.i16`, `feature.u8`, `symbols.u8` used for
checkpoint re-encoding. Those rows can be regenerated with GPU `capture.py`.
The release bundle contains all eleven files in `compression/inputs.json`,
including the full predictor cache, plus attribution and the inherited license.
The finished v2 submission is not included. The manifest records the bundle URL,
size and SHA-256; the release also provides `reproduction-inputs.sha256`.
The original local copy remains at `results/pipeline-recovery/inputs/`.
Checkpoint mode verifies every staged file and does not stage either of the two
already-encoded phase-2 archives into its working tree.

Public starting artifacts:

- [PR135 trained archive](https://github.com/codexblack/comma_video_compression_challenge/releases/download/semantic-pose-HPAC_CPR1_polished-f26/archive.zip): 186724 bytes, SHA-256 `12cf5d71a94065184f097c3e40dfe9f1db8402a1a76a80efc76a6956fe1e4004`.
- [v1 trained archive](https://github.com/ExplodingCB/comma_video_compression_challenge/releases/download/semantic-blocks-v1/archive.zip): 186151 bytes, SHA-256 `d8ae5a9f4c3205ff2f2d823943afd3d25f8eb5cf9fd1e148c1159175e8f80c48`. Alternatively reproduce it from PR135 with the recovered baseline compressor.
- [Pinned ExperimentBook source export](https://github.com/codexblack/CommaVideoCompressionChallenge_ExperimentBook/archive/f229b26735dffc53fdf1ac9987ac7c303298d028.tar.gz). Extract it and pass its root as `--reference-source`; per-file hashes are checked, independent of tar metadata.

Original evaluation inputs are Git LFS assets from challenge commit
`db52c5a9f05d5298e314fb0fa130290c4a350c4e`:

- [videos/0.mkv](https://media.githubusercontent.com/media/commaai/comma_video_compression_challenge/db52c5a9f05d5298e314fb0fa130290c4a350c4e/videos/0.mkv), 37545489 bytes, SHA-256 `2611f5f3e186f3529777749f97bd4cce3a208d6b3559e137bd45d256980d2fa9`.
- [models/posenet.safetensors](https://media.githubusercontent.com/media/commaai/comma_video_compression_challenge/db52c5a9f05d5298e314fb0fa130290c4a350c4e/models/posenet.safetensors), SHA-256 `0f3a0874c5c387f990d7b88bd1d7e1f6de35d98b45f2a289989db2c77b9b6576`.
- [models/segnet.safetensors](https://media.githubusercontent.com/media/commaai/comma_video_compression_challenge/db52c5a9f05d5298e314fb0fa130290c4a350c4e/models/segnet.safetensors), SHA-256 `68956e328d4c5d875389a1a444870e6bac1c052c9986123827af95c07c6991b6`.

Use the exact MKV for matching the recorded byte-rate score; remuxing the source
HEVC can change container bytes. Git LFS downloads are needed, not pointer files.

## Dependencies

- Composition: Python 3.11 with standard-library `lzma`/`zipfile`, Brotli **1.2.0**.
  `python -m pip install -r compression/requirements-compose.txt` in a new venv.
  `compress.sh` honors `PYTHON`; it does not install packages or download data.
- Checkpoint replay: Linux/WSL, NumPy, SciPy **1.16.2**, PyTorch and the inherited
  renderer imports (`einops`), plus a C compiler (`cc`, or executable `$CC`).
  It is CPU work when predictor rows are supplied, although the helpers import
  Torch. Use the pinned challenge environment to avoid mismatched numerics.
- Capture, optimization, inflation, evaluation: CUDA-capable GPU. Recorded local
  stack: Python **3.11.15**, Torch **2.9.0+cu128**, DALI **1.52.0**, RTX 5080,
  IEEE float32/TF32 disabled. The challenge's pinned `uv.lock` supplies NumPy and
  all metric/renderer dependencies. For example, in a separate challenge checkout:
  `uv sync --project "$CHALLENGE" --frozen --group cu128 --python 3.11`, then
  `source "$CHALLENGE/.venv/bin/activate"` and
  `python -m pip install -r "$SUB/compression/requirements-experiments.txt"`
  (or `uv pip install` if that environment has no pip).
- Optional broad lossless search imports pyppmd **1.3.1** and zstandard **0.25.0**.
  Neither is needed by `compose`, checkpoint replay, or the final decoder.
- Allow at least 12 GB scratch space for copied predictor rows and raw videos.
  On this WSL host `DALI_DISABLE_NVML=1` is needed for DALI affinity handling.
  The original compiler wrapper used Zig **0.14.1**; a working native C compiler
  is sufficient for the encoder ABI.

## Rebuild in a new directory

Set `SUB` to this submission directory and `INPUTS` to the recovered bundle root.
These commands use only provided inputs and code. Both reject existing outputs;
neither writes to the published `archive.zip`.

Download and verify the input bundle, then extract it to a new directory:

```bash
curl -fL -O https://github.com/ExplodingCB/comma_video_compression_challenge/releases/download/semantic-blocks-v2/reproduction-inputs.zip
curl -fL -O https://github.com/ExplodingCB/comma_video_compression_challenge/releases/download/semantic-blocks-v2/reproduction-inputs.sha256
sha256sum -c reproduction-inputs.sha256
mkdir reproduction-inputs
python -m zipfile -e reproduction-inputs.zip reproduction-inputs
```

```bash
SUB=/absolute/path/to/submissions/semantic_blocks
INPUTS=/absolute/path/to/reproduction-inputs
RUN=$(mktemp -d)
bash "$SUB/compress.sh" compose \
  --model-source "$INPUTS/results/phase2/pose/full-control/pass3.zip" \
  --entropy-source "$INPUTS/results/phase2/entropy/candidates/margin50_4_raw_1/archive.zip" \
  --output "$RUN/composed/archive.zip"

bash "$SUB/compress.sh" checkpoints \
  --inputs "$INPUTS" \
  --reference-source /absolute/path/to/pinned/ExperimentBook \
  --work-dir "$RUN/work"
cmp "$RUN/composed/archive.zip" "$RUN/work/rebuilt/archive.zip"
```

Each archive gets an adjacent `.rebuild.json` receipt with exact source, model,
tail and output hashes. Any target mismatch exits unsuccessfully and retains
the new archive and diagnostic receipt. A composition receipt never claims a
fresh optimization, causal decode, or metric run. Checkpoint replay additionally
leaves `results/phase2/entropy/{baseline-replay,encoded}.json` in the workbench.

## Rerun phase-2 optimization, starting with trained PR135 state

These are the recovered algorithm commands, not a claim of newly rerun training.
Use a **different, fresh workbench** from the checkpoint replay above. Prepare it
without `--inputs`, so no saved fit or optimized pose state is reused:

```bash
python "$SUB/compression/prepare_workbench.py" \
  --reference-source /absolute/path/to/pinned/ExperimentBook --work-dir "$RUN/search"
cd "$RUN/search"
# Place the hash-verified PR135 archive at artifacts/pr135/archive.zip.
bash submissions/semantic_blocks/compress.sh \
  --source artifacts/pr135/archive.zip --output submissions/semantic_blocks/archive.zip
# Place the pinned challenge checkout, video and weights at ./challenge.
mkdir submissions/semantic_blocks/archive
python -m zipfile -e submissions/semantic_blocks/archive.zip submissions/semantic_blocks/archive
bash submissions/semantic_blocks/inflate.sh submissions/semantic_blocks/archive \
  submissions/semantic_blocks/inflated challenge/public_test_video_names.txt
python experiments/pose/build_cache.py
python optimize_control.py
cc -O3 -shared -fPIC references/pr135-experiments/src/cpr1_sub4/entropy/rc64_backend.c \
  -o results/phase2/entropy/librc64.so -lm
cc -O3 -shared -fPIC submissions/semantic_blocks/runtime/entropy/rc64_backend.c \
  -o artifacts/pr135/rc64_backend.so
python experiments/entropy/capture.py
python experiments/entropy/replay_baseline.py
python experiments/entropy/fit.py fit_weighted
python experiments/entropy/fit.py screen
python experiments/entropy/encode.py
python experiments/combinations/compose.py \
  --source results/phase2/pose/full-control/pass3.zip \
  --runtime-source submissions/semantic_blocks \
  --entropy-source results/phase2/entropy/candidates/margin50_4_raw_1/archive.zip \
  --name recovered_control --refine-radius 32
```

The winning control schedule is three 96-pair warm-start passes followed by
three full-600 passes, batch 8, two neighbour dimensions in the full passes,
and no coordinate sweep. Only pairs with the identity stored selector are
eligible for the solver's updates. `optimize_control.py` extracts those calls
from `run_joint_screen.py` without launching unrelated rejected variants.
The calibrated table used 80 L-BFGS-B iterations and stopped at the iteration
limit (`optimizer_success=false`), not at a proven optimum. Every acceptance
and final code path remains the recovered implementation.

This reuses the original lossless search's saved recipe. To rerun that search,
first decode the v1 payload with `runtime.block_container.decode_model_blocks`
and save `results/lossless/{models,tail}.bin`. Run `experiments/lossless/refine.py`
and `prepare_candidate.py` before composition. `sweep.py`, `block_dp.py` and
`split_screen.py` are included for inspecting the broader search, but are not
prerequisites for that refinement path. Historical result counts and rejected
branches are recorded in the recovered READMEs; they are not fresh results.

Exact checkpoint replay does **not** recover the entire training lineage of
PR135/PR130/PR133 or train the renderer/HPAC/basis from only `0.mkv`. That larger
upstream optimization, its complete input/checkpoint chain, and cross-GPU search
determinism have not been established here. The published input bundle provides
the saved phase-2 state for checkpoint replay; it does not establish that larger
upstream training lineage. No missing algorithm in the local winning phase-2
path was found.

## Fresh inflation and evaluation

```bash
python "$SUB/compression/prepare_inflation.py" \
  --archive-file "$RUN/work/rebuilt/archive.zip" --output-dir "$RUN/eval"
bash "$RUN/eval/inflate.sh" "$RUN/eval/archive" "$RUN/eval/inflated" \
  "$CHALLENGE/public_test_video_names.txt"
# The unchanged local evaluator wrapper resolves ./challenge under the workbench.
# Copy or check out the pinned challenge there, with its weights and original video.
python "$RUN/work/scripts/evaluate_exact.py" \
  --submission-dir "$RUN/eval" --report "$RUN/eval/evaluation.json"
```

The original decoder resolves `archive.zip` beside its code and checks it against
the extracted `p` supplied by the challenge. The preparation helper creates that
layout without editing the decoder or using a published output directory.
Expected raw output is 3,662,409,600 bytes. Compare the newly measured tokens,
raw hash and metrics to the historical record; do not silently reuse an old raw
video. `compression/recorded/` preserves the **2026-09-07** results, and
`VALIDATION.md` distinguishes the new recovery run. Original `provenance.json`
describes composition before scoring; its historical "not scored" fields do not
override later evaluation receipts. Local GPU scores are not official standings.
