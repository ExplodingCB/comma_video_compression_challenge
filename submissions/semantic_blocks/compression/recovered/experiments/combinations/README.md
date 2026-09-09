This workstream composes a candidate's exact model bytes and entropy tail with
the best additional lossless block recipe. By default both come from one
candidate. The optional entropy source supplies a different complete table
and token stream when its HPAC predictor is byte-identical. Composition does
not rerun training or inherit a score from another archive. The final composed
archive still needs full inflation and evaluation.

All paths and task-owned dependencies remain inside this project. Run the full
composition and learned-state audits in the existing WSL environment because
the inherited reader imports Torch, even though these checks do not use a GPU:

```bash
cd /mnt/d/Projects/video-compression
source scripts/env.sh
python experiments/combinations/compose.py \
  --source results/phase2/renderer/archives/all_metadata_b2/archive.zip \
  --name example_b2_lossless \
  --refine-radius 32
```

Each output name must be new. Existing archives are never overwritten. A source
directory containing its own runtime is selected automatically. Otherwise the
submitted decoder is used. Pass `--runtime-source path/to/actual/candidate` for
an archive stored outside its decoder directory. New formats such as an ETP1
entropy tail require their actual reader; the submitted reader is rejected.

The tool copies that reader and its companion runtime files, then adds BLK2
container support and the project-contained Brotli dependency bootstrap. It
checks that no other inherited runtime algorithm changed. The source and
result must recover equal model bytes, entropy bytes, renderer tensors,
carrier/HPAC blobs, and residual-table values. The deterministic output ZIP,
exact input hash, chosen block recipe, and fingerprints are saved together.

To combine an immutable full-600 pose checkpoint with the margin-conditioned
entropy experiment, use one call:

```bash
python experiments/combinations/compose.py \
  --source results/phase2/pose/full-control/pass3.zip \
  --runtime-source submissions/semantic_blocks \
  --entropy-source results/phase2/entropy/candidates/margin50_4_raw_1/archive.zip \
  --name pose_full_control_entropy_lossless_rebuild \
  --refine-radius 32
```

The entropy decoder is selected from its archive directory, or can be supplied
with `--entropy-runtime-source`. Hybrid composition requires exactly equal
decoded HPAC bytes and matching HPAC/rendering implementation files. It keeps
all model-section bytes from `--source`, all entropy-tail bytes from
`--entropy-source`, and verifies decoded renderer/carrier state against the
model source and decoded table/token state against the entropy source. The
entropy source's ETP1-aware reader is copied, with only block-format dispatch
changed. Provenance includes both archive hashes and the compatibility gates.
These CPU checks do not verify causal token replay or video metrics; final GPU
inflation and all-frame evaluation remain separate mandatory gates.

After an interruption, audit a saved composition without rebuilding a separate
candidate or changing its archive:

```bash
python experiments/combinations/verify.py \
  --candidate results/phase2/combinations/renderer_all_metadata_b2_lossless \
  --candidate results/phase2/combinations/pose_rescue_control_lossless \
  --output results/phase2/combinations/restart-audit-wsl.json
```

The audit rejects changed input/output archives, altered recovered data,
nonidentical deterministic rebuilds, changed learned states, and unexpected
runtime edits. `--container-only` enables a Windows audit without Torch; it
explicitly reports that decoded learned-state checking was skipped. The
existing Windows and WSL audit files distinguish those validation levels.

Two complete compositions survived the shutdown:

| Candidate | Archive bytes | Saved versus its immediate source |
|---|---:|---:|
| renderer_all_metadata_b2_lossless | 184,908 | 515 |
| pose_rescue_control_lossless | 185,869 | 282 |

Their immediate source models differ from the submitted archive. These sizes
alone are not leaderboard improvements: their distortion must be evaluated.
After resuming, `renderer_b2_pose_refined_lossless_screen` composed the stable
subset-repaired B2 source into 184,903 bytes, saving 520 bytes versus that
source. Windows and WSL rebuilds match, with full CPU learned-state equality
checked during composition. This candidate is explicitly a subset screen;
full-600-frame coefficient recovery will receive a separate output directory.
The separate `experiments/lossless/candidate/archive.zip` is 185,876 bytes and
preserves the submitted candidate's model and entropy bytes exactly.

Hybrid integration then produced `pose_control_entropy_lossless_screen`
(185,704 bytes) from the stable subset checkpoint, and
`pose_full_control_entropy_lossless` (185,704 bytes) from the independently
audited full-600 `full-control/pass3.zip`. Equal archive sizes do not imply
equal learned data; their archive hashes and coefficient bytes differ. The
full-control hybrid hash is
`640e9a8d618652ce4e0570cbcc92348d3f2457b8b737a13ab68cdbb9595fcf2f`.
It saves 448 bytes versus its immediate 186,152-byte pose source and 447 bytes
versus the submitted archive. Its 74,861 model bytes are preserved from the
pose source and its 114,637 entropy-tail bytes from `margin50_4_raw_1`.
All learned-state compatibility gates passed during WSL composition; Windows
rebuilds match exactly. GPU scores are not assigned by this workstream.

The audited full-600 B2 counterpart, `pose_full_b2_entropy_lossless`, contains
184,747 bytes (679 saved versus its immediate 185,426-byte source). Its hash is
`b061370545ac9bd12f741a68ec8b0218bba0aa757e2c5442774ebfae5ff756f1`.
Reproduce it with the same command using
`results/phase2/pose/full-all_metadata_b2/pass3.zip` as the model source and a
fresh output name such as `pose_full_b2_entropy_lossless_rebuild`. The smaller
archive alone does not establish a better score; B2 changes renderer metadata
and its pose-recovery experiment must account for the resulting distortion.

Final hybrid audit receipts are `hybrid-full-control-audit-{windows,wsl}.json`
and `hybrid-full-b2-audit-{windows,wsl}.json` under the combinations result
directory. Candidate LINEAGE.md files are generated from the actual phase-2
inputs, with separate inherited/changed state and review status. Documentation
updates do not change archives or decoder algorithms.

`test_compose.py` exercises truncating/extending model lengths while preserving
their recovered bytes, immutable recipes, and output path containment. It is
complementary to the lossless container's own malformed-input tests.
