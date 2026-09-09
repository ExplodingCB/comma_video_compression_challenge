The best additional lossless candidate is `candidate/archive.zip`, 185,876 bytes.
That is 275 bytes smaller than the 186,151-byte submitted archive and 848 bytes
smaller than PR #135's 186,724-byte archive. Its predicted score using unchanged
local distortions is **0.1617056954760368**. This prediction is arithmetic from
byte savings; this workstream did not repeat GPU inflation or official scoring.

The complete archive SHA-256 is
`9a5d4064ce53ed338c15e9f34ae36c013dd4597df56e566c8ece067a26705e7c`.
The archive rebuilds identically on Windows Python 3.12 and WSL Python 3.11 with
Brotli 1.2.0. Its recovered 74,860 model bytes and 114,802 tail bytes match the
submitted archive exactly. Decoder checks cover both BLK1/BLK2, 78 byte-lane
transpose cases, and eight malformed containers.

The five final blocks are:

| Original model interval | Storage | Stored data bytes | Charged header bytes |
|---|---|---:|---:|
| 0–16,264 | Brotli | 14,600 | 4 |
| 16,264–24,917 | Byte-lane transpose, stride 2, then Brotli | 6,505 | 6 |
| 24,917–52,680 | Raw | 27,763 | 4 |
| 52,680–65,022 | Brotli | 12,237 | 4 |
| 65,022–74,860 | Raw | 9,838 | 4 |

Stored block data is 70,943 bytes. Block headers are 22 bytes, and the outer
model header is nine bytes, totaling a 70,974-byte model container. Adding the
unchanged 114,802-byte tail and 100-byte ZIP overhead gives 185,876 bytes.
All data-dependent block lengths and transform choices are in the archive.
No models, dictionaries, calibration parameters, or learned tables moved into
decoder source. The only new library is the general-purpose Brotli decoder.

The bounded searches completed were:

| Stage | Work | Best complete archive |
|---|---|---:|
| Codec screen | 530 settings across 10 spans: PPMd H/I, Brotli, zstd, raw, LZMA2 | 186,116 bytes when merged with the existing recipe |
| Layout screen | 8,700 codec/layout combinations, including byte transposes, deltas, XOR deltas, bit shuffle and nibble separation | 185,944 bytes |
| Partition DP | 72 boundaries, 2,556 intervals, seven raw/Brotli/LZMA2 settings per interval | 186,060 bytes |
| Coordinate refinement | 18,925 configurations across 757 intervals around the winning layout | 185,876 bytes |
| Schema split screen | Three decompositions suggested by the DP result | No further improvement |

PPMd 1.3.1 failed round-trip recovery or raised a decoder error for 3 of the
530 codec settings and 95 of the 8,700 layout combinations. Those records are
explicitly rejected. No accepted candidate depends on PPMd. zstd did not beat
the selected combination. The ledgers retain unsuccessful measurements.

To reproduce on Windows, install the optional screening packages into
`experiments/lossless/deps` and run `sweep.py`, `sweep.py --layouts`, `block_dp.py`,
`refine.py`, `split_screen.py`, `prepare_candidate.py`, and `verify_candidate.py`.
All scripts resolve inputs under this project. `block_dp.py` emits detailed
progress, so redirect output to its project-local result log.

For candidate deployment, `inflate.sh` and `compress.sh` automatically call
`prepare_dependencies.sh` when Brotli cannot be imported. Preparation uses
`uv pip --target` when uv is available, otherwise `python -m pip --target`.
Dependencies go in `candidate/.deps`; installer caches and temporary files go
in `candidate/.cache` and `candidate/.tmp`. The evaluated Python environment is
not modified. `test_bootstrap.sh` successfully exercises the missing-package
case in an isolated Python environment and checks the rebuilt archive hash.

For composition with a newly encoded entropy tail, import
`candidate/compress.py`, then call `build_payload(models, tail, recipe)` and
`zip_payload(payload)`. The existing recipe requires 74,860 model bytes and
must be searched again if model lengths change. Its decoder is
`candidate/runtime/block_container.py`; the inherited residual reader has one
additional format dispatch for `BLK2`. No submitted files or public PR assets
were modified by this workstream.

Results, complete ledgers, recipes, and platform verification records are in
`results/phase2/lossless/`. This is an internal research record, not a public PR
description.

Independent entropy-code review: probability screening should use the native
coder's quantized positive frequencies, not raw float32 probabilities. The
existing native exact re-encode and symbol replay gates correctly prevent
screening estimates from becoming unverified results. Systematic sampling
every 101st row can alias raster/group order and miss rare errors; proposal
fitting should retain all mistakes and use weighted random samples of correct
uncertain rows, or sample deterministically with a seeded random generator and
stratify by context/symbol/confidence.
