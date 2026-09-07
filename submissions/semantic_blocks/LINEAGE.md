# Provenance and review notes

This is a local research candidate derived from **semantic-pose-HPAC_CPR1_polished**,
[commaai PR #135](https://github.com/commaai/comma_video_compression_challenge/pull/135),
by codexblack. Source commit: `6dcf77164ccbdcc1e0e41c99312e65ace4bc1fb4`.

Original archive: [release download](https://github.com/codexblack/comma_video_compression_challenge/releases/download/semantic-pose-HPAC_CPR1_polished-f26/archive.zip).
SHA-256: `12cf5d71a94065184f097c3e40dfe9f1db8402a1a76a80efc76a6956fe1e4004`.
Original size: **186,724 bytes**. Published exact score: **0.16226842169958583**.

The original renderer, pose carrier, HPAC model, residual correction table,
token stream, and frame-zero selector are retained. `cpr1/` and the runtime
files are copied from PR #135, except `runtime/residual_archive.py`, which
accepts the additional BLK1 lossless container. `runtime/block_container.py`,
the small compression/inflation entry points, and recipe are the local change.
The upstream MIT license is included.

PR #135 builds on [PR #130](https://github.com/commaai/comma_video_compression_challenge/pull/130),
[PR #133](https://github.com/commaai/comma_video_compression_challenge/pull/133),
jas0xf's [PR #86](https://github.com/commaai/comma_video_compression_challenge/pull/86),
and EthanYangTW's [PR #67](https://github.com/commaai/comma_video_compression_challenge/pull/67)
and [PR #79](https://github.com/commaai/comma_video_compression_challenge/pull/79).
The model architecture and learned state are prior work, not a new invention here.

The compressor requires the prior trained archive; it is an archive repacker,
not an independent training pipeline from the original video. The decoder
uses only the submitted archive and bundled decoder code at runtime.

The local changes and experiments were made with Codex assistance.
