# Causal entropy table experiment

This branch keeps the PR #135 predictor weights and semantic tokens fixed and
fits a small correction to the predictor's class probabilities. Better
probabilities let the same arithmetic coder store the same tokens in fewer
bytes. All fitted table entries, scales and bin boundaries live in the archive.

The submitted table has 25 states: five causal boundary states times five
predicted classes. The best tested replacement splits each state by the gap
between the two largest uncorrected logits, using a boundary of 4. Its 50 states
hold five signed six-bit corrections each, with one serialized float16 scale.
The ETP1 header, boundary and table occupy 198 bytes versus the original 96.
The token stream shrinks from 114,706 to 114,439 bytes, leaving a net **165-byte
saving**, including ZIP overhead. The standalone archive is 185,986 bytes.

Its SHA-256 is
`9d0a14db2620d788a6a5007dd4b5e09db9610f996710e26f9a6062760a1e3db4`.
This is an experimental local candidate, separate from the submitted PR #141.

## Evidence and limits

`results/phase2/entropy/baseline-replay.json` verifies all 117,964,800 cached
probability rows against the deployed decoder and reproduces the entire
submitted arithmetic stream byte for byte. The probability SHA-256 is
`ea28d84909340b1479b62fdcb85bb682d197e9f27c176e218e3104b782721d32`.
`encoded.json` records actual archive sizes and exact cached-probability token
round trips for the three finalists. Cached replay is not independent causal
decoding: the final composed candidate's fresh GPU inflation and evaluation
provide that separate gate. This gate has now passed in the full control
composition using the exact same entropy tail and predictor bytes; the receipt
is `results/phase2/entropy/causal-validation.json`. Consult the composed
candidate's evaluation report for final metrics. `encoded.json` retains the
status at encoding time, before this independent validation.

The cache stores logits in eighths. An early screen incorrectly passed 256 as
the probability quantization precision; the inherited runtime uses **8**.
Those results are invalid and retained only under `precision256-debug/` and
`uniform-run/`. The current scripts obtain precision from the actual runtime.
All promoted candidates were fitted, screened and encoded after the baseline
replay gate passed at precision 8. The captured source cache itself was correct.

The weighted fit retains every mistaken or low-margin row and samples the
background with inverse-probability weights, for 1,047,608 fitting rows. Four
table families, two gauges and two update strengths give 16 proposals. Fitting
is bounded at 80 L-BFGS-B iterations; this is a search result, not a claim of
global optimality. Screening uses the coder's integer frequencies. The best
three candidates are then replayed across every row and actually encoded.

## Reproduction

Use WSL from the project root after sourcing `scripts/env.sh`. SciPy 1.16.2 is
installed only in `experiments/entropy/deps`; it is not a decoder dependency.
The native encoder is the attributed reference implementation, compiled with
the project compiler wrapper:

```bash
cc -O3 -shared -fPIC references/pr135-experiments/src/cpr1_sub4/entropy/rc64_backend.c -o results/phase2/entropy/librc64.so -lm
python experiments/entropy/capture.py
python experiments/entropy/replay_baseline.py
python experiments/entropy/fit.py fit_weighted
python experiments/entropy/fit.py screen
python experiments/entropy/encode.py
```

Capture reuses an already completed cache and needs exclusive GPU access when
creating one. Later steps use the CPU. The candidate decoder package is under
`results/phase2/entropy/candidates/margin50_4_raw_1/`. Composition with changed
pose coefficients and lossless model packing is described in
`experiments/combinations/README.md`.

The ETP1 decoder calculates its feature from the current causal predictor's
uncorrected logits. It uses a right-inclusive bin search, matching the encoder
at exactly 4. No original tokens, probability cache or target video is supplied
to the deployed decoder. Original model attribution and licenses are retained.
