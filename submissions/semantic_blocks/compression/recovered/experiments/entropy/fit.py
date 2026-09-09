"""Optimize charged causal calibration tables using captured frozen predictor rows.

Fitting uses a high-coverage uncertain-row subset; final screening replays every
row with the exact production quantization. Only actual re-encoded archives may
be called improvements. All learned table bytes are charged.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent / 'deps'))
import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results' / 'phase2' / 'entropy'
CACHE = OUT / 'cache'
SOURCE = ROOT / 'submissions' / 'semantic_blocks'
sys.path.insert(0, str(SOURCE))
from runtime.residual_archive import read_residual_archive, _probability_table
from runtime.f26_inflate import _load_renderer
PRECISION = _load_renderer(SOURCE / 'cpr1').HPAC_LOGIT_PRECISION
sys.path.insert(0, str(ROOT / 'references/pr135-experiments/src'))
from cpr1_sub4.entropy.rc64 import quantize_probabilities

ROWS = 600 * 384 * 512
CHUNK = 524288


def arrays():
    manifest = json.loads((CACHE / 'manifest.json').read_text())
    assert manifest['rows'] == ROWS
    return (np.memmap(CACHE / 'base_logits.i16', mode='r', dtype='<i2', shape=(ROWS, 5)),
            np.memmap(CACHE / 'feature.u8', mode='r', dtype='u1', shape=(ROWS,)),
            np.memmap(CACHE / 'symbols.u8', mode='r', dtype='u1', shape=(ROWS,)))


def features(base, original, bins):
    if not bins:
        return original.astype(np.int64)
    top = np.partition(base, -2, axis=1)[:, -2:]
    margin = top[:, 1].astype(np.int32) - top[:, 0].astype(np.int32)
    bucket = np.searchsorted(np.asarray(bins) * 8, margin, side='right')
    return original.astype(np.int64) * (len(bins) + 1) + bucket


def quantize(values, bits=6, scale_factor=1.0):
    # The fixed baseline decoder allows -32..31, use a symmetric subset.
    scale = float(np.float16(max(np.abs(values).max(), 1e-9) / ((1 << (bits-1))-1) * scale_factor))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Unrepresentable table scale')
    codes = np.clip(np.rint(values / scale), -(1 << (bits-1)), (1 << (bits-1))-1).astype(np.int8)
    return codes, scale, codes.astype(np.float32) * scale


def coding_nll(probabilities, symbols):
    """Ideal length after the deployed RC64 integer-frequency normalization."""
    frequencies = quantize_probabilities(probabilities)
    selected = frequencies[np.arange(len(symbols)), symbols].astype(np.float64)
    return float((31 - np.log2(selected)).sum())


def active_rows():
    destination = OUT / 'active.npz'
    if destination.exists():
        with np.load(destination) as a:
            return tuple(a[n] for n in ('base', 'feature', 'symbols'))
    base, context, symbols = arrays()
    parts = read_residual_archive(SOURCE / 'archive.zip')
    collected = [[], [], []]
    baseline_nll = 0.0
    omitted_bound_bits = 0.0
    for start in range(0, ROWS, CHUNK):
        end = min(start + CHUNK, ROWS)
        b = np.asarray(base[start:end])
        f = np.asarray(context[start:end])
        s = np.asarray(symbols[start:end])
        p = _probability_table(b.astype(np.float32)/8 + parts.table.values[f], PRECISION)
        baseline_nll -= np.log2(p[np.arange(len(s)), s].astype(np.float64)).sum()
        # Retain every mistaken prediction and rows where the total other-class
        # mass is >=exp(-22). Correct confident rows have negligible fit gradient.
        winner = p.argmax(1)
        other = p.astype(np.float64).sum(1) - p[np.arange(len(s)), winner]
        keep = (winner != s) | (other >= np.exp(-22))
        omitted_bound_bits += float(other[~keep].sum()/np.log(2))
        for dest, value in zip(collected, (b[keep], f[keep], s[keep])):
            dest.append(value.copy())
    active = tuple(np.concatenate(a) for a in collected)
    np.savez(destination, base=active[0], feature=active[1], symbols=active[2])
    report = dict(rows=ROWS, active_rows=len(active[0]), baseline_nll_bits=float(baseline_nll),
                  omitted_baseline_nll_bound_bits=omitted_bound_bits)
    (OUT / 'active.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)
    return active


def objective(values, base, feature, symbols, states, weights=None):
    table = values.reshape(states, 5)
    logits = base + table[feature]
    # Same clipping bounds as deployment, smooth proposal before exact rounding.
    unclipped = (logits > -32768/PRECISION) & (logits < 32767/PRECISION)
    logits = np.clip(logits, -32768/PRECISION, 32767/PRECISION)
    logits -= logits.max(1, keepdims=True)
    p = np.exp(logits)
    norm = p.sum(1, keepdims=True)
    losses = np.log(norm[:, 0]) - logits[np.arange(len(symbols)), symbols]
    value = losses.sum() if weights is None else np.dot(losses, weights)
    p /= norm
    p[np.arange(len(symbols)), symbols] -= 1
    p *= unclipped
    if weights is not None:
        p *= weights[:, None]
    gradient = np.zeros((states, 5), dtype=np.float64)
    for c in range(5):
        gradient[:, c] = np.bincount(feature, weights=p[:, c], minlength=states)
    denominator = len(base) if weights is None else weights.sum()
    return float(value)/denominator, gradient.ravel()/denominator


def fit(weighted=False):
    OUT.mkdir(parents=True, exist_ok=True)
    base, original, symbols = active_rows()
    parts = read_residual_archive(SOURCE / 'archive.zip')
    rows = []
    # Use a deterministic cap for proposal fitting; final evaluation uses all rows.
    stride = max(1, int(np.ceil(len(base)/750000)))
    fit_base, fit_original = base[::stride], original[::stride]
    s = symbols[::stride]
    weights = None
    if weighted:
        # Preserve every mistake and low-margin row. Uniform background sampling
        # uses inverse-probability weights, avoiding loss of rare costly symbols.
        rng = np.random.default_rng(20260907)
        gathered = [[], [], [], []]
        important_count = 0
        for start in range(0, len(base), CHUNK):
            bc = base[start:start+CHUNK]
            fc = original[start:start+CHUNK]
            sc = symbols[start:start+CHUNK]
            logits = np.clip(bc.astype(np.float32)/8 + parts.table.values[fc], -32768/PRECISION, 32767/PRECISION)
            top = np.partition(logits, -2, axis=1)[:, -2:]
            margin = top[:,1] - top[:,0]
            important = (logits.argmax(1) != sc) | (margin < 2)
            probability = np.where(margin < 8, 1/32, 1/512)
            keep = important | (rng.random(len(bc)) < probability)
            important_count += int(important.sum())
            selected_weights = np.where(important[keep], 1.0, 1/probability[keep])
            for dest, data in zip(gathered, (bc[keep], fc[keep], sc[keep], selected_weights)):
                dest.append(data)
        fit_base, fit_original, s, weights = (np.concatenate(g) for g in gathered)
        print(json.dumps(dict(weighted_fit_rows=len(s), important_rows=important_count,
                              represented_rows=float(weights.sum()))), flush=True)
    b = fit_base.astype(np.float64)/8
    for name, bins in (('fixed25', []), ('margin50_4', [4]), ('margin50_12', [12]), ('margin100', [2, 8, 16])):
        feature = features(fit_base, fit_original, bins)
        states = 25 * (len(bins)+1)
        initial = np.repeat(parts.table.values, len(bins)+1, axis=0).astype(np.float64)
        started = time.monotonic()
        result = minimize(objective, initial.ravel(), args=(b, feature, s, states, weights),
                          jac=True, method='L-BFGS-B',
                          bounds=[(float(v)-1.5, float(v)+1.5) for v in initial.ravel()],
                          options={'maxiter': 80 if weighted else 45, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 12})
        fitted = result.x.reshape(states, 5)
        # Softmax is invariant to a row-wise offset except at clipping boundaries.
        # Include both gauges and let exact full-row screening decide.
        for gauge, values in (('raw', fitted), ('centered', fitted-fitted.mean(1, keepdims=True))):
            for weight in (.5, 1.0):
                mixed = initial + weight*(values-initial)
                codes, scale, deployed = quantize(mixed)
                candidate = dict(name=f'{name}_{gauge}_{weight:g}', bins=bins, states=states,
                                 codes=codes.tolist(), scale=scale, bits=6,
                                 iterations=int(result.nit), optimizer_success=bool(result.success),
                                 optimizer_message=str(result.message),
                                 fit_rows=len(b), fit_seconds=time.monotonic()-started)
                candidate['fit_sampling'] = 'all_errors_low_margin_plus_weighted_background' if weighted else 'uniform_stride'
                # New format requires magic + bin count + int8 bin edges + u16 state
                # count + fp16 scale + packed codes. Existing fixed25 needs no header.
                candidate['table_bytes'] = (2 + (states*5*6+7)//8 if not bins else
                                            4+1+len(bins)+2+2+(states*5*6+7)//8)
                rows.append(candidate)
        print(json.dumps(dict(fitted=name, active_rows=len(base), fit_rows=len(b),
                              iterations=result.nit, seconds=time.monotonic()-started)), flush=True)
        (OUT / 'proposals.json').write_text(json.dumps(rows, indent=2)+'\n')
    return rows


def screen():
    proposals = json.loads((OUT / 'proposals.json').read_text())
    parts = read_residual_archive(SOURCE / 'archive.zip')
    baseline = dict(name='baseline', bins=[], states=25, codes=parts.table.codes.tolist(),
                    scale=parts.table.scale, bits=6, table_bytes=96)
    candidates = [baseline] + proposals
    b, f, s = active_rows()
    # Screen all retained uncertain rows first; then full replay top three.
    for row in candidates:
        table = np.asarray(row['codes'], dtype=np.float32) * row['scale']
        nll = 0.0
        for start in range(0, len(b), CHUNK):
            base = b[start:start+CHUNK]
            feature = features(base, f[start:start+CHUNK], row['bins'])
            symbols = s[start:start+CHUNK]
            p = _probability_table(base.astype(np.float32)/8 + table[feature], PRECISION)
            nll += coding_nll(p, symbols)
        row['active_nll_bits'] = float(nll)
        row['active_charged_bytes'] = float(nll/8 + row['table_bytes'])
    ranked = sorted(candidates, key=lambda r:r['active_charged_bytes'])
    finalists = [baseline] + [r for r in ranked if r['name'] != 'baseline'][:3]
    base, original, symbols = arrays()
    for row in finalists:
        table = np.asarray(row['codes'], dtype=np.float32) * row['scale']
        nll = 0.0
        started = time.monotonic()
        for start in range(0, ROWS, CHUNK):
            b = np.asarray(base[start:start+CHUNK])
            f = features(b, original[start:start+CHUNK], row['bins'])
            s = symbols[start:start+CHUNK]
            p = _probability_table(b.astype(np.float32)/8+table[f], PRECISION)
            nll += coding_nll(p, s)
        row['full_nll_bits'] = float(nll)
        row['full_charged_ideal_bytes'] = float(nll/8+row['table_bytes'])
        row['projected_archive_bytes'] = float(71249 + 100 + nll/8 + row['table_bytes'])
        row['full_screen_seconds'] = time.monotonic()-started
        print(json.dumps({k:v for k,v in row.items() if k not in ('codes','bins')}), flush=True)
        (OUT / 'screen.json').write_text(json.dumps(dict(status='RC64_integer_frequency_NLL_screen_not_actual_archives', precision=PRECISION,
                                                       all_candidates=ranked, finalists=finalists), indent=2)+'\n')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=('fit','fit_weighted','screen'))
    args = ap.parse_args()
    fit(weighted=args.stage == 'fit_weighted') if args.stage.startswith('fit') else screen()
