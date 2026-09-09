"""Validate cached-row replay against the actual deployed precision and bitstream."""
import hashlib
import json
import time
import numpy as np
from fit import OUT, SOURCE, ROWS, CHUNK, PRECISION, arrays, coding_nll
from runtime.residual_archive import read_residual_archive, _probability_table
from cpr1_sub4.entropy.rc64 import NativeEncoder


def main():
    base, feature, symbols = arrays()
    parts = read_residual_archive(SOURCE/'archive.zip')
    encoder = NativeEncoder(OUT/'librc64.so')
    digest = hashlib.sha256()
    nll = 0.0
    started = time.monotonic()
    for start in range(0, ROWS, CHUNK):
        b = np.asarray(base[start:start+CHUNK], dtype=np.float32)/8
        f = feature[start:start+CHUNK]
        s = np.asarray(symbols[start:start+CHUNK], dtype=np.int32)
        p = _probability_table(b+parts.table.values[f], PRECISION)
        digest.update(p.tobytes())
        encoder.encode(s, p)
        nll += coding_nll(p, s)
    stream = encoder.finish()
    encoder.close()
    expected_probability = 'ea28d84909340b1479b62fdcb85bb682d197e9f27c176e218e3104b782721d32'
    report = dict(precision=PRECISION, rows=ROWS, probability_sha256=digest.hexdigest(),
                  probability_matches=digest.hexdigest()==expected_probability,
                  token_stream_matches=stream==parts.token_stream, token_bytes=len(stream),
                  nll_bits=nll, overhead_bytes=len(stream)-nll/8,
                  seconds=time.monotonic()-started)
    (OUT/'baseline-replay.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
    if not report['probability_matches'] or not report['token_stream_matches']:
        raise RuntimeError('Baseline replay gate failed; no calibration result is valid')


if __name__ == '__main__':
    main()
