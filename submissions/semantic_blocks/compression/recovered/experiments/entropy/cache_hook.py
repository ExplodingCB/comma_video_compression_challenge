"""Capture frozen predictor rows during an unchanged production token decode.

Experimental instrumentation only; never included in a submission decoder.
The cache is in arithmetic-coding order, not raster order.
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROWS = 600 * 384 * 512


class Capture:
    def __init__(self, module, output):
        self.module = module
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.original_probability = module._probability_table
        self.original_decode = module.NativeDecoder.decode
        self.base = np.memmap(self.output / 'base_logits.i16', mode='w+', dtype='<i2', shape=(ROWS, 5))
        self.feature = np.memmap(self.output / 'feature.u8', mode='w+', dtype='u1', shape=(ROWS,))
        self.symbols = np.memmap(self.output / 'symbols.u8', mode='w+', dtype='u1', shape=(ROWS,))
        self.position = 0
        self.decoded = 0
        self.probability_hash = hashlib.sha256()
        self.base_hash = hashlib.sha256()

        def probability(logits, precision):
            caller = sys._getframe(1).f_locals
            if 'base_logits' not in caller or 'feature' not in caller:
                raise RuntimeError('Capture requires the production decode loop')
            base = np.asarray(caller['base_logits'], dtype=np.float32)
            codes = np.rint(base * 8).astype('<i2')
            if not np.array_equal(codes.astype(np.float32) / 8, base):
                raise RuntimeError('Base predictor logits are not exact signed-int16 eighths')
            n = len(base)
            end = self.position + n
            if end > ROWS or self.position != self.decoded:
                raise RuntimeError('Unexpected coding order while capturing')
            self.base[self.position:end] = codes
            self.feature[self.position:end] = caller['feature']
            self.base_hash.update(codes.tobytes())
            result = self.original_probability(logits, precision)
            self.probability_hash.update(result.tobytes())
            self.position = end
            return result

        def decode(decoder, probabilities):
            result = self.original_decode(decoder, probabilities)
            end = self.decoded + len(result)
            if end != self.position:
                raise RuntimeError('Probability and symbol captures diverged')
            self.symbols[self.decoded:end] = result
            self.decoded = end
            return result

        module._probability_table = probability
        module.NativeDecoder.decode = decode

    def finish(self):
        self.module._probability_table = self.original_probability
        self.module.NativeDecoder.decode = self.original_decode
        if self.position != ROWS or self.decoded != ROWS:
            raise RuntimeError(f'Incomplete capture: {self.position}, {self.decoded}')
        for array in (self.base, self.feature, self.symbols):
            array.flush()
        result = {
            'rows': ROWS,
            'layout': 'production arithmetic-coding order',
            'base_logit_scale': 8,
            'base_logits_sha256': self.base_hash.hexdigest(),
            'probability_sha256': self.probability_hash.hexdigest(),
            'expected_probability_sha256': 'ea28d84909340b1479b62fdcb85bb682d197e9f27c176e218e3104b782721d32',
        }
        if result['probability_sha256'] != result['expected_probability_sha256']:
            raise RuntimeError('Captured probabilities differ from the validated baseline')
        (self.output / 'manifest.json').write_text(json.dumps(result, indent=2) + '\n')
        return result
