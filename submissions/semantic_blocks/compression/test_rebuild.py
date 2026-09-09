"""Integration checks using recovered inputs; no synthetic codec implementation."""
import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

SUBMISSION = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SUBMISSION))
from compress import compose, TARGET_SHA256, read_source, MODEL_SHA256, ENTROPY_SHA256
from runtime.block_container import decode_model_blocks
from compression.blk2 import build_payload, zip_payload

INPUTS = None


class RebuildTests(unittest.TestCase):
    def setUp(self):
        self.models = INPUTS / 'results/phase2/pose/full-control/pass3.zip'
        self.entropy = INPUTS / 'results/phase2/entropy/candidates/margin50_4_raw_1/archive.zip'

    def test_real_sources_reproduce_target_and_decoded_state(self):
        from runtime.residual_archive import read_residual_archive
        from runtime.entropy.renderer_weight_codec import decode_wans1
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'archive.zip'
            with contextlib.redirect_stdout(io.StringIO()):
                compose(self.models, self.entropy, output)
            self.assertEqual(output.stat().st_size, 185704)
            self.assertEqual(hashlib.sha256(output.read_bytes()).hexdigest(), TARGET_SHA256)
            expected_model = read_residual_archive(self.models)
            expected_tail = read_residual_archive(self.entropy)
            actual = read_residual_archive(output)
            for name in ('semantic_blob', 'carrier_blob', 'hpac_blob'):
                self.assertEqual(getattr(actual, name), getattr(expected_model, name))
            for name in ('residual_payload', 'token_stream'):
                self.assertEqual(getattr(actual, name), getattr(expected_tail, name))
            self.assertEqual(expected_model.hpac_blob, expected_tail.hpac_blob)
            tensors = lambda p: {t.schema.name: t.values.tobytes() for t in decode_wans1(p.semantic_blob)}
            self.assertEqual(tensors(actual), tensors(expected_model))
            self.assertEqual(actual.table.margins, (4,))
            self.assertEqual(actual.table.codes.shape, (50, 5))
            self.assertEqual(actual.table.values.tobytes(), expected_tail.table.values.tobytes())

    def test_wrong_source_is_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'archive.zip'
            with self.assertRaisesRegex(ValueError, 'Wrong input'):
                compose(self.entropy, self.entropy, output)
            self.assertFalse(output.exists())
            self.assertFalse(output.with_name(output.name + '.rebuild.json').exists())

    def test_existing_output_and_published_path_are_protected(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'archive.zip'
            output.write_bytes(b'keep existing file')
            with self.assertRaisesRegex(ValueError, 'never overwritten'):
                compose(self.models, self.entropy, output)
            self.assertEqual(output.read_bytes(), b'keep existing file')
        with self.assertRaisesRegex(ValueError, 'never overwritten'):
            compose(self.models, self.entropy, SUBMISSION / 'archive.zip')

    def test_original_radius_32_search_recovers_fixed_recipe(self):
        import importlib.util
        # Load the actual recovered composition functions with their existing imports.
        # Its main() is not called; no historical output paths are written.
        script = SUBMISSION / 'compression/recovered/experiments/combinations/compose.py'
        sys.path.insert(0, str(script.parent))
        spec = importlib.util.spec_from_file_location('recovered_compose', script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _, models, _ = read_source(self.models, MODEL_SHA256)
        _, _, tail = read_source(self.entropy, ENTROPY_SHA256)
        original = json.loads((SUBMISSION / 'compression/lossless-recipe.json').read_text())
        selected, trials = module.refine(models, tail, module.adapt_recipe(models, original), 32)
        self.assertEqual(trials, 68)
        self.assertEqual(selected, json.loads((SUBMISSION / 'recipe.json').read_text()))
        self.assertEqual(decode_model_blocks(build_payload(models, tail, selected)), (models, tail))
        self.assertEqual(hashlib.sha256(zip_payload(build_payload(models, tail, selected))).hexdigest(), TARGET_SHA256)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    args, remaining = parser.parse_known_args()
    INPUTS = args.inputs.resolve()
    unittest.main(argv=[sys.argv[0], *remaining])
