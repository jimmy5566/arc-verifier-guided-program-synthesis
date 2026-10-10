import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from prepare_matched_atomic_to_compositional_transfer_v2 import (TUPLES, prepare, raw_scene, recolor_projection, select_projection)


class MatchedTransferSuccessorTests(unittest.TestCase):
    def test_freeze_has_deduplicated_public_prompts_and_sealed_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            result = prepare(Path(directory) / 'out', Path(directory) / 'sealed')
            manifest = json.loads(result['manifest'].read_text())
            protocol = json.loads(result['protocol'].read_text())
            sidecar = json.loads(result['sidecar'].read_text())
        self.assertEqual(len(manifest['episodes']), 168)
        self.assertEqual(len(manifest['latent_parameter_tuples']), TUPLES)
        self.assertEqual(len({item['canonical_latent_sha256'] for item in manifest['latent_parameter_tuples']}), TUPLES)
        for role in ('ATOMIC_SELECT_WITH_CUE', 'ATOMIC_PARAMETERIZED_RECOLOR', 'COMPOSITION_SELECT_RECOLOR'):
            prompts = [item['prompt_sha256'] for item in manifest['episodes'] if item['role'] == role]
            self.assertEqual(len(prompts), TUPLES); self.assertEqual(len(set(prompts)), TUPLES)
        self.assertEqual(len(sidecar['targets']), 168)
        self.assertEqual(protocol['inference']['decoding'], 'GREEDY_PRIMARY_ONLY')
        self.assertEqual(protocol['metrics']['common_support_requirement'].split()[0], 'at')

    def test_select_then_parameterized_recolor_reconstructs_each_composition_target(self):
        with tempfile.TemporaryDirectory() as directory:
            result = prepare(Path(directory) / 'out', Path(directory) / 'sealed')
            manifest = json.loads(result['manifest'].read_text())
            targets = json.loads(result['sidecar'].read_text())['targets']
        for latent in manifest['latent_parameter_tuples']:
            params = {key: value for key, value in latent.items() if key not in ('tuple_id', 'canonical_latent_sha256')}
            raw, anchor, marker = raw_scene(params, 2)
            selected = select_projection(raw, params, anchor, marker)
            expected = targets[f"{manifest['protocol_id']}:TUPLE:{latent['tuple_id'][1:]}:COMPOSITION_SELECT_RECOLOR"]
            self.assertEqual(recolor_projection(selected, params), expected)


if __name__ == '__main__':
    unittest.main()
