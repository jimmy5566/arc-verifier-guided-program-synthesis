from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("unified_prepare", ROOT / "scripts/prepare_unified_native_model_capability_baseline_v1.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


class UnifiedNativePreparationTests(unittest.TestCase):
    def test_synthetic_inputs_are_unique_and_targets_are_sealed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            made = MODULE.prepare(root / "out", root / "sealed", 17, 4)
            manifest = json.loads(made["manifest"].read_text(encoding="utf-8"))
            contract = json.loads(made["contract"].read_text(encoding="utf-8"))
            self.assertEqual(len(manifest["episodes"]), 20)
            self.assertTrue(manifest["content_level_audit"]["observation_hashes_unique"])
            self.assertFalse(any("output" in episode["observation"]["task"]["test"][0] for episode in manifest["episodes"]))
            self.assertTrue(made["sealed_target_sidecar"].is_file())
            self.assertNotIn("targets", manifest)
            self.assertEqual(contract["inference_contract"]["batch_size"], 1)
            self.assertIn("training", contract["forbidden"])
            self.assertIn("FINAL_AUDIT", contract["forbidden"])

    def test_checkpoint_discovery_keeps_historical_reference_unsubstituted(self):
        discovery = MODULE.discover_checkpoints()
        self.assertEqual(discovery["HISTORICAL_FOUNDATION_V2"]["status"], "UNAVAILABLE_NOT_SUBSTITUTED")
        self.assertEqual(discovery["HISTORICAL_FOUNDATION_V2"]["required_adapter_model_sha256"], MODULE.HISTORICAL_FOUNDATION_SHA256)


if __name__ == "__main__":
    unittest.main()
