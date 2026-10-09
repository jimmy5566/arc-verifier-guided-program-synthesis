from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("cpu_exact", ROOT / "scripts" / "unified_native_exact_grid_cpu_contract.py")
module = importlib.util.module_from_spec(spec)
assert spec.loader
spec.loader.exec_module(module)


class ExactGridCpuContractTests(unittest.TestCase):
    def test_grid_sidecar_accepts_grid_schema_and_returns_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sidecar.json"
            path.write_text(json.dumps({"protocol_id": "x", "targets": {"episode": [[1, 2], [3, 4]]}}), encoding="utf-8")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            hashes = module.read_grid_sidecar_hashes(path, digest, {"episode"})
            self.assertEqual(hashes["episode"], module.canonical_grid_hash([[1, 2], [3, 4]]))

    def test_grid_sidecar_rejects_token_schema_and_mapping_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sidecar.json"
            path.write_text(json.dumps({"target_token_ids_by_episode": {"episode": [1, 2]}}), encoding="utf-8")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaisesRegex(RuntimeError, "EPISODE_MAPPING"):
                module.read_grid_sidecar_hashes(path, digest, {"episode"})

    def test_exact_scores_top2_without_candidate_selection(self):
        target = module.canonical_grid_hash([[1]])
        row = {
            "checkpoint_condition": "RECONSTRUCTED_FOUNDATION_V2_V7",
            "episode_id": "episode",
            "family": "STRUCTURAL_TRANSLATION",
            "prompt_sha256": "p",
            "canonical_prediction_sha256": module.canonical_grid_hash([[2]]),
            "alternate_token_ids": [1, 15],
        }
        output = module.score_exact_grid_rows([row], {"episode": target})[0]
        self.assertFalse(output["greedy_exact_grid_match"])
        self.assertTrue(output["alternate_only_exact_grid_match"])
        self.assertTrue(output["complete_output_top2_coverage"])
        self.assertFalse(output["target_alignment_used_for_selection"])

    def test_invalid_alternate_is_an_incorrect_candidate_not_target_repair(self):
        target = module.canonical_grid_hash([[1]])
        row = {
            "checkpoint_condition": "RECONSTRUCTED_FOUNDATION_V2_V7",
            "episode_id": "episode",
            "family": "STRUCTURAL_TRANSLATION",
            "prompt_sha256": "p",
            "canonical_prediction_sha256": module.canonical_grid_hash([[2]]),
            "alternate_token_ids": [15],
        }
        output = module.score_exact_grid_rows([row], {"episode": target})[0]
        self.assertIsNone(output["alternate_prediction_sha256"])
        self.assertFalse(output["complete_output_top2_coverage"])


if __name__ == "__main__":
    unittest.main()
