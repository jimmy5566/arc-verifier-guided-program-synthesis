from __future__ import annotations

import hashlib
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.launch_paired_v7_family_balanced_rank_margin_v1 import build_binding


class LauncherTests(unittest.TestCase):
    def test_binding_requires_exact_head_hashes_fresh_output_and_subset(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); inputs = {}
            for key in ("raw_predictions", "input_manifest", "v7_manifest", "family_balanced_manifest", "sealed_sidecar"):
                path = root / f"{key}.json"; path.write_text("{}", encoding="utf-8"); inputs[key] = path
            config = {"runtime_inputs": {"raw_predictions_path": inputs["raw_predictions"].name, "input_manifest_path": inputs["input_manifest"].name, "v7_manifest_path": inputs["v7_manifest"].name, "family_balanced_manifest_path": inputs["family_balanced_manifest"].name, "expected_hashes": {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in inputs.items() if key != "sealed_sidecar"}}, "sealed_sidecar_sha256": hashlib.sha256(inputs["sealed_sidecar"].read_bytes()).hexdigest(), "batch1_subset_episode_ids": [str(index) for index in range(12)], "checkpoints": {}}
            with patch("scripts.launch_paired_v7_family_balanced_rank_margin_v1.git_head", return_value="a" * 40):
                binding = build_binding(root=root, config=config, sidecar=inputs["sealed_sidecar"], output=root / "fresh" / "output", receipt=root / "fresh" / "receipt", expected_commit="a" * 40)
            self.assertEqual(binding["runtime_cap_seconds"], 900)
            self.assertEqual(set(binding["input_paths"]), set(binding["expected_hashes"]))


if __name__ == "__main__":
    unittest.main()
