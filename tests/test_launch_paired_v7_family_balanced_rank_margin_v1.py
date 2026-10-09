from __future__ import annotations

import hashlib
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from scripts.launch_paired_v7_family_balanced_rank_margin_v1 import build_binding, execute_bounded


class LauncherTests(unittest.TestCase):
    def test_binding_requires_exact_head_hashes_fresh_output_and_subset(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); inputs = {}
            for key in ("raw_predictions", "input_manifest", "v7_manifest", "family_balanced_manifest", "sealed_sidecar"):
                path = root / f"{key}.json"; path.write_text("{}", encoding="utf-8"); inputs[key] = path
            output, receipt = root / "fresh" / "output", root / "fresh" / "receipt"
            names = ("config_sha256", "contract_sha256", "runtime_contract_sha256", "launch_contract_sha256", "worker_sha256", "launcher_sha256", "postprocessor_sha256")
            implementation_files, identity = {}, {}
            for key in names:
                path = root / f"{key}.txt"; path.write_text(key, encoding="utf8"); implementation_files[key] = path.name; identity[key] = hashlib.sha256(path.read_bytes()).hexdigest()
            config = {"runtime_inputs": {"raw_predictions_path": inputs["raw_predictions"].name, "input_manifest_path": inputs["input_manifest"].name, "v7_manifest_path": inputs["v7_manifest"].name, "family_balanced_manifest_path": inputs["family_balanced_manifest"].name, "expected_hashes": {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in inputs.items() if key != "sealed_sidecar"}}, "sealed_sidecar_sha256": hashlib.sha256(inputs["sealed_sidecar"].read_bytes()).hexdigest(), "batch1_subset_episode_ids": [str(index) for index in range(12)], "checkpoints": {}, "implementation_identity": identity, "implementation_files": implementation_files, "launch_identity": {"nonce": "0" * 32, "source_commit": "a" * 40, "origin_ref": "origin/test", "output_path": str(output), "receipt_path": str(receipt)}}
            with patch("scripts.launch_paired_v7_family_balanced_rank_margin_v1.git_head", return_value="a" * 40), patch("scripts.launch_paired_v7_family_balanced_rank_margin_v1.git_ref", return_value="a" * 40), patch("scripts.launch_paired_v7_family_balanced_rank_margin_v1.require_clean_tracked_checkout"):
                binding = build_binding(root=root, config=config, sidecar=inputs["sealed_sidecar"], output=output, receipt=receipt, expected_commit="a" * 40)
            self.assertEqual(binding["runtime_cap_seconds"], 900)
            self.assertEqual(set(binding["input_paths"]), set(binding["expected_hashes"]))

    def test_timeout_kills_worker_and_preserves_timeout_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            receipt = Path(temporary) / "receipt.json"
            process = Mock(); process.pid = 123; process.wait.side_effect = [subprocess.TimeoutExpired(["worker"], 900), None]
            with patch("scripts.launch_paired_v7_family_balanced_rank_margin_v1.subprocess.Popen", return_value=process), patch("scripts.launch_paired_v7_family_balanced_rank_margin_v1.os.killpg", create=True) as kill:
                self.assertEqual(execute_bounded(["worker"], receipt=receipt, cap_seconds=900), 124)
            kill.assert_called_once()
            self.assertEqual(__import__("json").loads(receipt.read_text())["status"], "TIMEOUT_NO_UPDATE")


if __name__ == "__main__":
    unittest.main()
