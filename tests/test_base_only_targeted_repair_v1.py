from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "base_only_targeted_repair_v1.py"


def load_module():
    spec = importlib.util.spec_from_file_location("base_only_targeted_repair_v1", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def sample_row(episode_id: str) -> dict:
    return {"episode_id": episode_id, "split": "TRAIN", "role": "ATOMIC_REPAIR", "family": "connected components", "task": {"train": [{"input": [[1]], "output": [[2]]}], "test": [{"input": [[3]], "output": [[4]]}]}}


class BaseOnlyTargetedRepairTests(unittest.TestCase):
    def test_native_schedule_is_repeatable_and_complete_groups(self) -> None:
        with self.subTest("deterministic"), __import__("tempfile").TemporaryDirectory() as root:
            path = Path(root) / "train.jsonl"
            path.write_text("\n".join(json.dumps(sample_row(f"train-{i}")) for i in range(4)) + "\n", encoding="utf-8")
            mod = load_module(); samples = mod.training_samples(path)
            first, second = mod.schedule_for(samples), mod.schedule_for(samples)
            self.assertEqual(first, second); self.assertEqual(len(first["episodes"]) % mod.ACCUMULATION, 0)
            self.assertGreaterEqual(first["actual_transformer_tokens"], mod.NOMINAL_TOKENS)
            self.assertTrue(all(item["token_label_sha256"] for item in first["episodes"]))

    def test_training_rejects_non_train_surface(self) -> None:
        with __import__("tempfile").TemporaryDirectory() as root:
            row = sample_row("bad"); row["split"] = "TARGET_DEV"; path = Path(root) / "bad.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "TRAIN_SURFACE_ROLE_OR_SPLIT_INVALID"):
                load_module().training_samples(path)

    def test_dummy_launcher_never_charges_or_constructs_optimizer(self) -> None:
        with __import__("tempfile").TemporaryDirectory() as root:
            preflight, output = Path(root) / "preflight.json", Path(root) / "dummy.json"
            preflight.write_text(json.dumps({"status": "PASS_TECHNICAL_BASE_REFERENCE_PENDING"}), encoding="utf-8")
            subprocess.run([sys.executable, str(SCRIPT), "launch", "--preflight", str(preflight), "--dummy", "--output", str(output)], check=True)
            value = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(value["status"], "DUMMY_CPU_ONLY_PASS")
            self.assertEqual(value["charged_gpu_training_seconds"], 0); self.assertFalse(value["optimizer_constructed"])
