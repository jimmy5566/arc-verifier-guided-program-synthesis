from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("native_launcher_v2", ROOT / "scripts" / "launch_unified_native_model_capability_baseline_v2.py")
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(mod)


class NativeLauncherV2Tests(unittest.TestCase):
    @staticmethod
    def active_binding_path():
        bindings = sorted((ROOT / "experiments/capability_repair_baseline_v1/unified_native_model_capability_baseline_v1").glob("LAUNCH_BINDING_V*.json"))
        return bindings[-1]
    def test_command_has_only_no_update_worker_arguments(self):
        binding = {"worker_path": "scripts/run_unified_native_model_capability_baseline_v2.py", "manifest_path": "m.json", "discovery_path": "d.json"}
        command = mod.worker_command(binding, Path("/fresh"))
        self.assertIn("--runtime-seconds", command); self.assertIn("9000", command)
        self.assertNotIn("--train", command); self.assertNotIn("--optimizer", command)

    def test_binding_rejects_nonstandard_cap_before_launch(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "binding.json"
            path.write_text(json.dumps({"nonce":"n","output_root":"x","worker_path":"scripts/run_unified_native_model_capability_baseline_v2.py","worker_sha256":"x","manifest_path":"m","discovery_path":"d","runtime_cap_seconds":1,"expected_source_commit":"a" * 40}))
            with self.assertRaisesRegex(RuntimeError, "LAUNCH_BINDING_INVALID"):
                mod.load_binding(path)

    def test_actual_frozen_v5_binding_resolves_to_worker_command(self):
        binding_path = self.active_binding_path()
        binding = mod.load_binding(binding_path)
        command = mod.worker_command(binding, Path("/fresh"))
        self.assertEqual(binding["runtime_cap_seconds"], 9000)
        self.assertIn(str(ROOT / binding["worker_path"]), command)

    def test_main_with_actual_v5_binding_writes_success_and_failure_and_cleans_lock(self):
        binding_path = self.active_binding_path()
        original_load = mod.load_binding
        class Child:
            def __init__(self, code): self.code = code
            def wait(self, timeout): return self.code
        for code, expected in ((0, "SUCCESS"), (1, "FAILURE")):
            with self.subTest(code=code), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary) / "fresh"
                def load_actual(path, *, expected_checkout=None):
                    value = original_load(path, expected_checkout=expected_checkout)
                    value["output_root"] = str(root)
                    return value
                with patch.object(mod, "git_head", return_value="a" * 40), patch.object(mod, "load_binding", side_effect=load_actual), patch.object(mod.subprocess, "Popen", return_value=Child(code)), patch.object(sys, "argv", ["launcher", "--binding", str(binding_path), "--expected-source-commit", "a" * 40]):
                    self.assertEqual(mod.main(), 0 if code == 0 else 1)
                receipt = json.loads((root / "TERMINAL_RECEIPT.json").read_text())
                self.assertEqual(receipt["status"], expected)
                self.assertEqual(receipt["expected_source_commit"], "a" * 40)
                self.assertFalse((root.parent / ("." + root.name + ".lock")).exists())

    def test_launcher_module_is_cpu_only(self):
        source = (ROOT / "scripts" / "launch_unified_native_model_capability_baseline_v2.py").read_text(encoding="utf-8")
        self.assertNotRegex(source, r"(?m)^\\s*(from|import)\\s+(torch|transformers|peft)\\b")


if __name__ == "__main__":
    unittest.main()
