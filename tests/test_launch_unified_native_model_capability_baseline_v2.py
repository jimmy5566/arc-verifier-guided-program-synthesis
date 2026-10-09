from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("native_launcher_v2", ROOT / "scripts" / "launch_unified_native_model_capability_baseline_v2.py")
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(mod)


class NativeLauncherV2Tests(unittest.TestCase):
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

    def test_launcher_module_is_cpu_only(self):
        source = (ROOT / "scripts" / "launch_unified_native_model_capability_baseline_v2.py").read_text(encoding="utf-8")
        self.assertNotRegex(source, r"(?m)^\\s*(from|import)\\s+(torch|transformers|peft)\\b")


if __name__ == "__main__":
    unittest.main()
