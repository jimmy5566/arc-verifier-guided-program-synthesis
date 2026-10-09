from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("unified_preflight_v2", ROOT / "scripts" / "preflight_unified_native_model_capability_baseline_v2.py")
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(mod)


class UnifiedPreflightV2Tests(unittest.TestCase):
    def test_source_binding_fails_closed_before_runtime_access(self):
        prior = os.environ.pop("ARC2_EXPECTED_SOURCE_COMMIT", None)
        try:
            with self.assertRaisesRegex(RuntimeError, "SOURCE_COMMIT_ENV_BINDING_MISMATCH"):
                mod.validate(Path("does-not-matter"), "a" * 40)
        finally:
            if prior is not None:
                os.environ["ARC2_EXPECTED_SOURCE_COMMIT"] = prior

    def test_verify_entries_rejects_wrong_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); (root / "a").write_bytes(b"a")
            with self.assertRaisesRegex(RuntimeError, "RUNTIME_IDENTITY_MISMATCH"):
                mod.verify_entries(root, [{"name": "a", "bytes": 1, "sha256": "0" * 64}])

    def test_module_is_cpu_only(self):
        source = (ROOT / "scripts" / "preflight_unified_native_model_capability_baseline_v2.py").read_text(encoding="utf-8")
        self.assertNotRegex(source, r"(?m)^\\s*(from|import)\\s+torch\\b")
        self.assertNotRegex(source, r"(?m)^\\s*(from|import)\\s+transformers\\b")


if __name__ == "__main__":
    unittest.main()
