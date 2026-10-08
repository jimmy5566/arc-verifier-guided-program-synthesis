from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("forward_collector", ROOT / "scripts" / "collect_capability_repair_baseline_v1.py")
assert SPEC and SPEC.loader
collector = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(collector)


class ForwardLaunchV3Tests(unittest.TestCase):
    def test_review_must_bind_every_launch_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); contract = root / "contract.json"; brief = root / "brief.json"; review = root / "review.json"
            contract.write_text("{}", encoding="utf-8")
            brief.write_text("{}", encoding="utf-8")
            launch = {
                "governor_review_path": str(review), "contract_path": str(contract), "contract_sha256": "contract",
                "authorization_request_sha256": "request", "review_brief_path": str(brief), "baseline_identity": {"sha256": "baseline"},
                "checkpoint_manifest_sha256": "manifest-file", "checkpoint_manifest_identity": "manifest",
                "datasets": {"TARGET_DEV": {"sha256": "target", "rows": 192}, "RETENTION_SENTINEL": {"sha256": "retention", "rows": 96}},
                "source_commit": "source", "executable": {"sha256": "executable"}, "nonce_sha256": "nonce",
                "output_root": "/fresh/output", "receipt_path": "/fresh/receipt", "runtime_cap_seconds": 7200,
            }
            expected = {
                "decision": "CONTINUE_CONTROLLER", "authorization_request_sha256": "request", "reviewed_brief_sha256": collector.sha(brief), "launch_contract_file_sha256": collector.sha(contract),
                "launch_contract_identity": "contract", "baseline_identity_sha256": "baseline", "checkpoint_manifest_file_sha256": "manifest-file",
                "checkpoint_manifest_identity": "manifest", "target_dev_sha256": "target", "target_dev_rows": 192,
                "retention_sha256": "retention", "retention_rows": 96, "source_commit": "source", "executable_sha256": "executable",
                "nonce_sha256": "nonce", "output_root": "/fresh/output", "receipt_path": "/fresh/receipt", "runtime_cap_seconds": 7200,
            }
            review.write_text(json.dumps(expected), encoding="utf-8")
            collector.verify_governor_review(launch, review)
            expected["nonce_sha256"] = "wrong"; review.write_text(json.dumps(expected), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "BINDING_MISMATCH"):
                collector.verify_governor_review(launch, review)

    def test_entrypoint_rejects_argv_environment_or_cap_substitution(self) -> None:
        launch = {"argv": [str(Path(sys.executable).resolve()), str(ROOT / "worker.py"), "--x", "1"], "runtime_cap_seconds": 7200, "environment": {"PYTHONHASHSEED": "0"}}
        args = type("Args", (), {"runtime_limit_seconds": 7200})()
        with mock.patch.object(sys, "argv", [str(ROOT / "worker.py"), "--x", "1"]), mock.patch.dict(os.environ, {"PYTHONHASHSEED": "0"}, clear=False):
            collector.verify_entrypoint(launch, args)
            with mock.patch.object(sys, "argv", [str(ROOT / "worker.py"), "--x", "2"]):
                with self.assertRaisesRegex(RuntimeError, "ARGV_BINDING_MISMATCH"):
                    collector.verify_entrypoint(launch, args)
        args.runtime_limit_seconds = 1
        with mock.patch.object(sys, "argv", [str(ROOT / "worker.py"), "--x", "1"]), mock.patch.dict(os.environ, {"PYTHONHASHSEED": "0"}, clear=False):
            with self.assertRaisesRegex(RuntimeError, "RUNTIME_CAP_BINDING_MISMATCH"):
                collector.verify_entrypoint(launch, args)


if __name__ == "__main__":
    unittest.main()
