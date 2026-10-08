from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.collect_capability_repair_baseline_v1 import verify_decoder_validity_evidence


class MinimalGateTests(unittest.TestCase):
    def test_decoder_contract_is_adapter_independent_but_target_blind(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            evidence = Path(root) / "evidence.json"
            evidence.write_text(json.dumps({
                "status": "VALID_MEASUREMENT_PASS",
                "target_blind": True,
                "correctness_scoring_present": False,
                "raw_token_evidence": True,
                "checkpoint_manifest_sha256": "historical-adapter-only-provenance",
            }), encoding="utf-8")
            digest = hashlib.sha256(evidence.read_bytes()).hexdigest()
            parsed = verify_decoder_validity_evidence(evidence, digest)
            self.assertTrue(parsed["target_blind"])

    def test_decoder_evidence_rejects_missing_raw_token_auditability(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            evidence = Path(root) / "evidence.json"
            evidence.write_text(json.dumps({
                "status": "VALID_MEASUREMENT_PASS",
                "target_blind": True,
                "correctness_scoring_present": False,
                "raw_token_evidence": False,
            }), encoding="utf-8")
            digest = hashlib.sha256(evidence.read_bytes()).hexdigest()
            with self.assertRaisesRegex(RuntimeError, "VALID_MEASUREMENT_GATE_FAIL"):
                verify_decoder_validity_evidence(evidence, digest)


if __name__ == "__main__":
    unittest.main()
