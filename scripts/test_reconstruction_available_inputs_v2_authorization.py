from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reconstruction_available_inputs_v2_authorization import consume_once, validate


class AuthorizationTest(unittest.TestCase):
    def fixtures(self):
        contract = {"protocol_id": "RECONSTRUCTION_FROM_AVAILABLE_FROZEN_INPUTS_V2", "round_id": "RECONSTRUCTED_AVAILABLE_INPUTS_V2_001", "launch_nonce": "fresh-nonce"}
        digest = "contract-digest"
        directive = {"decision": "CONTINUE", "scientific_training_authorized": True, "protocol_id": contract["protocol_id"], "scope": contract["round_id"], "contract_sha256": digest}
        gate = {"protocol_id": contract["protocol_id"], "GPU_GATE_READY": True, "AUTO_SCIENTIFIC_EXECUTION_AUTHORIZED": True, "authorization": {"directive_sha256": "directive-digest", "protocol_id": contract["protocol_id"], "contract_sha256": digest, "launch_nonce": "fresh-nonce"}}
        return contract, digest, directive, gate

    def test_exact_authorization_is_required(self) -> None:
        contract, digest, directive, gate = self.fixtures()
        validate(contract, digest, directive, "directive-digest", gate)
        directive["contract_sha256"] = "other"
        with self.assertRaisesRegex(RuntimeError, "DIRECTIVE_CONTRACT_MISMATCH"):
            validate(contract, digest, directive, "directive-digest", gate)

    def test_nonce_is_atomic_and_single_use(self) -> None:
        contract, digest, directive, gate = self.fixtures()
        with tempfile.TemporaryDirectory() as raw:
            destination = Path(raw)
            receipt = consume_once(destination, contract, digest, "directive-digest")
            self.assertEqual("CONSUMED_ONCE", json.loads(receipt.read_text(encoding="utf-8"))["status"])
            with self.assertRaisesRegex(RuntimeError, "LAUNCH_NONCE_ALREADY_CONSUMED"):
                consume_once(destination, contract, digest, "directive-digest")

    def test_non_qualifying_directive_fails_closed(self) -> None:
        contract, digest, directive, gate = self.fixtures()
        directive["decision"] = "REQUIRE_CHANGES"
        with self.assertRaisesRegex(RuntimeError, "DIRECTIVE_DECISION_NOT_QUALIFYING"):
            validate(contract, digest, directive, "directive-digest", gate)


if __name__ == "__main__":
    unittest.main()
