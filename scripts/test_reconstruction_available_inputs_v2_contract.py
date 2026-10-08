from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "experiments" / "reconstruction_from_available_frozen_inputs_v2"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.contract_path = DEST / "RECONSTRUCTED_AVAILABLE_INPUTS_V2_001_CONTRACT.json"
        self.contract = json.loads(self.contract_path.read_text(encoding="utf-8"))
        self.binding_path = DEST / "RECONSTRUCTION_AVAILABLE_INPUTS_V2_LAUNCH_BINDING.json"
        self.binding = json.loads(self.binding_path.read_text(encoding="utf-8"))
        self.gate = json.loads((DEST / "SCIENTIFIC_EXECUTION_GATE_AVAILABLE_INPUTS_V2.json").read_text(encoding="utf-8"))

    def test_hash_bindings_and_new_identity(self) -> None:
        protocol = json.loads((DEST / "RECONSTRUCTION_FROM_AVAILABLE_FROZEN_INPUTS_V2_PROTOCOL.json").read_text(encoding="utf-8"))
        self.assertEqual("RECONSTRUCTION_FROM_AVAILABLE_FROZEN_INPUTS_V2", self.contract["protocol_id"])
        self.assertNotIn("FOUNDATION_V2_RECONSTRUCTION_AND_TARGETED_REPAIR_V1", self.contract["protocol_id"])
        self.assertEqual(digest(self.binding_path), self.contract["launch_binding_sha256"])
        self.assertEqual(digest(self.contract_path), self.gate["contract_sha256"])
        self.assertEqual(digest(self.contract_path), protocol["contract_sha256"])

    def test_gate_is_pending_and_nonce_is_fresh(self) -> None:
        self.assertFalse(self.gate["GPU_GATE_READY"])
        self.assertFalse(self.gate["AUTO_SCIENTIFIC_EXECUTION_AUTHORIZED"])
        self.assertIsNone(self.gate["authorization"]["directive_sha256"])
        self.assertEqual(self.contract["launch_nonce"], self.gate["launch_nonce"])
        self.assertNotEqual("3a52d466-e8ec-4d51-96d2-505853df2c2c", self.contract["launch_nonce"])

    def test_all_output_paths_are_new_and_inputs_are_boundary_safe(self) -> None:
        for value in self.contract["fresh_paths"].values():
            self.assertNotIn("reconstructed_foundation_v2_v1", value)
        self.assertEqual(11, len([x for x in self.binding["required_files"] if x["identity_role"] == "BASE_OR_TOKENIZER"]))
        all_inputs = [x["path"].lower() for x in self.binding["required_files"]] + [x["root"].lower() for x in self.binding["dataset_contracts"]] + [self.binding["replay_contract"]["path"].lower()]
        self.assertFalse(any(term in path for term in self.binding["scientific_boundaries"]["forbidden_path_terms"] for path in all_inputs))
        self.assertEqual(28800, self.binding["budget_contract"]["cap_seconds"])
        self.assertEqual(7200, self.binding["budget_contract"]["reservation_seconds"])


if __name__ == "__main__":
    unittest.main()
