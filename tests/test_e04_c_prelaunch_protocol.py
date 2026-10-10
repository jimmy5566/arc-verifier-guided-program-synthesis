import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/capability_repair_baseline_v1/e04_c_matched_fixed_turn_rotation_repair_pilot_v1"


class E04CPrelaunchProtocolTests(unittest.TestCase):
    def test_frozen_dependencies_and_single_variable_contract(self):
        protocol = json.loads((BASE / "E04_C_PRELAUNCH_PROTOCOL_V1.json").read_text())
        schedule = BASE / "schedule_freeze_v1"
        self.assertEqual(protocol["status"], "FROZEN_CPU_ONLY_PRELAUNCH_PENDING_DIRECTOR_REVIEW")
        self.assertFalse(protocol["execution"]["execution_authorized"])
        self.assertIn("only_scientific_variable", protocol["arms"])
        self.assertEqual(protocol["recipe"]["optimizer_steps_per_arm"], 96)
        self.assertEqual(protocol["recipe"]["gradient_accumulation"], 4)
        for relative, expected in [
            ("MATCHED_SCHEDULE_BINDING_V1.json", protocol["schedule"]["binding_sha256"]),
            ("NATIVE_TOKEN_STATIC_PREFLIGHT_V1.json", protocol["schedule"]["native_token_preflight_sha256"]),
        ]:
            self.assertEqual(hashlib.sha256((schedule / relative).read_bytes()).hexdigest(), expected)

    def test_decision_rule_keeps_independent_claim_separate(self):
        protocol = json.loads((BASE / "E04_C_PRELAUNCH_PROTOCOL_V1.json").read_text())
        self.assertEqual(protocol["cohorts"]["independent_non_gold_sentinel"]["status"], "UNAVAILABLE_NO_ELIGIBLE_FROZEN_COHORT_IDENTIFIED")
        self.assertEqual(protocol["decision_rules"]["bootstrap"]["replicates"], 10000)
        self.assertTrue(protocol["decision_rules"]["stop"]["no_automatic_expansion"])


if __name__ == "__main__":
    unittest.main()
