import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts import e04_c_matched_rotation_execution as execution
from scripts import prepare_e04_c_execution_binding as binding_builder
from scripts import run_e04_c_matched_rotation_repair_pilot as worker


class E04CExecutionContractTests(unittest.TestCase):
    def test_cpu_schedule_proof_matches_frozen_totals_and_replay(self):
        proof = execution.static_schedule_preflight()
        self.assertEqual(proof["status"], "PASS_CPU_NATIVE_SCHEDULE")
        self.assertEqual(proof["arm_totals"][execution.ARMS[0]], {"total_tokens": 207360, "supervised_tokens": 28800})
        self.assertEqual(proof["arm_totals"][execution.ARMS[1]], {"total_tokens": 207360, "supervised_tokens": 28800})
        self.assertEqual(proof["protected_replay_slots"], 288)

    def test_binding_is_fresh_and_preimport_validation_does_not_load_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "binding.json"
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=execution.ROOT, text=True).strip()
            previous = list(__import__("sys").argv)
            try:
                __import__("sys").argv = ["binding", "--output", str(out), "--worker-source-commit", commit]
                binding_builder.main()
            finally:
                __import__("sys").argv = previous
            binding = json.loads(out.read_text())
            self.assertTrue(binding["execution_authorized"])
            self.assertEqual(binding["director_response_sha256"], "121f314f1651929d2a857d096bcc43a89676dcbe52235fb6b49d0520aca90a4f")
            self.assertEqual(binding["arm_order"], list(execution.ARMS))
            self.assertEqual(binding["arm_output_roots"], {arm: binding["output_root"] + "/arms/" + arm for arm in execution.ARMS})
            self.assertEqual(binding["static_preflight"]["status"], "PASS_CPU_NATIVE_SCHEDULE")
            worker.load_binding(out, Path(binding["output_root"]), commit)

    def test_malformed_protected_replay_fails_before_model_import(self):
        protocol, frozen, static, schedules = execution.load_frozen_inputs()
        original = schedules[execution.ARMS[1]]["episodes"][96]["treatment_task"]
        schedules[execution.ARMS[1]]["episodes"][96]["treatment_task"] = {"train": [], "test": []}
        # This mutation makes the schedule invalid at the pre-model serialization gate.
        with self.assertRaises(execution.E04CFailure):
            execution.build_arm_samples(execution.ARMS[1], schedules[execution.ARMS[1]])
        schedules[execution.ARMS[1]]["episodes"][96]["treatment_task"] = original




if __name__ == "__main__":
    unittest.main()