import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "experiments/capability_repair_baseline_v1/connected_components_object_extraction_capability_pilot_v1"


def digest(relative):
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


class CountCodePairCpuIntegrationTest(unittest.TestCase):
    def test_real_launcher_worker_stop_before_model_load(self):
        source_files = [
            "scripts/connected_components_object_extraction_semantic_oracle.py",
            "scripts/decision_token_supervision.py",
            "scripts/e04_e_equal_slot_loss.py",
            "scripts/run_connected_component_count_decision_token_supervision_pilot_v1.py",
            "scripts/run_connected_component_count_code_pair_supervision_worker_v1.py",
            "scripts/run_connected_components_object_extraction_pilot_v1.py",
            "scripts/score_connected_components_object_extraction_pilot_v1.py",
            "src/inference/arc_native_io.py", "src/inference/nvarc_native.py", "src/training_data/pipeline.py",
        ]
        inputs = {
            "diverse_schedule": f"{BASE}/TREATMENT_SCHEDULE.json",
            "ood_prompts": f"{BASE}/OOD_INPUT_PROMPTS.json",
            "retention_prompts": f"{BASE}/RETENTION_INPUT_PROMPTS.json",
            "composition_prompts": f"{BASE}/COMPOSITION_SENTINEL_INPUT_PROMPTS.json",
            "protected_prompts": f"{BASE}/measurement_contract_v1/PROTECTED_CAPABILITY_INPUT_PROMPTS.json",
        }
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "cpu-integration-output"
            actual_output = Path(temporary) / "future-production-output"
            binding = {
                "experiment_id": "CONNECTED_COMPONENT_COUNT_CODE_PAIR_SUPERVISION_PILOT_V1",
                "scientific_source_commit": "TEST_ONLY_UNCOMMITTED",
                "source_files": {path: digest(path) for path in source_files},
                "inputs": {name: {"path": path, "sha256": digest(path)} for name, path in inputs.items()},
                "v7": {"checkpoint_manifest_path": "experiments/capability_repair_baseline_v1/CHECKPOINT_MANIFEST_REMOTE_V1.json", "checkpoint_manifest_sha256": digest("experiments/capability_repair_baseline_v1/CHECKPOINT_MANIFEST_REMOTE_V1.json")},
                "arms": {
                    "control": {"schedule_path": inputs["diverse_schedule"], "schedule_sha256": digest(inputs["diverse_schedule"]), "loss_allocation": {"kind": "UNIFORM_SEQUENCE_LOSS"}},
                    "treatment": {"schedule_path": inputs["diverse_schedule"], "schedule_sha256": digest(inputs["diverse_schedule"]), "loss_allocation": {"kind": "COUNT_CODE_PAIR_REWEIGHTED", "decision_token_multiplier": 2.0}},
                },
                "training": {"torch_seed": 2000031, "learning_rate": 5e-5, "updates_per_arm": 96, "slots_per_update": 4},
                "evaluation": {"fallback_ladder": [32, 16, 8, 4, 1], "max_new_tokens": 192, "context_window": 32768, "b1_subset_by_checkpoint": {}},
                "dataset_paths": {},
                "runtime": {"context_window": 32768},
                "execution": {
                    "worker_path": "scripts/run_connected_component_count_code_pair_supervision_worker_v1.py", "worker_sha256": digest("scripts/run_connected_component_count_code_pair_supervision_worker_v1.py"),
                    "launcher_path": "scripts/run_connected_component_count_decision_token_supervision_pilot_v1.py", "launcher_sha256": digest("scripts/run_connected_component_count_decision_token_supervision_pilot_v1.py"),
                    "runtime_python": sys.executable, "nonce": "cpu-integration-nonce-0001", "remote_output_root": str(actual_output).replace("\\", "/"),
                    "cpu_integration_root": str(output).replace("\\", "/"), "enforce_runtime_environment": False,
                    "training_cap_seconds": 1, "evaluation_cap_seconds": 1, "combined_cap_seconds": 2, "retry": False,
                    "terminal_receipt": "TERMINAL_RECEIPT.json", "authorization_required": True,
                },
            }
            binding_path = Path(temporary) / "binding.json"
            binding_path.write_text(json.dumps(binding), encoding="utf-8")
            result = subprocess.run([sys.executable, str(ROOT / "scripts/run_connected_component_count_decision_token_supervision_pilot_v1.py"), "--binding", str(binding_path), "--output-root", str(output), "--cpu-integration"], cwd=ROOT, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr or result.stdout)
            receipt = json.loads((output / "TERMINAL_RECEIPT.json").read_text())
            self.assertEqual(receipt["status"], "PASS_CPU_INTEGRATION_STOP_BEFORE_MODEL_LOAD")
            self.assertEqual(receipt["per_slot_weight_mass"], [60.0] * 4)
            self.assertFalse(receipt["model_loaded"])
            self.assertFalse(receipt["gpu_used"])
            self.assertFalse(receipt["sealed_sidecar_opened"])


if __name__ == "__main__":
    unittest.main()
