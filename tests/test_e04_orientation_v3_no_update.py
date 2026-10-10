import hashlib
import json
import tempfile
import unittest
import subprocess
from pathlib import Path

from scripts import e04_orientation_v3_no_update as e
from scripts import score_e04_orientation_v3_fixed_baseline as scorer

class E04V3NoUpdateTest(unittest.TestCase):
    def test_worker_exposes_src_inference_without_model_import(self):
        from scripts import run_e04_orientation_v3_fixed_baseline as worker
        from inference.nvarc_native import parse_native_grid
        self.assertIsNotNone(worker.ROOT / "src")
        self.assertEqual(parse_native_grid("1"), [[1]])

    def test_contract_is_target_blind_and_cpu_valid(self):
        contract = e.freeze_contract()
        self.assertEqual(contract["requested_batch_size"], 32)
        self.assertEqual(contract["batch1_sensitivity_rows"], 64)
        outcome = e.validate_contract()
        self.assertEqual(outcome["status"], "PASS_NO_MODEL_IMPORT")
        self.assertFalse(outcome["model_imported"])
        self.assertFalse(outcome["target_sidecar_accessed"])

    def _raw(self, prompts, indexes, mutate=None):
        rows = []
        for index in indexes:
            grid = None if mutate != index else [[1]]
            rows.append({
                "row_index": index,
                "input_sha256": hashlib.sha256(e.canon(prompts[index])).hexdigest(),
                "text": "" if grid is None else "1",
                "parser_valid": grid is not None,
                "parsed_grid": grid,
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "elapsed_seconds": 0.0,
                "effective_batch_size": 32 if len(indexes) == 768 else 1,
            })
        return rows

    def test_scorer_requires_raw_freeze_then_scores(self):
        contract = json.loads(e.CONTRACT.read_text())
        prompts = e.read_jsonl(e.ROOT / contract["input_prompts_path"])
        subset = json.loads((e.ROOT / contract["batch1_sensitivity_manifest_path"]).read_text())
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            primary, b1 = tmp / "primary.jsonl", tmp / "b1.jsonl"
            e.atomic_jsonl(primary, self._raw(prompts, list(range(768))))
            e.atomic_jsonl(b1, self._raw(prompts, subset["row_indexes"]))
            receipt = tmp / "freeze.json"
            e.atomic_json(receipt, {
                "status": "RAW_GENERATIONS_FROZEN_NO_TARGETS",
                "target_sidecar_accessed": False,
                "primary_raw_sha256": e.sha_path(primary),
                "batch1_raw_sha256": e.sha_path(b1),
            })
            result = scorer.score(primary, b1, receipt, tmp / "result.json")
            self.assertEqual(result["status"], "SCORED_DEVELOPMENT_EVIDENCE")
            self.assertEqual(result["batch1_gate"], "PASS")

    def test_scorer_classifies_b1_disagreement_inconclusive(self):
        contract = json.loads(e.CONTRACT.read_text())
        prompts = e.read_jsonl(e.ROOT / contract["input_prompts_path"])
        subset = json.loads((e.ROOT / contract["batch1_sensitivity_manifest_path"]).read_text())
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            primary, b1 = tmp / "primary.jsonl", tmp / "b1.jsonl"
            e.atomic_jsonl(primary, self._raw(prompts, list(range(768))))
            e.atomic_jsonl(b1, self._raw(prompts, subset["row_indexes"], mutate=subset["row_indexes"][0]))
            receipt = tmp / "freeze.json"
            e.atomic_json(receipt, {
                "status": "RAW_GENERATIONS_FROZEN_NO_TARGETS",
                "target_sidecar_accessed": False,
                "primary_raw_sha256": e.sha_path(primary),
                "batch1_raw_sha256": e.sha_path(b1),
            })
            result = scorer.score(primary, b1, receipt, tmp / "result.json")
            self.assertEqual(result["status"], "NUMERICAL_OR_BATCH_SENSITIVITY_INCONCLUSIVE")
            self.assertEqual(len(result["batch1_disagreement_row_indexes"]), 1)

    def test_raw_schema_rejects_missing_parse_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw.jsonl"
            e.atomic_jsonl(raw, [{"row_index": 0, "input_sha256": "x", "text": "", "parser_valid": True,
                                  "parsed_grid": None, "prompt_tokens": 1, "completion_tokens": 1,
                                  "elapsed_seconds": 0.0, "effective_batch_size": 1}])
            with self.assertRaisesRegex(e.E04ExecutionFailure, "E04_RAW_PARSE_CONTRACT"):
                e.validate_raw_generation(raw, [0])


    def test_binding_preflight_accepts_exact_detached_source_without_model_import(self):
        from scripts import run_e04_orientation_v3_fixed_baseline as worker
        contract = json.loads(e.CONTRACT.read_text())
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "fresh"
            binding = {
                "schema_version": 1,
                "protocol_id": contract["protocol_id"],
                "authorization_id": "E04_V3_IMPORT_PATH_REPLACEMENT_ONE_SHOT_CONDITIONAL_20261010", "director_response_sha256": "0" * 64,
                "execution_authorized": True,
                "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=e.ROOT, text=True).strip(),
                "worker_sha256": e.sha_path(e.ROOT / "scripts/run_e04_orientation_v3_fixed_baseline.py"),
                "contract_path": str(e.CONTRACT.relative_to(e.ROOT)).replace("\\\\", "/"),
                "contract_sha256": e.sha_path(e.CONTRACT),
                "runtime_config_path": "experiments/capability_repair_baseline_v1/e04_orientation_fixed_demonstration_baseline_v3/E04_V3_NATIVE_GREEDY_RUNTIME_CONFIG_V1.json",
                "runtime_config_sha256": e.sha_path(e.ROOT / "experiments/capability_repair_baseline_v1/e04_orientation_fixed_demonstration_baseline_v3/E04_V3_NATIVE_GREEDY_RUNTIME_CONFIG_V1.json"),
                "checkpoint_manifest_path": "experiments/capability_repair_baseline_v1/CHECKPOINT_MANIFEST_REMOTE_V1.json",
                "checkpoint_manifest_sha256": e.sha_path(e.ROOT / "experiments/capability_repair_baseline_v1/CHECKPOINT_MANIFEST_REMOTE_V1.json"),
                "native_config_dir": str(e.ROOT / "configs/nvarc_native_846d0198"),
                "native_config_provenance_path": "configs/NVARC_NATIVE_INTERFACE_846D0198_PROVENANCE.json",
                "native_config_provenance_sha256": e.sha_path(e.ROOT / "configs/NVARC_NATIVE_INTERFACE_846D0198_PROVENANCE.json"),
                "native_config_runtime_identity_path": "experiments/capability_repair_baseline_v1/e04_orientation_fixed_demonstration_baseline_v3/E04_V3_RUNTIME_NATIVE_CONFIG_IDENTITY_V1.json",
                "native_config_runtime_identity_sha256": e.sha_path(e.ROOT / "experiments/capability_repair_baseline_v1/e04_orientation_fixed_demonstration_baseline_v3/E04_V3_RUNTIME_NATIVE_CONFIG_IDENTITY_V1.json"),
                "output_root": str(out), "nonce": "unit-test-nonce",
                "hard_runtime_cap_seconds": 1800, "jobs": 1, "retry": False,
            }
            bp = Path(tmp) / "binding.json"
            e.atomic_json(bp, binding)
            self.assertEqual(worker.load_binding(bp, out)["nonce"], "unit-test-nonce")

    def test_runtime_native_identity_validates_runtime_and_vendored_bytes_without_model_import(self):
        from scripts import run_e04_orientation_v3_fixed_baseline as worker
        manifest = e.ROOT / "experiments/capability_repair_baseline_v1/e04_orientation_fixed_demonstration_baseline_v3/E04_V3_RUNTIME_NATIVE_CONFIG_IDENTITY_V1.json"
        binding = {"native_config_runtime_identity_path": str(manifest.relative_to(e.ROOT)).replace("\\", "/"),
                   "native_config_dir": str(e.ROOT / "configs/nvarc_native_846d0198")}
        worker.verify_runtime_native_config_identity(binding)
        bad = dict(binding); bad["native_config_dir"] = str(e.ROOT / "configs")
        with self.assertRaisesRegex(e.E04ExecutionFailure, "E04_NATIVE_CONFIG_RUNTIME_IDENTITY"):
            worker.verify_runtime_native_config_identity(bad)

    def test_binding_rejects_wrong_execution_identity(self):
        from scripts import run_e04_orientation_v3_fixed_baseline as worker
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "fresh"
            binding = {
                "schema_version": 1, "protocol_id": "E04_ORIENTATION_V3_FIXED_INDEPENDENT_DEMONSTRATION_BASELINE",
                "authorization_id": "E04_V3_IMPORT_PATH_REPLACEMENT_ONE_SHOT_CONDITIONAL_20261010", "director_response_sha256": "0" * 64,
                "execution_authorized": True, "source_commit": "wrong", "worker_sha256": "wrong",
                "contract_path": str(e.CONTRACT.relative_to(e.ROOT)).replace("\\\\", "/"), "contract_sha256": e.sha_path(e.CONTRACT),
                "runtime_config_path": "experiments/capability_repair_baseline_v1/e04_orientation_fixed_demonstration_baseline_v3/E04_V3_NATIVE_GREEDY_RUNTIME_CONFIG_V1.json", "runtime_config_sha256": e.sha_path(e.ROOT / "experiments/capability_repair_baseline_v1/e04_orientation_fixed_demonstration_baseline_v3/E04_V3_NATIVE_GREEDY_RUNTIME_CONFIG_V1.json"),
                "checkpoint_manifest_path": "experiments/capability_repair_baseline_v1/CHECKPOINT_MANIFEST_REMOTE_V1.json",
                "checkpoint_manifest_sha256": e.sha_path(e.ROOT / "experiments/capability_repair_baseline_v1/CHECKPOINT_MANIFEST_REMOTE_V1.json"),
                "native_config_dir": str(e.ROOT / "configs/nvarc_native_846d0198"),
                "native_config_provenance_path": "configs/NVARC_NATIVE_INTERFACE_846D0198_PROVENANCE.json",
                "native_config_provenance_sha256": e.sha_path(e.ROOT / "configs/NVARC_NATIVE_INTERFACE_846D0198_PROVENANCE.json"),
                "native_config_runtime_identity_path": "experiments/capability_repair_baseline_v1/e04_orientation_fixed_demonstration_baseline_v3/E04_V3_RUNTIME_NATIVE_CONFIG_IDENTITY_V1.json",
                "native_config_runtime_identity_sha256": e.sha_path(e.ROOT / "experiments/capability_repair_baseline_v1/e04_orientation_fixed_demonstration_baseline_v3/E04_V3_RUNTIME_NATIVE_CONFIG_IDENTITY_V1.json"),
                "output_root": str(out), "nonce": "unit-test-nonce", "hard_runtime_cap_seconds": 1800, "jobs": 1, "retry": False,
            }
            bp = Path(tmp) / "binding.json"
            e.atomic_json(bp, binding)
            with self.assertRaisesRegex(e.E04ExecutionFailure, "E04_WORKER_HASH"):
                worker.load_binding(bp, out)
if __name__ == "__main__":
    unittest.main()




