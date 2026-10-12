import copy
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/capability_repair_baseline_v1/connected_components_object_extraction_capability_pilot_v1"


def load(relative, name):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


WORKER = load("scripts/run_connected_components_object_extraction_pilot_v1.py", "e1_worker_v7_test")
SCORER = load("scripts/score_connected_components_object_extraction_pilot_v1.py", "e1_scorer_v7_test")
LAUNCHER = load("scripts/launch_connected_components_object_extraction_external_auth_v3.py", "e1_launcher_v7_test")
ORACLE = load("scripts/connected_components_object_extraction_semantic_oracle.py", "e1_oracle_v7_test")


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ConnectedComponentsProductionPathV7(unittest.TestCase):
    def binding(self, output):
        source = BASE / "execution_binding_v4/EXECUTION_BINDING_EXTERNAL_AUTH_V3.json"
        binding = json.loads(source.read_text())
        binding["worker_sha256"] = file_sha(ROOT / binding["worker_path"])
        binding["launcher_sha256"] = file_sha(ROOT / binding["launcher_path"])
        binding["score_entrypoint_sha256"] = file_sha(ROOT / binding["score_entrypoint_path"])
        successor = BASE / "scientific_successor_v2"
        validation = successor / "SCIENTIFIC_CONSTRUCT_VALIDATION.json"
        composition = successor / "COMPOSITION_SENTINEL_INPUT_PROMPTS.json"
        binding["scientific_construct_validation_path"] = str(validation.relative_to(ROOT)).replace("\\", "/")
        binding["scientific_construct_validation_sha256"] = file_sha(validation)
        binding["dataset_paths"] = {"COMPOSITION": str(composition.relative_to(ROOT)).replace("\\", "/")}
        binding["inputs"][binding["scientific_construct_validation_path"]] = file_sha(validation)
        binding["inputs"][binding["dataset_paths"]["COMPOSITION"]] = file_sha(composition)
        binding["evaluation"]["batch1_gate_records_by_checkpoint"] = {"V7": 8, "CONTROL_NARROW_STRUCTURE": 12, "TREATMENT_STRUCTURE_DIVERSE": 12}
        binding["runtime"]["python_executable"] = "/root/arc2_r5_f0353c17/.venv-r5/bin/python"
        binding["runtime"]["pythonpath_prepend"] = ["/root/arc2_e1_v7_bnb_overlay"]
        binding["output"]["remote_output_root"] = output.as_posix()
        return binding

    def test_all_384_training_slots_and_first_replay_transition(self):
        report = WORKER.validate_training_schedules({})
        self.assertEqual(set(report), {"CONTROL_NARROW_STRUCTURE", "TREATMENT_STRUCTURE_DIVERSE"})
        for arm in report:
            self.assertEqual(report[arm]["slots"], 384)
            self.assertEqual(report[arm]["first_replay_slot"], 96)
            self.assertIn("SERIALIZER_VIEW:", report[arm]["first_replay_sample_id"])
        self.assertEqual(report["CONTROL_NARROW_STRUCTURE"]["supervised_tokens"], report["TREATMENT_STRUCTURE_DIVERSE"]["supervised_tokens"])

    def test_frozen_b1_identity_not_positional_slice(self):
        with tempfile.TemporaryDirectory() as raw:
            binding = self.binding(Path(raw) / "out")
            for tag, arm in (("V7", None), ("CONTROL_NARROW_STRUCTURE", "CONTROL_NARROW_STRUCTURE"), ("TREATMENT_STRUCTURE_DIVERSE", "TREATMENT_STRUCTURE_DIVERSE")):
                selected = WORKER._select_b1_items(WORKER._datasets(binding, arm), binding, tag)
                self.assertEqual([row["record_id"] for row in selected], binding["evaluation"]["b1_subset_by_checkpoint"][tag])
            self.assertEqual(len(WORKER._select_b1_items(WORKER._datasets(binding), binding, "V7")), 8)

    def test_retention_valid_and_v1_composition_invalid(self):
        retention = json.loads((BASE / "RETENTION_INPUT_PROMPTS.json").read_text())
        for row in retention:
            ORACLE.validate_demonstrations(row["task"], "NO_TRANSFORM_RETENTION")
            ORACLE.target_for_family("NO_TRANSFORM_RETENTION", row["task"]["test"][0]["input"])
        old = json.loads((BASE / "COMPOSITION_SENTINEL_INPUT_PROMPTS.json").read_text())
        with self.assertRaises(ValueError):
            ORACLE.target_for_family("COUNT_THEN_OBJECT_EXTRACTION", old[0]["task"]["test"][0]["input"])
        successor = json.loads((BASE / "scientific_successor_v2/COMPOSITION_SENTINEL_INPUT_PROMPTS.json").read_text())
        self.assertEqual(len({hashlib.sha256(SCORER.canon(row["task"]["test"][0]["input"])).hexdigest() for row in successor}), 16)
        for row in successor:
            ORACLE.validate_demonstrations(row["task"], "COUNT_THEN_OBJECT_EXTRACTION")
            ORACLE.target_for_family("COUNT_THEN_OBJECT_EXTRACTION", row["task"]["test"][0]["input"])

    def test_launcher_worker_scorer_complete_cpu_path(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            output = root / "output"
            binding_path = root / "binding.json"
            binding = self.binding(output)
            binding_path.write_text(json.dumps(binding))
            WORKER.preflight(binding_path)

            payload = {
                "binding_sha256": file_sha(binding_path), "execution_source_commit": binding["execution_source_commit"],
                "nonce": binding["output"]["nonce"], "output_root": output.as_posix(), "detached_jobs": 1, "retry": False,
                "training_cap_seconds": binding["runtime"]["training_cap_seconds"], "evaluation_cap_seconds": binding["runtime"]["evaluation_cap_seconds"],
                "combined_cap_seconds": binding["runtime"]["combined_cap_seconds"], "forbidden_targets": binding["target_isolation"]["forbidden"],
            }
            receipt_path = root / "authorization.json"
            receipt_path.write_text(json.dumps({"state": "UNCONSUMED", "payload": payload, "payload_sha256": hashlib.sha256(LAUNCHER.canon(payload)).hexdigest()}))
            LAUNCHER.validate(binding_path, receipt_path, output)
            class Child: pid = 42
            with patch.object(LAUNCHER.subprocess, "Popen", return_value=Child()) as popen:
                control, pid = LAUNCHER.launch(binding_path, receipt_path, output)
            self.assertEqual(pid, 42)
            self.assertEqual(popen.call_args.args[0][-1], "--execute")
            self.assertEqual(popen.call_args.args[0][0], binding["runtime"]["python_executable"])
            self.assertTrue(popen.call_args.kwargs["env"]["PYTHONPATH"].startswith("/root/arc2_e1_v7_bnb_overlay"))
            self.assertTrue((control / "LAUNCH_RECEIPT.json").is_file())

            # The mocked launcher leaves the scientific output root fresh.  A
            # complete synthetic raw freeze exercises production scorer logic.
            output.mkdir()
            expected, _, _, task_rows = SCORER._target_context(binding)
            raw_hashes = {}
            for tag in SCORER.ALL_TAGS:
                primary_tag = tag.removesuffix("_B1")
                arm = None if primary_tag == "V7" else primary_tag
                items = WORKER._datasets(binding, arm)
                if tag.endswith("_B1"):
                    items = WORKER._select_b1_items(items, binding, primary_tag)
                rows = []
                for item in items:
                    target = expected(item["record_id"])
                    wrong = primary_tag == "CONTROL_NARROW_STRUCTURE" and item["record_id"].startswith("OOD:") and int(item["record_id"].split(":")[1]) < 5
                    parsed = [[9]] if wrong else target
                    rows.append({"record_id": item["record_id"], "split": item["split"], "prompt_sha256": hashlib.sha256(SCORER.canon(task_rows[item["record_id"]])).hexdigest(), "generated_token_ids": [1, 2, 3], "parsed_grid": parsed, "parser_valid": True, "effective_batch_size": 1 if tag.endswith("_B1") else 32})
                path = output / f"{tag}_RAW.jsonl"
                path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
                raw_hashes[tag] = file_sha(path)
            (output / "RAW_FREEZE_RECEIPT.json").write_text(json.dumps({"status": "COMPLETE_RAW_FROZEN_NO_TARGETS", "targets_accessed": False, "raw_sha256": raw_hashes}))
            result = SCORER.score(binding_path, output)
            self.assertEqual(result["batch1_validity"]["status"], "PASS")
            self.assertEqual(result["paired_primary_ood_treatment_minus_control"]["delta_correct"], 5)
            self.assertEqual(result["promotion"]["decision"], "PROMOTE")

    def test_scorer_fails_before_targets_on_incomplete_raw_set(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); output = root / "output"; output.mkdir()
            binding = self.binding(output); binding_path = root / "binding.json"; binding_path.write_text(json.dumps(binding))
            one = output / "V7_RAW.jsonl"; one.write_text("")
            (output / "RAW_FREEZE_RECEIPT.json").write_text(json.dumps({"status": "COMPLETE_RAW_FROZEN_NO_TARGETS", "targets_accessed": False, "raw_sha256": {"V7": file_sha(one)}}))
            with self.assertRaisesRegex(RuntimeError, "RAW_ARTIFACT_SET"):
                SCORER.score(binding_path, output)
            self.assertFalse((output / "SCORE_RECEIPT.json").exists())


if __name__ == "__main__":
    unittest.main()
