import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts import e04_orientation_v3_no_update as e
from scripts import score_e04_orientation_v3_fixed_baseline as scorer

class E04V3NoUpdateTest(unittest.TestCase):
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

if __name__ == "__main__":
    unittest.main()
