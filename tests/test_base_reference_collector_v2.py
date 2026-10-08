from __future__ import annotations
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COLLECTOR = ROOT / "scripts" / "collect_base_reference_remote_first_v2.py"
CONTRACT = ROOT / "experiments" / "base_only_targeted_repair_remote_first_v2" / "execution_contract_v6" / "BASELINE_INFERENCE_CONTRACT_V2.json"
spec = importlib.util.spec_from_file_location("collector_v2", COLLECTOR)
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

RECORD = {"episode_id": "redaction-001", "role": "TARGETED_EVALUATION", "task": {
    "train": [{"input": [[1, 2]], "output": [[3, 4]]}],
    "test": [{"input": [[5, 6]], "output": [[9, 8]]}],
}}

class CollectorV2Tests(unittest.TestCase):
    def test_observation_prompt_redacts_test_output_and_uses_native_grid(self) -> None:
        contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
        observation = mod.observation(RECORD); text = mod.prompt(RECORD, contract)
        self.assertNotIn("output", observation["test"][0])
        self.assertNotIn("98", text)
        self.assertIn("<|im_start|>user\n12<|im_end|>", text)
        self.assertIn("<|im_start|>assistant\n34<|im_end|>", text)
        self.assertTrue(text.endswith("<|im_start|>assistant\n"))
    def test_parser_accepts_exact_native_grid_and_rejects_prose_json_and_ragged(self) -> None:
        self.assertEqual(mod.parse_grid("12\n34\n"), [[1, 2], [3, 4]])
        for invalid in ("[[1,2]]", "answer\n12", "12\n3", "1 2"):
            self.assertIsNone(mod.parse_grid(invalid))
    def test_duplicate_and_bad_file_identity_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "target.jsonl"
            path.write_text(json.dumps(RECORD) + "\n" + json.dumps(RECORD) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "DUPLICATE"):
                mod.read_rows(path, {"TARGETED_EVALUATION"})
            file = Path(td) / "identity.txt"; file.write_text("actual", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "HASH_MISMATCH"):
                mod.verify_file(file, "0" * 64)
    def test_cpu_mock_and_runtime_cap_are_non_training(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); target = root / "target.jsonl"; retention = root / "retention.jsonl"
            target.write_text(json.dumps(RECORD) + "\n", encoding="utf-8")
            r = dict(RECORD); r["episode_id"] = "retention-001"; r["role"] = "RETENTION_SENTINEL"
            retention.write_text(json.dumps(r) + "\n", encoding="utf-8")
            binding = root / "binding.json"; binding.write_text(json.dumps({"base_path": "/workspace/arc2/models/qwen3_4b_grids15_sft139", "base_files": {}}), encoding="utf-8")
            out, receipt = root / "out.json", root / "receipt.json"
            cmd = [sys.executable, str(COLLECTOR), "--target-dev", str(target), "--retention", str(retention), "--binding", str(binding), "--contract", str(CONTRACT), "--output", str(out), "--receipt", str(receipt), "--cpu-mock"]
            self.assertEqual(subprocess.run(cmd, cwd=ROOT).returncode, 0)
            self.assertEqual(json.loads(out.read_text())["status"], "CPU_MOCK_PASS_NO_MODEL")
            self.assertFalse(json.loads(receipt.read_text())["scientific_training_started"])
            out.unlink(); receipt.unlink()
            capped = subprocess.run(cmd + ["--cpu-mock-simulate-runtime-cap"], cwd=ROOT)
            self.assertNotEqual(capped.returncode, 0)
            self.assertEqual(json.loads(out.read_text())["status"], "PARTIAL_RUNTIME_CAP")
            self.assertNotEqual(json.loads(receipt.read_text())["status"], "COLLECTED_PASS")
    def test_forbidden_path_missing_authorization_and_nonoverwrite_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); final = root / "FINAL_AUDIT.jsonl"
            final.write_text(json.dumps(RECORD) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "FORBIDDEN_INPUT_PATH"):
                mod.read_rows(final, {"TARGETED_EVALUATION"})
            target = root / "target.jsonl"; target.write_text(json.dumps(RECORD) + "\n", encoding="utf-8")
            retention = root / "retention.jsonl"; rr = dict(RECORD); rr["episode_id"] = "ret"; rr["role"] = "RETENTION_SENTINEL"; retention.write_text(json.dumps(rr) + "\n", encoding="utf-8")
            binding = root / "binding.json"; binding.write_text(json.dumps({"base_path": "/workspace/arc2/models/qwen3_4b_grids15_sft139", "base_files": {}}), encoding="utf-8")
            out, receipt = root / "out.json", root / "receipt.json"; out.write_text("already", encoding="utf-8")
            cmd = [sys.executable, str(COLLECTOR), "--target-dev", str(target), "--retention", str(retention), "--binding", str(binding), "--contract", str(CONTRACT), "--output", str(out), "--receipt", str(receipt), "--cpu-mock"]
            self.assertNotEqual(subprocess.run(cmd, cwd=ROOT).returncode, 0)
            out.unlink()
            # Non-mock mode refuses before any model import with no Director authorization.
            self.assertNotEqual(subprocess.run(cmd[:-1], cwd=ROOT).returncode, 0)

if __name__ == "__main__":
    unittest.main()
