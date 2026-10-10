import hashlib
import tempfile
import unittest
from unittest import mock
from pathlib import Path

from scripts import e04_v3_fixed64_batch_rung_localization as e
from scripts import launch_e04_v3_fixed64_batch_rung_localization as launch
from scripts import run_e04_v3_fixed64_batch_rung_localization as worker


class Fixed64LocalizationTest(unittest.TestCase):
    def _row(self, index, arm, tokens=(1, 15)):
        return {"row_index": index, "input_sha256": hashlib.sha256(str(index).encode()).hexdigest(),
                "generated_token_ids": list(tokens), "text": "1", "parser_valid": True, "parsed_grid": [[1]],
                "prompt_tokens": 9, "completion_tokens": len(tokens), "elapsed_seconds": 0.0,
                "effective_batch_size": arm, "physical_returned_token_ids": list(tokens), "pad_token_count": 0}

    def test_config_is_target_blind_cpu_valid(self):
        result = e.load_and_validate_config()
        self.assertEqual(result["batch_arms"], [16, 8, 4, 1])
        self.assertEqual(result["reference_arm"], 1)

    def test_raw_token_capture_is_required_and_comparable(self):
        indexes = [1, 2]; hashes = {str(v): hashlib.sha256(str(v).encode()).hexdigest() for v in indexes}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "arm.jsonl"
            rows = [self._row(v, 8) for v in indexes]
            e.atomic_jsonl(path, rows)
            self.assertEqual(e.validate_arm_raw(path, indexes, hashes, 8), rows)
            changed = [dict(row) for row in rows]; changed[1]["generated_token_ids"] = [2, 15]
            self.assertEqual(e.compare_non_target(rows, changed), [2])

    def test_child_accepts_only_the_parent_workspace(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "out"; output.mkdir(); (output / "PREFLIGHT_RECEIPT.json").write_text("{}")
            (output / "raw_arms").mkdir()
            binding = {"schema_version": 1, "protocol_id": worker.PROTOCOL_ID, "authorization_id": "a",
                       "director_response_sha256": "a" * 64, "execution_authorized": True, "source_commit": "commit",
                       "worker_sha256": "worker", "config_path": "config.json", "config_sha256": "config",
                       "output_root": str(output).replace("\\", "/"), "nonce": "nonce",
                       "hard_runtime_cap_seconds": worker.CAP_SECONDS, "jobs": 1, "retry": False}
            def fake_hash(path):
                return "worker" if Path(path).name.startswith("run_e04") else "config"
            with mock.patch.object(worker, "read_json", return_value=binding), \
                 mock.patch.object(worker, "sha_path", side_effect=fake_hash), \
                 mock.patch.object(worker.subprocess, "check_output", return_value="commit\n"):
                self.assertEqual(worker.load_binding(Path("binding.json"), output, allow_parent_workspace=True), binding)
                with self.assertRaisesRegex(worker.LocalizationFailure, "FRESH_OUTPUT"):
                    worker.load_binding(Path("binding.json"), output)
            (output / "unexpected").write_text("x")
            with mock.patch.object(worker, "read_json", return_value=binding), \
                 mock.patch.object(worker, "sha_path", side_effect=fake_hash), \
                 mock.patch.object(worker.subprocess, "check_output", return_value="commit\n"):
                with self.assertRaisesRegex(worker.LocalizationFailure, "FRESH_OUTPUT"):
                    worker.load_binding(Path("binding.json"), output, allow_parent_workspace=True)

    def test_launch_receipt_contains_a_governor_poll_contract(self):
        binding = {"output_root": "/workspace/out", "source_commit": "a" * 40, "nonce": "n"}
        rendered = launch.remote_script(binding, "b" * 40, "binding.json", "c" * 64, "pod@example")
        self.assertIn('"ssh_target":"%s"', rendered)
        self.assertIn('"expected_terminal_receipt":"%s/TERMINAL_RECEIPT.json"', rendered)
        self.assertIn('"primary_process":{"host":"RUNPOD"', rendered)
        self.assertIn('"$ssh_target"', rendered)

    def test_retokens_or_wrong_arm_fail_closed(self):
        indexes = [1]; hashes = {"1": hashlib.sha256(b"1").hexdigest()}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "arm.jsonl"; row = self._row(1, 4); del row["generated_token_ids"]
            e.atomic_jsonl(path, [row])
            with self.assertRaisesRegex(e.LocalizationFailure, "ARM_SCHEMA"):
                e.validate_arm_raw(path, indexes, hashes, 4)


if __name__ == "__main__":
    unittest.main()
