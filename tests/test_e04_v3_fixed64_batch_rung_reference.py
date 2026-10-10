import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.inspect_e04_v3_fixed64_batch_rung_reference import inspect_reference


class Fixed64ReferenceAuditTest(unittest.TestCase):
    def _legacy(self, index: int) -> dict:
        return {"row_index": index, "input_sha256": "x", "text": "1", "parser_valid": True,
                "parsed_grid": [[1]], "prompt_tokens": 1, "completion_tokens": 1,
                "elapsed_seconds": 0.0, "effective_batch_size": 1}

    def test_legacy_b1_has_no_raw_token_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "b1.jsonl"
            path.write_text(json.dumps(self._legacy(3)) + "\n", encoding="utf-8")
            result = inspect_reference(path, hashlib.sha256(path.read_bytes()).hexdigest(), [3])
            self.assertEqual(result["status"], "RAW_TOKEN_REFERENCE_UNAVAILABLE")
            self.assertEqual(result["raw_token_rows"], 0)
            self.assertFalse(result["target_sidecar_accessed"])

    def test_sha_and_row_coverage_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "b1.jsonl"
            path.write_text(json.dumps(self._legacy(3)) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                inspect_reference(path, "0" * 64, [3])
            with self.assertRaisesRegex(ValueError, "COVERAGE"):
                inspect_reference(path, hashlib.sha256(path.read_bytes()).hexdigest(), [4])


if __name__ == "__main__":
    unittest.main()
