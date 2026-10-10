import hashlib
import tempfile
import unittest
from pathlib import Path

from scripts import e04_v3_fixed64_batch_rung_localization as e


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

    def test_retokens_or_wrong_arm_fail_closed(self):
        indexes = [1]; hashes = {"1": hashlib.sha256(b"1").hexdigest()}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "arm.jsonl"; row = self._row(1, 4); del row["generated_token_ids"]
            e.atomic_jsonl(path, [row])
            with self.assertRaisesRegex(e.LocalizationFailure, "ARM_SCHEMA"):
                e.validate_arm_raw(path, indexes, hashes, 4)


if __name__ == "__main__":
    unittest.main()
