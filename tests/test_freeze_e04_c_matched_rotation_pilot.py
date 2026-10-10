import json
import tempfile
import unittest
from pathlib import Path

from scripts import freeze_e04_c_matched_rotation_pilot as pilot


class FreezeE04CPilotTests(unittest.TestCase):
    def test_freeze_is_train_only_and_matched(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "cohort"
            manifest = pilot.freeze(out)
            self.assertEqual(manifest["pairs"], 1152)
            self.assertTrue(manifest["training_split_only"])
            self.assertEqual(manifest["validation_rows_read"], 0)
            treatment = [json.loads(x) for x in (out / "TREATMENT_FIXED_TURN_ROTATION_TRAIN.jsonl").read_text().splitlines()]
            control = [json.loads(x) for x in (out / "CONTROL_NO_TRANSFORM_TRAIN.jsonl").read_text().splitlines()]
            self.assertEqual([x["pair_id"] for x in treatment], [x["pair_id"] for x in control])
            for treated, baseline in zip(treatment, control):
                self.assertEqual(treated["treatment"]["test"][0]["input"], baseline["control"]["test"][0]["input"])
                self.assertNotEqual(treated["treatment"]["test"][0]["output"], baseline["control"]["test"][0]["output"])

    def test_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "cohort"
            pilot.freeze(out)
            with self.assertRaises(pilot.FreezeFailure):
                pilot.freeze(out)

