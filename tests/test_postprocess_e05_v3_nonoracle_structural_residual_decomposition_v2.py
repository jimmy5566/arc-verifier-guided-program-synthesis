import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("e05_post", ROOT / "scripts" / "postprocess_e05_v3_nonoracle_structural_residual_decomposition_v2.py")
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


class TestE05PostprocessV2(unittest.TestCase):
    def test_mixed_condition_dominance_is_inconclusive(self):
        with tempfile.TemporaryDirectory() as raw:
            out = Path(raw)
            result = {"status": "COMPLETE_VALID_DESCRIPTIVE_ONLY", "prohibitions_observed": {}, "aggregate": {
                "E04E_RUN003_TREATMENT_EQUAL_SLOT|MARKER_BINDING_CONTROL": {"bootstrap": {"status": "VALID", "category_ci95": {"A": [.9, 1], "B": [0, .1]}}},
                "E04E_RUN003_TREATMENT_EQUAL_SLOT|MARKER_CONDITIONED_ROTATION": {"bootstrap": {"status": "VALID", "category_ci95": {"A": [0, .1], "B": [.9, 1]}}},
            }}
            for name, value in (("RESULT_V2.json", result), ("EXECUTION_RECEIPT_V2.json", {}), ("PER_BASE_VECTORS_V2.json", {})):
                (out / name).write_text(json.dumps(value))
            with mock.patch.object(MOD, "OUT", out), mock.patch.object(MOD, "RESULT", out / "RESULT_V2.json"), mock.patch.object(MOD, "RECEIPT", out / "EXECUTION_RECEIPT_V2.json"), mock.patch.object(MOD, "VECTORS", out / "PER_BASE_VECTORS_V2.json"), mock.patch.object(MOD, "CONCLUSION", out / "RESULT_CONCLUSION_V2.json"), mock.patch.object(MOD, "BRIEF", out / "RESULT_REVIEW_BRIEF_V2B.json"):
                MOD.main()
            self.assertEqual("INCONCLUSIVE_MIXED_REQUIRED_CONDITIONS", json.loads((out / "RESULT_CONCLUSION_V2.json").read_text())["primary_prospective_decision"])


if __name__ == "__main__":
    unittest.main()
