import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/capability_repair_baseline_v1/e05_v3_semantically_identifiable_demonstrations_and_adapter_storage_recovery"


class TestE05ResidualPrefreeze(unittest.TestCase):
    def test_prefreeze_is_exhaustive_cpu_only_and_unexecuted(self):
        subprocess.run([sys.executable, str(ROOT / "scripts/freeze_e05_v3_nonoracle_structural_residual_decomposition_prefreeze.py")], check=True)
        spec = json.loads((OUT / "E05_V3_NONORACLE_STRUCTURAL_RESIDUAL_DECOMPOSITION_PREFREEZE_SPEC_V1.json").read_text())
        expected = {
            "EXACT_INPUT_COPY",
            "TARGET_FOREGROUND_GEOMETRY_COLOR_ONLY_ERROR",
            "TARGET_CONGRUENT_GEOMETRY_WRONG_PLACEMENT",
            "TARGET_CELL_COUNT_PRESERVED_NONCONGRUENT_GEOMETRY",
            "DIMENSION_OR_CELL_COUNT_FAILURE",
        }
        self.assertEqual(expected, set(spec["taxonomy_order"]))
        self.assertEqual(expected, set(spec["taxonomy_precedence"]))
        self.assertFalse(spec["execution_authorization"])
        self.assertTrue(spec["preexecution_review_required"])
        self.assertEqual(10000, spec["bootstrap"]["replicates"])
        self.assertIn("NO_GPU_WORK", spec["prohibitions"])
        self.assertIn("NO_MODEL_LOADING", spec["prohibitions"])


if __name__ == "__main__":
    unittest.main()
