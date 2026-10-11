import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/capability_repair_baseline_v1/e05_v3_semantically_identifiable_demonstrations_and_adapter_storage_recovery"


def conditional_share(rows, category):
    """Frozen primary calculation: pooled canonical-base numerator/denominator."""
    return sum(row["category"].get(category, 0) for row in rows) / sum(row["eligible"] for row in rows)


def unconditional_prevalence(rows, category):
    return sum(row["category"].get(category, 0) / 4 for row in rows) / len(rows)


class TestE05ResidualPrefreezeV2(unittest.TestCase):
    def test_v2_freeze_binds_director_repair_without_execution(self):
        subprocess.run([sys.executable, str(ROOT / "scripts/freeze_e05_v3_nonoracle_structural_residual_decomposition_prefreeze_v2.py")], check=True)
        spec = json.loads((OUT / "E05_V3_NONORACLE_STRUCTURAL_RESIDUAL_DECOMPOSITION_PREFREEZE_SPEC_V2.json").read_text())
        self.assertFalse(spec["execution_authorization"])
        self.assertTrue(spec["preexecution_review_required"])
        self.assertEqual(10000, spec["bootstrap"]["replicates"])
        self.assertIn("sum n_category(base) / sum n_eligible(base)", spec["bootstrap"]["resampling"])
        self.assertIn("unconditional", spec["population"]["unconditional_diagnostics"])
        self.assertIn("NO_GPU_WORK", spec["prohibitions"])

    def test_mixed_view_fixture_uses_conditional_primary_not_unconditional_prevalence(self):
        # Base A: four eligible, three in category. Base B: one eligible, one in category;
        # other views are parser-invalid/exact-target and remain retained in the base.
        rows = [
            {"eligible": 4, "category": {"EXACT_INPUT_COPY": 3}},
            {"eligible": 1, "category": {"EXACT_INPUT_COPY": 1}},
        ]
        self.assertEqual(0.8, conditional_share(rows, "EXACT_INPUT_COPY"))
        self.assertEqual(0.5, unconditional_prevalence(rows, "EXACT_INPUT_COPY"))
        self.assertNotEqual(conditional_share(rows, "EXACT_INPUT_COPY"), unconditional_prevalence(rows, "EXACT_INPUT_COPY"))


if __name__ == "__main__":
    unittest.main()
