import json
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/capability_repair_baseline_v1/e05_v3_semantically_identifiable_demonstrations_and_adapter_storage_recovery/e05_v3_nonoracle_structural_residual_decomposition_v2"
CATEGORIES = {
    "EXACT_INPUT_COPY",
    "TARGET_FOREGROUND_GEOMETRY_COLOR_ONLY_ERROR",
    "TARGET_CONGRUENT_GEOMETRY_WRONG_PLACEMENT",
    "TARGET_CELL_COUNT_PRESERVED_NONCONGRUENT_GEOMETRY",
    "DIMENSION_OR_CELL_COUNT_FAILURE",
}


class TestE05ResidualExecutionV2Result(unittest.TestCase):
    def test_frozen_vectors_reconstruct_primary_conditional_estimates(self):
        result = json.loads((OUT / "RESULT_V2.json").read_text())
        vectors = json.loads((OUT / "PER_BASE_VECTORS_V2.json").read_text())["vectors"]
        self.assertEqual("COMPLETE_VALID_DESCRIPTIVE_ONLY", result["status"])
        self.assertEqual(288, len(vectors))
        for item in vectors:
            self.assertEqual(4, len(item["four_view_vector"]))
            self.assertEqual([0, 1, 2, 3], [view["turn"] for view in item["four_view_vector"]])
            self.assertTrue(set(item["category_counts"]).issubset(CATEGORIES))
            self.assertEqual(item["eligible_count"], sum(item["category_counts"].values()))
        grouped = {}
        for item in vectors:
            grouped.setdefault((item["arm"], item["condition"]), []).append(item)
        for key, bases in grouped.items():
            self.assertEqual(48, len(bases))
            aggregate = result["aggregate"]["|".join(key)]
            denominator = sum(base["eligible_count"] for base in bases)
            self.assertEqual(denominator, aggregate["eligible_view_count"])
            for category in CATEGORIES:
                expected = sum(base["category_counts"][category] for base in bases) / denominator
                self.assertAlmostEqual(expected, aggregate["primary_conditional_category_share"][category])
        self.assertFalse(result["prohibitions_observed"]["model_loaded"])
        self.assertFalse(result["prohibitions_observed"]["gpu_used"])
        self.assertFalse(result["prohibitions_observed"]["training"])


if __name__ == "__main__":
    unittest.main()
