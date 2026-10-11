import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("e05_residual_v2", ROOT / "scripts" / "run_e05_v3_nonoracle_structural_residual_decomposition_v2.py")
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


class TestResidualDecompositionV2(unittest.TestCase):
    def test_strict_precedence_and_exhaustiveness(self):
        target = [[0, 0, 0], [0, 2, 0], [0, 0, 0]]
        self.assertEqual("EXACT_INPUT_COPY", MOD.classify([[1, 1, 1], [1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1], [1, 1, 1]], target))
        self.assertEqual("TARGET_FOREGROUND_GEOMETRY_COLOR_ONLY_ERROR", MOD.classify([[0, 0, 0], [0, 3, 0], [0, 0, 0]], [[9]], target))
        self.assertEqual("TARGET_CONGRUENT_GEOMETRY_WRONG_PLACEMENT", MOD.classify([[0, 0, 0], [0, 0, 2], [0, 0, 0]], [[9]], target))
        two_cell_target = [[0, 0, 0], [0, 2, 2], [0, 0, 0]]
        self.assertEqual("TARGET_CELL_COUNT_PRESERVED_NONCONGRUENT_GEOMETRY", MOD.classify([[0, 2, 0], [0, 0, 2], [0, 0, 0]], [[9]], two_cell_target))
        self.assertEqual("DIMENSION_OR_CELL_COUNT_FAILURE", MOD.classify([[0, 2]], [[9]], target))

    def test_primary_bootstrap_uses_pooled_conditional_denominator(self):
        vectors = [
            {"eligible_count": 4, "category_counts": {category: (3 if category == "EXACT_INPUT_COPY" else 0) for category in MOD.CATEGORIES}},
            {"eligible_count": 1, "category_counts": {category: (1 if category == "EXACT_INPUT_COPY" else 0) for category in MOD.CATEGORIES}},
        ]
        result = MOD.bootstrap(vectors, 20261013, 10)
        self.assertEqual("VALID", result["status"])
        self.assertGreater(result["category_ci95"]["EXACT_INPUT_COPY"][0], 0.5)


if __name__ == "__main__":
    unittest.main()
