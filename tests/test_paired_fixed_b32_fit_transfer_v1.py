import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("paired_fixed", ROOT / "scripts" / "run_paired_fixed_b32_fit_transfer_v1.py")
WORKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WORKER)


class PairedFixedB32Test(unittest.TestCase):
    def test_identical_signs_and_conclusion_are_stable(self):
        self.assertTrue(WORKER.conclusion_stable({"width": -0.2, "orientation": 0.1}, {"width": -0.01, "orientation": 0.02}, "H1", "H1"))

    def test_sign_or_conclusion_change_is_numerically_inconclusive(self):
        self.assertFalse(WORKER.conclusion_stable({"width": -0.2}, {"width": 0.01}, "H3", "H3"))
        self.assertFalse(WORKER.conclusion_stable({"width": -0.2}, {"width": -0.01}, "H3", "H1"))


if __name__ == "__main__":
    unittest.main()
