import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("e06_matrix", ROOT / "scripts" / "freeze_e01_e05_mechanism_evidence_synthesis_for_e06_selection_v1.py")
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


class TestE06EvidenceMatrix(unittest.TestCase):
    def test_matrix_is_selection_only_and_names_required_discriminators(self):
        rows = MOD.matrix()
        self.assertEqual(6, len(rows))
        self.assertTrue(all(row["unresolved_discriminator"] for row in rows))
        self.assertIn("SEQUENCE_OBJECTIVE_OR_RANK_MISMATCH", {row["mechanism_id"] for row in rows})
        self.assertIn("COMPOSITIONAL_TRANSFER_LIMITATION", {row["mechanism_id"] for row in rows})

    def test_frozen_output_preserves_e05_and_all_matrix_requirements(self):
        out = ROOT / "experiments/capability_repair_baseline_v1/e06_selection_evidence_synthesis_v1"
        report = json.loads((out / "E01_E05_MECHANISM_EVIDENCE_MATRIX_V1.json").read_text())
        self.assertEqual("INCONCLUSIVE_MIXED_REQUIRED_CONDITIONS", report["e05_prospective_conclusion_preserved"])
        self.assertFalse(report["prohibitions_observed"]["gpu_used"])
        self.assertFalse(report["prohibitions_observed"]["training"])
        self.assertEqual(6, len(report["matrix"]))


if __name__ == "__main__":
    unittest.main()
