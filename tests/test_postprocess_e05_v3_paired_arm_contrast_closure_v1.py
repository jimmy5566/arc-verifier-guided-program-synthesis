import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("e05_closure", ROOT / "scripts" / "postprocess_e05_v3_paired_arm_contrast_closure_v1.py")
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


class TestPairedClosure(unittest.TestCase):
    def test_paired_report_retains_all_bases_and_transitions(self):
        index = {}
        for arm in (MOD.TREATMENT, "REF"):
            for base in [f"b{number:02d}" for number in range(48)]:
                for turn in range(4):
                    index[(arm, "C", base, turn, f"{base}-{turn}")] = "EXACT_INPUT_COPY" if arm == "REF" else ("DIMENSION_OR_CELL_COUNT_FAILURE" if base == "b00" else "EXACT_INPUT_COPY")
        with mock.patch.object(MOD, "TREATMENT", MOD.TREATMENT):
            report = MOD.paired_report(index, "REF", "C")
        self.assertEqual(48, report["canonical_base_count"])
        self.assertEqual(192, report["transition_total_check"])
        category = report["outcomes"]["DIMENSION_OR_CELL_COUNT_FAILURE"]
        self.assertEqual(1 / 48, category["mean_paired_delta"])
        self.assertEqual(["b00"], category["changed_canonical_base_ids"])


if __name__ == "__main__":
    unittest.main()
