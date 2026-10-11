import hashlib
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/capability_repair_baseline_v1/e05_v3_semantically_identifiable_demonstrations_and_adapter_storage_recovery/e05_v3_paired_arm_contrast_closure_v1"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TestFrozenClosureResult(unittest.TestCase):
    def test_all_reports_are_exhaustive_and_bound_to_frozen_inputs(self):
        contrasts = json.loads((OUT / "PAIRED_CONTRASTS_V1.json").read_text())
        source = ROOT / "experiments/capability_repair_baseline_v1/e05_v3_semantically_identifiable_demonstrations_and_adapter_storage_recovery/e05_v3_nonoracle_structural_residual_decomposition_v2"
        self.assertEqual(4, len(contrasts["reports"]))
        self.assertEqual(sha(source / "RESULT_V2.json"), contrasts["input_identities"]["result_v2_sha256"])
        self.assertEqual(sha(source / "PER_BASE_VECTORS_V2.json"), contrasts["input_identities"]["per_base_vectors_v2_sha256"])
        for report in contrasts["reports"]:
            self.assertEqual(48, report["canonical_base_count"])
            self.assertEqual(4, report["views_per_base"])
            self.assertEqual(192, report["total_paired_views"])
            self.assertEqual(192, report["transition_total_check"])
            self.assertEqual(49, len(report["complete_paired_transition_table"]))
            for outcome in report["outcomes"].values():
                self.assertEqual(48, len(outcome["per_base_four_view_share_delta"]))
                self.assertEqual(outcome["changed_base_count"], len(outcome["changed_canonical_base_ids"]))


if __name__ == "__main__":
    unittest.main()
