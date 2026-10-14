from __future__ import annotations

import unittest

from scripts.run_e4h_counterfactual_ood_v1 import paired_bootstrap, score
from scripts.e4h_inside_contains_counterfactual_ood import build_panel


class E4HWorkerTests(unittest.TestCase):
    def test_bootstrap_is_paired_over_48_canonical_units(self):
        self.assertEqual(paired_bootstrap([0] * 48), [0, 0])
        self.assertEqual(paired_bootstrap([1] * 48), [48, 48])

    def test_panel_has_exact_scoring_identity(self):
        records, targets = build_panel()
        self.assertEqual(len(records), 96)
        self.assertEqual({row["record_id"] for row in records}, set(targets))


if __name__ == "__main__":
    unittest.main()
