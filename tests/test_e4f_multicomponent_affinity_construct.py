from __future__ import annotations
import unittest
from scripts.audit_e4f_multicomponent_affinity_construct import audit

class E4FMulticomponentAffinityTests(unittest.TestCase):
    def test_equal_per_slot_pairs_are_semantically_valid_but_chebyshev_shortcuts(self):
        value = audit()
        self.assertEqual(value["status"], "FAIL_CLOSED_E4F_MULTICOMPONENT_PAIR_SHORTCUT")
        selected = value["selected_pair_set"]
        self.assertEqual((selected["treated_slots"], selected["same_component"], selected["different_component"]), (32, 32, 32))
        baseline = value["shortcut_audit"]["predeclared_baselines"][0]
        self.assertEqual(baseline["accuracy"], 1.0)
        self.assertLess(baseline["max_same"], baseline["min_different"])

if __name__ == "__main__": unittest.main()
