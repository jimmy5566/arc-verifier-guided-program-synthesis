from __future__ import annotations
import unittest

from scripts.audit_e4f_component_affinity_construct_validity import audit


class E4FConstructValidityTests(unittest.TestCase):
    def test_frozen_all_slot_foreground_pair_domain_has_no_count_one_negative_pair(self):
        value = audit()
        self.assertEqual(value["status"], "FAIL_CLOSED_E4F_PAIR_CONSTRUCT_NOT_IDENTIFIABLE_UNDER_FROZEN_CONSTRAINTS")
        self.assertEqual(value["strata"]["1"]["slots"], 16)
        self.assertEqual(value["strata"]["1"]["different_pairs_total"], 0)
        self.assertGreater(value["strata"]["1"]["same_pairs_total"], 0)


if __name__ == "__main__":
    unittest.main()
