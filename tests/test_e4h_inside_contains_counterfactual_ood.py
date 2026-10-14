from __future__ import annotations

import json
import unittest

from scripts.e4h_inside_contains_counterfactual_ood import (
    E4G,
    OUT,
    QUERY_COLOR_PAIRS,
    audit_e4g_shortcuts,
    build,
    build_panel,
    oracle_target,
    semantic_roles,
    validate_panel,
)


class E4HCounterfactualOODTests(unittest.TestCase):
    def test_original_e4g_shortcuts_are_exactly_quantified(self):
        audit = audit_e4g_shortcuts()
        self.assertEqual(audit["training_primary_tasks"], 48)
        self.assertEqual(audit["training_demo1_matches_query_operation"], 48)
        self.assertEqual(audit["ood_demo1_query_match"], {"A": 48, "B": 0})
        self.assertEqual(audit["ood_color_pairs"], {"1,2": 48, "4,6": 48})
        self.assertEqual(audit["ood_frame_color_lt_inner_color"], 96)
        self.assertEqual(audit["marker_plus_numeric_order_shortcut_exact"], 96)

    def test_counterfactual_panel_is_balanced_and_target_blind(self):
        records, targets = build_panel()
        audit = validate_panel(records, targets)
        self.assertEqual(audit["canonical_units"], 48)
        self.assertEqual(audit["numeric_order"], {"FRAME_LT_INNER": 24, "FRAME_GT_INNER": 24})
        self.assertEqual(audit["marker_plus_numeric_order_shortcut_exact"], 48)
        self.assertEqual(set(audit["query_color_pairs"]), {f"{a},{b}" for a, b in QUERY_COLOR_PAIRS})
        self.assertTrue(all("output" not in row["input_task"]["test"][0] for row in records))

    def test_semantic_oracle_requires_containment_and_equalizes_simple_counts(self):
        records, targets = build_panel()
        for row in records:
            grid = row["input_task"]["test"][0]["input"]
            roles = semantic_roles(grid)
            self.assertEqual(oracle_target(grid), targets[row["record_id"]])
            frame_count = sum(value == roles["frame"] for line in grid for value in line)
            inner_count = sum(value == roles["inner"] for line in grid for value in line)
            self.assertEqual(frame_count, inner_count)

    def test_fixed_demos_are_query_independent_and_no_exact_query_duplicate_exists(self):
        records, _ = build_panel()
        demo_hashes = {
            json.dumps(row["input_task"]["train"], sort_keys=True, separators=(",", ":"))
            for row in records
        }
        self.assertEqual(len(demo_hashes), 2)  # the same two demos, in A/B order
        for row in records:
            query = row["input_task"]["test"][0]["input"]
            self.assertTrue(all(demo["input"] != query for demo in row["input_task"]["train"]))

    def test_build_is_deterministic_and_writes_no_target_sidecar(self):
        first = build()
        second = build()
        self.assertEqual(first, second)
        self.assertFalse((OUT / "COUNTERFACTUAL_OOD_TARGET_SIDECAR.json").exists())
        panel = json.loads((OUT / "COUNTERFACTUAL_OOD_INPUTS_AND_COMMITMENTS.json").read_text())
        self.assertEqual(panel["status"], "FROZEN_TARGET_BLIND_CPU_VALIDATED")
        protocol = json.loads((OUT / "PROTOCOL.json").read_text())
        self.assertEqual(protocol["training"], "FORBIDDEN_INITIAL_DIAGNOSTIC")
        self.assertEqual(protocol["requested_batch_size"], 32)
        self.assertTrue((E4G / "E4G_VALID_REJECT_CLOSEOUT.json").is_file())


if __name__ == "__main__":
    unittest.main()
