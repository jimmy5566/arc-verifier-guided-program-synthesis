from __future__ import annotations

import unittest
from scripts import paired_v7_family_balanced_rank_margin_contract as module

CONTRACT = module.TokenGridContract(tuple(range(10)), 10, 15, 13)


class RankMarginContractTests(unittest.TestCase):
    def test_target_layout_is_disjoint_exhaustive_and_round_trips(self):
        layout = module.serialize_native_grid_target([[1, 2], [3, 4]], CONTRACT)
        self.assertEqual(layout.token_ids, (1, 2, 10, 3, 4, 15))
        self.assertEqual(layout.grid_content_positions, (0, 1, 3, 4))
        self.assertEqual(layout.row_separator_positions, (2,))
        self.assertEqual(layout.eos_positions, (5,))
        module.require_layout(layout, CONTRACT)

    def test_layout_rejects_ragged_or_invalid_grid(self):
        with self.assertRaisesRegex(RuntimeError, "RAGGED"):
            module.serialize_native_grid_target([[1], [2, 3]], CONTRACT)
        with self.assertRaisesRegex(RuntimeError, "DIGIT"):
            module.serialize_native_grid_target([[10]], CONTRACT)

    def test_fixed_batch1_subset_is_target_blind_deterministic(self):
        rows = [{"episode_id": f"e{i:02d}", "prompt_sha256": f"{59-i:064x}"} for i in range(60)]
        subset = module.fixed_batch1_subset(rows)
        self.assertEqual(len(subset), 12)
        self.assertEqual(subset, module.fixed_batch1_subset(list(reversed(rows))))

    def test_sensitivity_gate_rejects_rank_or_margin_change(self):
        base = {"e": {"grid_content_correct_top1_rate": 1.0, "grid_content_correct_top2_rate": 1.0, "grid_content_mean_margin": 1.0, "first_error_rank": None, "first_error_margin": None}}
        self.assertEqual(module.sensitivity_gate(base, base)["status"], "PASS")
        changed = {"e": {**base["e"], "grid_content_mean_margin": 0.5}}
        self.assertEqual(module.sensitivity_gate(base, changed)["status"], "FAIL_MATERIAL_RANK_OR_METRIC_DRIFT")


if __name__ == "__main__":
    unittest.main()
