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
        rows = [
            {"episode_id": f"e{i:02d}", "checkpoint_condition": condition, "prompt_sha256": f"{59-i:064x}"}
            for i in range(60)
            for condition in ("RECONSTRUCTED_FOUNDATION_V2_V7", "FAMILY_BALANCED")
        ]
        subset = module.fixed_batch1_subset(rows)
        self.assertEqual(len(subset), 12)
        self.assertEqual(subset, module.fixed_batch1_subset(list(reversed(rows))))

    def test_pairing_rejects_missing_condition_or_prompt_mismatch(self):
        rows = [
            {"episode_id": f"e{i:02d}", "checkpoint_condition": condition, "prompt_sha256": f"{i:064x}"}
            for i in range(60)
            for condition in ("RECONSTRUCTED_FOUNDATION_V2_V7", "FAMILY_BALANCED")
        ]
        self.assertEqual(len(module.require_two_checkpoint_prompt_pairing(rows)), 60)
        mismatched = [dict(row) for row in rows]
        mismatched[1]["prompt_sha256"] = "f" * 64
        with self.assertRaisesRegex(RuntimeError, "PROMPT_HASH_MISMATCH"):
            module.require_two_checkpoint_prompt_pairing(mismatched)
        missing = rows[:-1] + [dict(rows[-2])]
        with self.assertRaisesRegex(RuntimeError, "CONDITION_PAIR_INVALID"):
            module.require_two_checkpoint_prompt_pairing(missing)

    def test_sensitivity_gate_rejects_discrete_but_not_margin_change(self):
        base = {"e": {"grid_content_correct_top1_rate": 1.0, "grid_content_correct_top2_rate": 1.0, "grid_content_mean_margin": 1.0, "first_error_rank": None, "first_error_parser_status": "VALID"}}
        self.assertEqual(module.sensitivity_gate(base, base)["status"], "PASS")
        changed = {"e": {**base["e"], "grid_content_mean_margin": 0.5}}
        self.assertEqual(module.sensitivity_gate(base, changed)["status"], "PASS")
        discrete_changed = {"e": {**base["e"], "grid_content_correct_top1_rate": 0.0}}
        self.assertEqual(module.sensitivity_gate(base, discrete_changed)["status"], "FAIL_MATERIAL_RANK_OR_METRIC_DRIFT")

    def test_margin_u_and_interval_are_not_exact_float_equality(self):
        b32 = {"e": {"v7_grid_margin": 1.0, "family_balanced_grid_margin": 1.5}}
        b1 = {"e": {"v7_grid_margin": 1.01, "family_balanced_grid_margin": 1.49}}
        self.assertAlmostEqual(module.paired_margin_sensitivity_u(b32, b1), 0.02)
        self.assertEqual(module.adjusted_interval((0.1, 0.3), 0.02), (0.08, 0.32))

    def test_family_bootstrap_is_seed_deterministic(self):
        rows = [{"family": f"f{family}", "family_balanced_minus_v7": episode - 6} for family in range(5) for episode in range(12)]
        first = module.family_stratified_bootstrap_deltas(rows, replicates=20)
        self.assertEqual(first, module.family_stratified_bootstrap_deltas(rows, replicates=20))


if __name__ == "__main__":
    unittest.main()
