from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("measurement_contract", ROOT / "scripts" / "unified_native_baseline_measurement_contract.py")
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(mod)


class MeasurementContractTests(unittest.TestCase):
    def rows(self):
        return [{"episode_id": f"e{index:02d}", "prompt_tokens": 100 - index} for index in range(20)]

    def test_length_bucketing_and_ladder_are_frozen(self):
        batches = mod.length_bucketed_batches(self.rows())
        self.assertEqual([len(batch) for batch in batches], [20])
        self.assertEqual([row["episode_id"] for row in batches[0]][0], "e19")
        self.assertEqual(mod.next_fallback(32), 16)
        self.assertEqual(mod.next_fallback(1), None)

    def test_fixed_batch1_subset_is_deterministic(self):
        first = [row["episode_id"] for row in mod.fixed_validation_subset(self.rows())]
        second = [row["episode_id"] for row in mod.fixed_validation_subset(reversed(self.rows()))]
        self.assertEqual(first, second); self.assertEqual(len(first), 12)

    def test_material_token_or_metric_drift_fails_closed(self):
        common = {"episode_id":"e", "generated_token_ids":[1], "termination_status":"EOS", "eos_observed":True, "canonical_prediction_sha256":"x", "parse_valid":True}
        self.assertEqual(mod.validation_gate([common], [dict(common)])["status"], "PASS")
        changed = dict(common); changed["generated_token_ids"] = [2]
        self.assertEqual(mod.validation_gate([common], [changed])["status"], "FAIL_MATERIAL_DRIFT")

    def test_eos_padding_normalization_and_rank2_binding(self):
        padded = mod.normalize_eos_terminated_trajectory([1, 2, 15, 13, 13], eos_token_id=15, pad_token_id=13)
        serial = mod.normalize_eos_terminated_trajectory([1, 2, 15], eos_token_id=15, pad_token_id=13)
        self.assertEqual(padded["generated_token_ids"], serial["generated_token_ids"])
        self.assertEqual(padded["framework_trailing_pad_count"], 2)
        self.assertEqual(mod.normalize_eos_terminated_trajectory([1, 13], eos_token_id=15, pad_token_id=13)["generated_token_ids"], [1, 13])
        binding = mod.choose_rank2_alternate_binding([13, 9, 7, 4, 5], 3, [
            {"position":0,"rank2_token_id":8,"top1_top2_margin":0.4},
            {"position":1,"rank2_token_id":6,"top1_top2_margin":0.1},
        ])
        self.assertEqual(binding["primary_prefix_token_ids"], [13, 9, 7, 4])
        self.assertEqual(binding["forced_rank2_token_id"], 6)

    def test_correct_target_rank_is_not_generated_token_rank(self):
        trace = mod.align_target_evidence(generated_token_ids=[5, 7], target_token_ids=[5, 9], target_ranks=[1, 3], target_margins=[0.4, -0.2])
        self.assertEqual(trace["correct_target_token_ranks"], [1, 3])
        self.assertEqual(trace["first_free_running_error_index"], 1)
        self.assertFalse(trace["target_alignment_used_for_selection"])


if __name__ == "__main__":
    unittest.main()
