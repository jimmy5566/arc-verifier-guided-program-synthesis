from __future__ import annotations

import unittest

from scripts.postprocess_paired_v7_family_balanced_rank_margin_v1 import postprocess


class PostprocessTests(unittest.TestCase):
    def row(self, condition: str, mode: str, family: str, episode: int, offset: float) -> dict:
        return {
            "checkpoint_condition": condition,
            "mode_id": mode,
            "episode_id": f"{family}-{episode}",
            "family": family,
            "components": {"GRID_CONTENT": {"token_count": 2, "correct_top1_count": 1 if offset == 0 else (2 if offset > 0 else 0), "correct_top2_count": 2, "margin_sum": 2.0 + offset}, "ROW_SEPARATOR": {"token_count": 1, "correct_top1_count": 1, "correct_top2_count": 1, "margin_sum": 1.0}, "EOS_END": {"token_count": 1, "correct_top1_count": 1, "correct_top2_count": 1, "margin_sum": 1.0}},
            "first_free_running_error": None,
        }

    def test_complete_synthetic_result_is_target_blind_and_bootstrapped(self):
        rows = []
        for family in range(5):
            for episode in range(12):
                candidate_offset = -0.5 if family == 0 else 0.5
                for condition, offset in (("RECONSTRUCTED_FOUNDATION_V2_V7", 0.0), ("FAMILY_BALANCED", candidate_offset)):
                    rows.append(self.row(condition, "PRIMARY_B32", str(family), episode, offset))
                if episode == 0 or episode == 11:
                    for condition, offset in (("RECONSTRUCTED_FOUNDATION_V2_V7", 0.0), ("FAMILY_BALANCED", candidate_offset)):
                        rows.append(self.row(condition, "SENSITIVITY_B1", str(family), episode, offset))
        # Add deterministic ten remaining B1 episodes: two per family.
        for family in range(5):
            for episode in (1,):
                candidate_offset = -0.5 if family == 0 else 0.5
                for condition, offset in (("RECONSTRUCTED_FOUNDATION_V2_V7", 0.0), ("FAMILY_BALANCED", candidate_offset)):
                    rows.append(self.row(condition, "SENSITIVITY_B1", str(family), episode, offset))
        # The sensitivity cohort must be exactly twelve episodes, hence use
        # first twelve IDs from the primary cohort rather than changing data.
        b1 = [row for row in rows if row["mode_id"] == "SENSITIVITY_B1"][:24]
        primary = [row for row in rows if row["mode_id"] == "PRIMARY_B32"]
        result = postprocess(primary + b1, protected_families={"0"})
        self.assertEqual(result["raw_rows"], 144)
        self.assertEqual(result["batch1_discrete_gate"]["status"], "PASS")
        self.assertFalse(result["target_token_ids_persisted"])
        self.assertEqual(result["classification"], "FAMILY_SPECIFIC_RANKING_HARM")
        self.assertEqual(result["family_specific_ranking_harm"], ["0"])
        self.assertEqual(set(result["component_results"]), {"GRID_CONTENT", "ROW_SEPARATOR", "EOS_END"})
        self.assertIn("first_error_summary", result)


if __name__ == "__main__":
    unittest.main()
