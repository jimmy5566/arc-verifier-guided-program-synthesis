import unittest

from scripts.audit_decoder_pruning_counterfactual import evaluate_cell


def token(position, gold, *, cumulative, strict, floor, local, distribution):
    import json
    return {
        "task_id": "task", "output_id": "task:o0", "output_index": "0", "depth": "24", "view": "identity",
        "token_position": str(position), "gold_token_id": str(gold), "cumulative_gold_nll": str(cumulative),
        "strict_gold_survives": str(strict), "v5_floor_restores_gold": str(floor),
        "v5_local_gold_survives": str(local), "scoring_path": "INCREMENTAL_KV_REPLAY",
        "legal_arc_token_logprobs_json": json.dumps(distribution),
    }


class DecoderPruningCounterfactualTests(unittest.TestCase):
    def test_topk_two_can_retain_second_ranked_gold_when_current_v5_prunes_it(self):
        rows = [token(0, 2, cumulative=2.0, strict=False, floor=False, local=False, distribution={"1": -0.1, "2": -2.0, "3": -3.0, "4": -4.0, "5": -5.0, "6": -6.0, "7": -7.0, "8": -8.0, "9": -9.0, "10": -10.0, "15": -15.0, "0": -20.0})]
        self.assertFalse(evaluate_cell(rows, "CURRENT_V5", 1.6094379124341003)["gold_path_survives"])
        self.assertFalse(evaluate_cell(rows, "TOPK_INSURANCE_1", 1.6094379124341003)["gold_path_survives"])
        self.assertTrue(evaluate_cell(rows, "TOPK_INSURANCE_2", 1.6094379124341003)["gold_path_survives"])

    def test_relative_regret_uses_cumulative_regret_not_probability_threshold(self):
        rows = [
            token(0, 2, cumulative=0.2, strict=True, floor=False, local=True, distribution={"1": -0.1, "2": -0.2, "3": -3.0, "4": -4.0, "5": -5.0, "6": -6.0, "7": -7.0, "8": -8.0, "9": -9.0, "10": -10.0, "15": -15.0, "0": -20.0}),
            token(1, 2, cumulative=0.4, strict=True, floor=False, local=True, distribution={"1": -0.1, "2": -0.2, "3": -3.0, "4": -4.0, "5": -5.0, "6": -6.0, "7": -7.0, "8": -8.0, "9": -9.0, "10": -10.0, "15": -15.0, "0": -20.0}),
        ]
        result = evaluate_cell(rows, "RELREGRET_0.5", 1.6094379124341003)
        self.assertTrue(result["gold_path_survives"])
        self.assertGreaterEqual(result["estimated_retained_branches_per_step_max"], 1)
