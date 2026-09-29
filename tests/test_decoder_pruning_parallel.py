import json
import unittest

from scripts.audit_decoder_pruning_parallel import (
    Policy,
    local_retained,
    policy,
    validate_summary,
)


def distribution():
    return [(1, -0.1), (2, -0.2), (3, -1.0), (4, -2.0), (5, -3.0), (6, -4.0),
            (7, -5.0), (8, -6.0), (9, -7.0), (10, -8.0), (15, -9.0), (0, -10.0)]


class DecoderPruningParallelTests(unittest.TestCase):
    def test_margin_reserve_only_retains_rank_two_for_small_gap(self):
        ranked = distribution()
        small = policy("p", "HYBRID", "v5_margin_reserve", delta=0.2)
        large = policy("p2", "HYBRID", "v5_margin_reserve", delta=0.05)
        self.assertIn(2, local_retained(small, ranked, prefix_nll=2.0, prefix_regret=0.0, regret_window=[], position=0))
        self.assertNotIn(2, local_retained(large, ranked, prefix_nll=2.0, prefix_regret=0.0, regret_window=[], position=0))

    def test_affine_rank_two_preserves_second_legal_token(self):
        ranked = distribution()
        spec = policy("p", "HYBRID", "affine_rank2", tau0=0.1, lambda_=0.0)
        kept = local_retained(spec, ranked, prefix_nll=2.0, prefix_regret=0.0, regret_window=[], position=0)
        self.assertEqual({1, 2}, kept)

    def test_summary_inventory_rejects_mismatch(self):
        grouped = {("a", 0, 12, "identity"): [{"output_id": "a:o0"}]}
        with self.assertRaises(RuntimeError):
            validate_summary([], grouped)


if __name__ == "__main__":
    unittest.main()
