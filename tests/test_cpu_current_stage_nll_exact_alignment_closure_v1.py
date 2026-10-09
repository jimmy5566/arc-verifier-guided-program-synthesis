import copy
import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "closure",
    ROOT / "scripts/analyze_cpu_current_stage_nll_exact_alignment_closure_v1.py",
)
closure = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(closure)


class AlignmentUnitTests(unittest.TestCase):
    def test_exact_transition_categories(self):
        self.assertEqual(closure.exact_transition(False, True), "FIX")
        self.assertEqual(closure.exact_transition(True, False), "HARM")
        self.assertEqual(closure.exact_transition(True, True), "STABLE_CORRECT")
        self.assertEqual(closure.exact_transition(False, False), "STABLE_WRONG")

    def test_delta_summary_preserves_all_categories(self):
        rows = []
        cases = [
            (False, True, -1.0),
            (True, False, -0.5),
            (True, True, 0.25),
            (False, False, 0.0),
        ]
        for baseline, candidate, delta in cases:
            rows.append(
                {
                    "supervised_token_count": 2,
                    "v7_per_token_nll": 2.0,
                    "family_balanced_per_token_nll": 2.0 + delta,
                    "v7_exact_grid_match": baseline,
                    "family_balanced_exact_grid_match": candidate,
                    "exact_grid_transition": closure.exact_transition(baseline, candidate),
                    "per_token_nll_delta_candidate_minus_v7": delta,
                }
            )
        summary = closure.delta_summary(rows)
        self.assertEqual(summary["episodes"], 4)
        self.assertEqual(summary["exact_grid_transition_counts"], {
            "FIX": 1,
            "HARM": 1,
            "STABLE_CORRECT": 1,
            "STABLE_WRONG": 1,
        })
        self.assertEqual(summary["per_token_nll_delta_candidate_minus_v7"]["improved_negative_count"], 2)
        self.assertAlmostEqual(summary["pooled_token_weighted_nll"]["delta_candidate_minus_v7"], -0.3125)

    def test_atomic_identity_mismatch_fails_closed(self):
        gate = closure.read_json(closure.PATHS["atomic_gate"])
        checkpoint = closure.read_json(closure.PATHS["atomic_checkpoint_manifest"])
        train_package = closure.read_json(closure.PATHS["train_execution_package"])
        train_result = closure.read_json(closure.PATHS["train_result_package"])
        bad_gate = copy.deepcopy(gate)
        bad_gate["candidate_result"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "ATOMIC_RESULT_SHA_MISMATCH"):
            closure.build_atomic_posthoc(bad_gate, checkpoint, train_package, train_result)


if __name__ == "__main__":
    unittest.main()
