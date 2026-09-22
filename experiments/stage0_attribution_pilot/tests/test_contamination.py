from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from metrics import contamination_simulation, thresholds


ROWS = [
    {"any_of_k_correct": True, "top1_correct": True},
    {"any_of_k_correct": True, "top1_correct": False},
    {"any_of_k_correct": False, "top1_correct": False},
]


def test_contamination_is_reproducible_and_one_sided():
    first = contamination_simulation(ROWS, [0.0, 0.5], repetitions=100, seed=7)
    second = contamination_simulation(ROWS, [0.0, 0.5], repetitions=100, seed=7)
    assert first == second
    rows = {(row["rho"], row["method"]): row for row in first}
    assert rows[(0.5, "NAIVE")]["estimate_mean"] < rows[(0.0, "NAIVE")]["estimate_mean"]
    assert rows[(0.5, "ORACLE_SKIP")]["estimate_mean"] == rows[(0.0, "ORACLE_SKIP")]["estimate_mean"]
    assert rows[(0.5, "NAIVE")]["naive_minus_oracle_skip_upper_95"] < 0


def test_threshold_output_is_defined():
    results = contamination_simulation(ROWS, [0.0, 0.5], repetitions=100, seed=2)
    found = thresholds(results)
    assert {entry["threshold"] for entry in found} == {
        "statistical_naive_vs_oracle_skip_95pct_excludes_zero",
        "practical_naive_bias_exceeds_5pp",
    }
