from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from run_sensitivity import THRESHOLD, evaluate_policy


def test_always_and_never_strong_have_fixed_call_rates():
    source = np.array([0, 2, 3, 4])
    state = np.zeros(4, dtype=int)
    prediction = np.full((4, 5), .2)
    draw = np.array([.1, .1, .1, .1])
    always = evaluate_policy(source, state, prediction, "ALWAYS_STRONG", .5, .2, draw)
    never = evaluate_policy(source, state, prediction, "NEVER_STRONG", .5, .2, draw)
    assert always["strong_call_rate"] == 1.0
    assert never["strong_call_rate"] == 0.0
    assert always["total_verification_cost"] == .2
    assert never["wrong_commit_rate"] == .25


def test_predicted_policy_is_deterministic_and_telemetry_probabilistic():
    source = np.array([0, 2, 1, 3, 4])
    state = np.array([0, 0, 1, 1, 1])
    prediction = np.array([
        [.80, .05, .05, .05, .05], [.10, .10, .60, .10, .10],
        [.10, .60, .10, .10, .10], [.10, .10, .30, .40, .10],
        [.10, .10, .20, .10, .50],
    ])
    draw = np.full(5, .1)
    first = evaluate_policy(source, state, prediction, "PREDICTED_ATTRIBUTION", .7, .05, draw)
    second = evaluate_policy(source, state, prediction, "PREDICTED_ATTRIBUTION", .7, .05, draw)
    assert first == second
    assert first["threshold"] == THRESHOLD
    assert 0 <= first["cost_adjusted_utility"] <= 1
