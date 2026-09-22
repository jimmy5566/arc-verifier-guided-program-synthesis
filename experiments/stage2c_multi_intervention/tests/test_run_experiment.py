from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from run_experiment import ACTIONS, COSTS, EFFECTS, OOD_SOURCE_P, OOD_STATE_P, expected_actions, run_policy, uncertainty_actions
from simulate import FEATURES


def test_expected_utility_selects_matching_high_effect_action():
    probabilities = np.eye(5)
    actions = expected_actions(probabilities, EFFECTS["BASE"], COSTS["BASE"])
    assert tuple(actions) == tuple(range(len(ACTIONS)))


def test_commit_and_oracle_policy_metrics_are_deterministic():
    source = np.array([0, 1, 2, 3, 4])
    x = np.zeros((5, 14))
    x[:, 5:8] = [.5, .5, .5]
    probability = np.eye(5)[source]
    draw = np.full((5, len(ACTIONS)), .01)
    common = (x, source, probability, probability, np.full(5, .5), probability, probability, EFFECTS["BASE"], COSTS["BASE"], draw)
    commit, _ = run_policy("COMMIT_ONLY", *common)
    oracle, actions = run_policy("ORACLE_ATTRIBUTION", *common)
    assert commit["action_COMMIT_rate"] == 1.0
    assert oracle["wrong_intervention_rate"] == 0.0
    assert tuple(actions) == tuple(range(len(ACTIONS)))


def test_uncertainty_router_uses_only_documented_observable_columns():
    x = np.zeros((2, 14))
    x[0, 5:8] = [.95, .05, .05]
    x[1, 5:8] = [.05, 1.10, .95]
    assert tuple(uncertainty_actions(x)) == (0, 2)


def test_hidden_source_is_not_an_observable_telemetry_feature_and_ood_is_valid():
    assert "source" not in FEATURES
    assert np.isclose(OOD_STATE_P.sum(), 1.0)
    assert np.allclose(OOD_SOURCE_P.sum(axis=1), 1.0)
