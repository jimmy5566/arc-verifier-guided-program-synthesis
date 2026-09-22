from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from run_experiment import ACTIONS, CATEGORY_FAULTS, COSTS, EFFECTS, ID_CATEGORY_P, OOD_CATEGORY_P, action_sequences, episodes, run_policy, select_expected, sequence_terms


def test_fault_distributions_are_valid_and_include_single_and_dual_faults():
    assert np.allclose(ID_CATEGORY_P.sum(axis=1), 1.0)
    assert np.allclose(OOD_CATEGORY_P.sum(axis=1), 1.0)
    assert set(CATEGORY_FAULTS.sum(axis=1)) == {0, 1, 2}


def test_episode_telemetry_hides_fault_matrix_and_is_reproducible():
    first = episodes(100, 91); second = episodes(100, 91)
    for left, right in zip(first, second): assert np.array_equal(left, right)
    x, state, faults, category = first
    assert x.shape == (100, 14)
    assert faults.shape == (100, 4)
    assert not np.array_equal(x[:, :4], faults)


def test_budget_two_can_combine_partial_repairs_and_expected_selector_is_deterministic():
    repair, retention, cost = sequence_terms((1, 2), EFFECTS["BASE"], COSTS["MEDIUM"])
    assert repair[0] > EFFECTS["BASE"][1, 0]
    assert repair[1] > EFFECTS["BASE"][2, 1]
    probabilities = np.array([[.8,.7,.05,.05]])
    choice, sequences = select_expected(probabilities, 2, EFFECTS["BASE"], COSTS["MEDIUM"])
    assert len(action_sequences(2)) == 11
    assert sequences[choice[0]] in action_sequences(2)


def test_single_sequence_intervention_is_not_misclassified_as_no_action():
    faults = np.array([[1, 0, 0, 0]])
    x = np.zeros((1, 14)); probabilities = np.zeros((1, 4)); draw = np.zeros((1, 11))
    row, _, _ = run_policy("ALWAYS_STRONG_VERIFY", x, faults, probabilities, probabilities, np.zeros(1), probabilities, 1, EFFECTS["BASE"], COSTS["MEDIUM"], draw)
    assert row["wrong_intervention_rate"] == 1.0
