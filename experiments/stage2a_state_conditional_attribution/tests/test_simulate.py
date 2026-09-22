from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from simulate import FEATURES, SOURCE_P, STATE_P, episodes, update_policy


def test_state_episode_generation_is_reproducible_and_has_no_source_feature():
    first = episodes(100, 71)
    second = episodes(100, 71)
    for left, right in zip(first, second):
        assert np.array_equal(left, right)
    telemetry, state, source = first
    assert telemetry.shape == (100, len(FEATURES))
    assert state.shape == source.shape == (100,)
    assert np.isclose(STATE_P.sum(), 1.0)
    assert np.allclose(SOURCE_P.sum(axis=1), 1.0)
    assert "source" not in FEATURES


def test_oracle_state_update_excludes_non_selector_failure_from_reliability():
    # G is intentionally ignored: it must not reduce selector reliability.
    source = np.array([1, 0, 2])
    state = np.array([0, 0, 0])
    prediction = np.eye(5)[source]
    rows = update_policy(source, state, prediction, "ORACLE_STATE", threshold=.0)
    state_a = next(row for row in rows if row["state"] == "A_high")
    # Prior Beta(1,1), then one selector success and one selector failure.
    assert state_a["final_estimate"] == .5


def test_predicted_attribution_retains_ambiguous_fractional_updates():
    source = np.array([0, 2])
    state = np.array([0, 0])
    prediction = np.array([[.7, .1, .1, .05, .05], [.1, .2, .4, .2, .1]])
    rows = update_policy(source, state, prediction, "PREDICTED_STATE", threshold=.0)
    state_a = next(row for row in rows if row["state"] == "A_high")
    assert 0 < state_a["final_estimate"] < 1
