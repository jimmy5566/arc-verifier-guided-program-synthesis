from __future__ import annotations

import pytest

from arc.solution_normalization import normalize_arc_solutions


TASKS = ["a", "b"]
COUNTS = {"a": 1, "b": 2}
GRID0 = [[0, 1], [2, 3]]
GRID1 = [[4]]
GRID2 = [[5, 6]]


def test_dict_form_normalizes_compact_and_task_object_values() -> None:
    raw = {"a": [GRID0], "b": {"test": [{"output": GRID1}, {"output": GRID2}]}}
    assert normalize_arc_solutions(raw, task_ids_in_challenge_order=TASKS, expected_output_counts=COUNTS) == {"a": [GRID0], "b": [GRID1, GRID2]}


def test_list_form_uses_explicit_challenge_order() -> None:
    raw = [[GRID0], [GRID1, GRID2]]
    assert normalize_arc_solutions(raw, task_ids_in_challenge_order=TASKS, expected_output_counts=COUNTS) == {"a": [GRID0], "b": [GRID1, GRID2]}


def test_list_length_mismatch_rejected() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        normalize_arc_solutions([[GRID0]], task_ids_in_challenge_order=TASKS, expected_output_counts=COUNTS)


def test_output_count_mismatch_rejected() -> None:
    with pytest.raises(ValueError, match="output count mismatch"):
        normalize_arc_solutions({"a": [GRID0], "b": [GRID1]}, task_ids_in_challenge_order=TASKS, expected_output_counts=COUNTS)


def test_unknown_task_id_rejected() -> None:
    with pytest.raises(ValueError, match="do not exactly match"):
        normalize_arc_solutions({"a": [GRID0], "unknown": [GRID1, GRID2]}, task_ids_in_challenge_order=TASKS, expected_output_counts=COUNTS)


def test_duplicate_challenge_task_id_and_malformed_grid_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        normalize_arc_solutions([[GRID0], [GRID1, GRID2]], task_ids_in_challenge_order=["a", "a"], expected_output_counts={"a": 1})
    with pytest.raises(ValueError, match="rectangular"):
        normalize_arc_solutions({"a": [[[0], [1, 2]]], "b": [GRID1, GRID2]}, task_ids_in_challenge_order=TASKS, expected_output_counts=COUNTS)
