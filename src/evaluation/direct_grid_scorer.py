"""Local post-freeze exact and partial metrics for direct ARC grid outputs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from inference.direct_grid_solver import Attempt, Grid, grids_equal, validate_grid


@dataclass(frozen=True)
class DirectGridScore:
    attempt_1_valid: bool
    attempt_2_valid: bool
    top1_exact: bool
    any_of_2_exact: bool
    correct_dimensions: bool
    color_set_valid: bool
    cell_accuracy: float
    changed_precision: float
    changed_recall: float
    failure: str | None


@dataclass(frozen=True)
class DirectTaskScore:
    """Task-level exactness with per-test partial metrics averaged fairly."""

    attempt_1_valid: bool
    attempt_2_valid: bool
    top1_exact: bool
    any_of_2_exact: bool
    correct_dimensions: bool
    color_set_valid: bool
    cell_accuracy: float
    changed_precision: float
    changed_recall: float
    failure: str | None


def _shape(grid: Grid | None) -> tuple[int, int] | None:
    return None if grid is None else (len(grid), len(grid[0]))


def _changes(source: Grid, grid: Grid) -> set[tuple[int, int]]:
    return {(row, col) for row in range(len(source)) for col in range(len(source[0])) if source[row][col] != grid[row][col]}


def _failure(first: Grid | None, target: Grid, source: Grid, forensic_classification: str | None) -> str | None:
    if first is None:
        return "INVALID_GRID_OUTPUT"
    if _shape(first) != _shape(target):
        return "WRONG_DIMENSION"
    if grids_equal(first, target):
        return None
    if _shape(source) == _shape(target):
        predicted_changes, expected_changes = _changes(source, first), _changes(source, target)
        if predicted_changes == expected_changes:
            return "RIGHT_SHAPE_WRONG_COLOR"
        if forensic_classification == "MISSING_ITERATION_OR_REPEAT":
            return "MISSED_ITERATION" if not predicted_changes else "WRONG_ITERATION_COUNT"
        if forensic_classification == "MISSING_RELATIONAL_CAPABILITY":
            return "WRONG_RELATION"
        if forensic_classification == "MISSING_CONDITIONAL_OR_ROLE_LOGIC":
            return "WRONG_CONDITIONAL_RULE"
        if predicted_changes < expected_changes:
            return "UNDER_TRANSFORM"
        if expected_changes < predicted_changes:
            return "OVER_TRANSFORM"
        if len(predicted_changes) == len(expected_changes):
            return "RIGHT_OBJECT_WRONG_POSITION"
        if predicted_changes & expected_changes:
            return "PARTIAL_TRANSFORMATION"
    return "OTHER_GRID_ERROR"


def score_direct_prediction(prediction: Any, target: Any, test_input: Any, forensic_classification: str | None = None) -> DirectGridScore:
    """Score after the prediction artifact has already been frozen.

    Color validity is the ARC palette validity (all values are integer 0..9),
    independent of whether a color appeared in the test input.
    """
    first, second = validate_grid(prediction.get("attempt_1")), validate_grid(prediction.get("attempt_2"))
    expected, source = validate_grid(target), validate_grid(test_input)
    if expected is None or source is None:
        raise ValueError("targets and test inputs must be valid ARC grids")
    top1 = grids_equal(first, expected)
    any_exact = top1 or grids_equal(second, expected)
    dimensions = _shape(first) == _shape(expected)
    cells = 0.0
    precision = recall = 0.0
    if dimensions and first is not None:
        total = len(expected) * len(expected[0])
        cells = sum(first[row][col] == expected[row][col] for row in range(len(expected)) for col in range(len(expected[0]))) / total
        if _shape(source) == _shape(expected):
            predicted_changes, expected_changes = _changes(source, first), _changes(source, expected)
            overlap = len(predicted_changes & expected_changes)
            precision = 1.0 if not predicted_changes and not expected_changes else (overlap / len(predicted_changes) if predicted_changes else 0.0)
            recall = 1.0 if not expected_changes and not predicted_changes else (overlap / len(expected_changes) if expected_changes else 0.0)
    return DirectGridScore(
        attempt_1_valid=first is not None,
        attempt_2_valid=second is not None,
        top1_exact=top1,
        any_of_2_exact=any_exact,
        correct_dimensions=dimensions,
        color_set_valid=first is not None,
        cell_accuracy=cells,
        changed_precision=precision,
        changed_recall=recall,
        failure=_failure(first, expected, source, forensic_classification),
    )


def _attempt_grids(value: Attempt | None, test_count: int) -> list[Grid] | None:
    if value is None:
        return None
    if test_count == 1:
        grid = validate_grid(value)
        return None if grid is None else [grid]
    if not isinstance(value, list) or len(value) != test_count:
        return None
    grids = [validate_grid(item) for item in value]
    return None if any(item is None for item in grids) else [item for item in grids if item is not None]


def score_direct_task_prediction(prediction: Any, targets: list[Any], test_inputs: list[Any], forensic_classification: str | None = None) -> DirectTaskScore:
    """Score one ARC task, including its occasional multiple test inputs."""
    if len(targets) != len(test_inputs) or not targets:
        raise ValueError("targets and test inputs must have matching non-zero length")
    first = _attempt_grids(prediction.get("attempt_1"), len(targets))
    second = _attempt_grids(prediction.get("attempt_2"), len(targets))
    # Reuse the single-grid scorer for all partial measurements. Passing an
    # invalid placeholder preserves its deterministic invalid classification.
    first_records = [score_direct_prediction({"attempt_1": first[index] if first else None, "attempt_2": None}, target, source, forensic_classification) for index, (target, source) in enumerate(zip(targets, test_inputs))]
    second_records = [score_direct_prediction({"attempt_1": second[index] if second else None, "attempt_2": None}, target, source, forensic_classification) for index, (target, source) in enumerate(zip(targets, test_inputs))]
    top1 = first is not None and all(item.top1_exact for item in first_records)
    second_exact = second is not None and all(item.top1_exact for item in second_records)
    failures = [item.failure for item in first_records if item.failure]
    return DirectTaskScore(
        attempt_1_valid=first is not None,
        attempt_2_valid=second is not None,
        top1_exact=top1,
        any_of_2_exact=top1 or second_exact,
        correct_dimensions=first is not None and all(item.correct_dimensions for item in first_records),
        color_set_valid=first is not None,
        cell_accuracy=sum(item.cell_accuracy for item in first_records) / len(first_records),
        changed_precision=sum(item.changed_precision for item in first_records) / len(first_records),
        changed_recall=sum(item.changed_recall for item in first_records) / len(first_records),
        failure=failures[0] if failures else None,
    )


def recognition_overlap(recognition_success: dict[str, bool], direct_exact: dict[str, bool]) -> dict[str, int]:
    """Return mutually exclusive task-level overlap categories."""
    if set(recognition_success) != set(direct_exact):
        raise ValueError("recognition and direct task sets must match")
    result = {
        "recognition_success_direct_exact": 0,
        "recognition_success_direct_fail": 0,
        "recognition_fail_direct_exact": 0,
        "recognition_fail_direct_fail": 0,
    }
    for task_id in recognition_success:
        key = ("recognition_success" if recognition_success[task_id] else "recognition_fail") + ("_direct_exact" if direct_exact[task_id] else "_direct_fail")
        result[key] += 1
    return result
