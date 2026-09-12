"""Gold-blind raw-grid direct ARC prompting and strict candidate parsing.

This module deliberately knows nothing about semantic IR, capability libraries,
program synthesis, compilers, executors, verification, or ARC targets.  It
turns an ARC task's train pairs and test input into one deterministic prompt,
then validates the two permitted output-grid candidates.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Sequence


Grid = list[list[int]]
Attempt = Grid | list[Grid]


@dataclass(frozen=True)
class DirectGridPrediction:
    """A parsed response, retaining a valid first attempt despite a bad second."""

    status: str
    attempt_1: Attempt | None
    attempt_2: Attempt | None
    raw_text: str


def _as_grid(value: Any) -> Grid:
    if not isinstance(value, list) or not value:
        raise ValueError("grid must be a non-empty list of rows")
    if any(not isinstance(row, list) or not row for row in value):
        raise ValueError("grid rows must be non-empty lists")
    width = len(value[0])
    if any(len(row) != width for row in value):
        raise ValueError("grid rows must have one common width")
    result: Grid = []
    for row in value:
        checked: list[int] = []
        for color in row:
            if isinstance(color, bool) or not isinstance(color, int) or color not in range(10):
                raise ValueError("grid colors must be integer values 0..9")
            checked.append(color)
        result.append(checked)
    return result


def validate_grid(value: Any) -> Grid | None:
    """Return a detached ARC grid or ``None`` when the value is invalid."""
    try:
        return _as_grid(value)
    except ValueError:
        return None


def build_direct_grid_prompt(task: Any) -> str:
    """Serialize complete train evidence and one test input without derived hints.

    ``task_id`` is intentionally not serialized.  A task with several test
    inputs still makes one model call: each attempt is an ordered output-grid
    sequence in that case.  Single-input tasks preserve the compact grid form.
    """
    train = [
        {"input": example.input.to_list(), "output": example.output.to_list()}
        for example in task.train
    ]
    payload = {
        "train_examples": train,
        "test_inputs": [example.input.to_list() for example in task.test],
        "instruction": (
            "Infer the transformation from every training input/output pair and apply it to every test_inputs grid in order. "
            "Return only a JSON object with exactly these keys: attempt_1 and attempt_2. "
            "For one test input, each non-null attempt must be one complete rectangular output grid of integer colors 0 through 9. "
            "For multiple test inputs, each non-null attempt must be an ordered JSON list containing one such grid for each test input. "
            "attempt_2 must be a second candidate or null. Do not include an explanation, rule, code, or any other keys."
        ),
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _as_attempt(value: Any, test_count: int) -> Attempt:
    if test_count < 1:
        raise ValueError("test_count must be positive")
    if test_count == 1:
        return _as_grid(value)
    if not isinstance(value, list) or len(value) != test_count:
        raise ValueError("multi-test attempt must contain one grid per test input")
    return [_as_grid(item) for item in value]


def parse_direct_grid_response(raw_text: str, test_count: int = 1) -> DirectGridPrediction:
    """Strictly parse the frozen one-call/two-candidate response contract."""
    try:
        value = json.loads(raw_text)
    except (TypeError, json.JSONDecodeError):
        return DirectGridPrediction("INVALID_JSON", None, None, raw_text)
    if not isinstance(value, dict) or set(value) != {"attempt_1", "attempt_2"}:
        return DirectGridPrediction("INVALID_OUTPUT_SCHEMA", None, None, raw_text)
    try:
        first = _as_attempt(value["attempt_1"], test_count)
    except ValueError:
        return DirectGridPrediction("INVALID_ATTEMPT_1", None, None, raw_text)
    second_value = value["attempt_2"]
    if second_value is None:
        return DirectGridPrediction("SUCCESS", first, None, raw_text)
    try:
        second = _as_attempt(second_value, test_count)
    except ValueError:
        return DirectGridPrediction("INVALID_ATTEMPT_2", first, None, raw_text)
    return DirectGridPrediction("SUCCESS", first, second, raw_text)


def grids_equal(left: Sequence[Sequence[int]] | None, right: Sequence[Sequence[int]] | None) -> bool:
    return left is not None and right is not None and [list(row) for row in left] == [list(row) for row in right]
