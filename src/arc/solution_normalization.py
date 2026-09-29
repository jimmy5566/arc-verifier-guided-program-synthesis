"""Strict, schema-checked normalization for official ARC solution JSON."""
from __future__ import annotations

from typing import Any, Mapping, Sequence


Grid = list[list[int]]


def _grid(value: Any, *, task_id: str, output_index: int) -> Grid:
    if not isinstance(value, list) or not value or not all(isinstance(row, list) and row for row in value):
        raise ValueError(f"{task_id}:o{output_index}: grid must be non-empty rows")
    width = len(value[0])
    if any(len(row) != width for row in value):
        raise ValueError(f"{task_id}:o{output_index}: grid is not rectangular")
    if any(isinstance(cell, bool) or not isinstance(cell, int) or cell < 0 or cell > 9 for row in value for cell in row):
        raise ValueError(f"{task_id}:o{output_index}: grid contains invalid ARC colors")
    return value


def _outputs(value: Any, *, task_id: str) -> list[Any]:
    """Accept only documented compact or task-object values."""
    if isinstance(value, list):
        return value
    if isinstance(value, dict) and set(value) == {"test"} and isinstance(value["test"], list):
        result = []
        for index, item in enumerate(value["test"]):
            if not isinstance(item, dict) or set(item) != {"output"}:
                raise ValueError(f"{task_id}: test[{index}] must contain exactly an output grid")
            result.append(item["output"])
        return result
    raise ValueError(f"{task_id}: unsupported solution value schema {type(value).__name__}")


def normalize_arc_solutions(raw: Any, *, task_ids_in_challenge_order: Sequence[str], expected_output_counts: Mapping[str, int]) -> dict[str, list[Grid]]:
    """Return ``task_id -> output grids`` without inferring list ordering.

    A list-form collection is valid only with the exact supplied challenge order;
    a dict-form collection must name every expected task exactly once.
    """
    task_ids = list(task_ids_in_challenge_order)
    if not task_ids or len(task_ids) != len(set(task_ids)):
        raise ValueError("challenge task ordering is empty or contains duplicate task IDs")
    if set(task_ids) != set(expected_output_counts):
        raise ValueError("expected output counts must cover exactly the challenge task IDs")
    if isinstance(raw, dict):
        if set(raw) != set(task_ids):
            missing, extra = sorted(set(task_ids) - set(raw)), sorted(set(raw) - set(task_ids))
            raise ValueError(f"solution task IDs do not exactly match challenge: missing={missing[:3]} extra={extra[:3]}")
        source = raw
    elif isinstance(raw, list):
        if len(raw) != len(task_ids):
            raise ValueError(f"list-form solution length mismatch: {len(raw)} != {len(task_ids)}")
        source = dict(zip(task_ids, raw, strict=True))
    else:
        raise ValueError(f"unsupported top-level solutions schema {type(raw).__name__}")

    normalized: dict[str, list[Grid]] = {}
    for task_id in task_ids:
        outputs = _outputs(source[task_id], task_id=task_id)
        expected = int(expected_output_counts[task_id])
        if len(outputs) != expected:
            raise ValueError(f"{task_id}: output count mismatch: {len(outputs)} != {expected}")
        normalized[task_id] = [_grid(grid, task_id=task_id, output_index=index) for index, grid in enumerate(outputs)]
    return normalized
