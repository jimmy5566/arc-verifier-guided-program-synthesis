"""Independent semantic oracles for the E1 atomic pilot.

This module intentionally does not import the cohort generator.  It derives
answers from observable grid semantics so a generator fall-through cannot
validate itself.
"""
from __future__ import annotations

from collections import deque
from typing import Iterable


Grid = list[list[int]]


def components(grid: Grid) -> list[dict]:
    height, width = len(grid), len(grid[0])
    seen: set[tuple[int, int]] = set()
    found: list[dict] = []
    for y in range(height):
        for x in range(width):
            color = grid[y][x]
            if color == 0 or (y, x) in seen:
                continue
            queue = deque([(y, x)])
            seen.add((y, x))
            cells: list[tuple[int, int]] = []
            while queue:
                cy, cx = queue.popleft()
                cells.append((cy, cx))
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < height and 0 <= nx < width and (ny, nx) not in seen and grid[ny][nx] == color:
                        seen.add((ny, nx))
                        queue.append((ny, nx))
            found.append({"color": color, "cells": tuple(sorted(cells))})
    return found


def _render_component(component: dict, *, canvas: int = 3) -> Grid:
    cells = component["cells"]
    min_y = min(y for y, _ in cells)
    min_x = min(x for _, x in cells)
    max_y = max(y for y, _ in cells)
    max_x = max(x for _, x in cells)
    if max_y - min_y >= canvas or max_x - min_x >= canvas:
        raise ValueError("component does not fit output canvas")
    output = [[0] * canvas for _ in range(canvas)]
    for y, x in cells:
        output[y - min_y][x - min_x] = component["color"]
    return output


def connected_component_count(grid: Grid) -> Grid:
    found = components(grid)
    colors = {component["color"] for component in found}
    if len(colors) != 1 or not 1 <= len(found) <= 3:
        raise ValueError("count cohort must contain one color and one to three components")
    output = [[0] * 3 for _ in range(3)]
    color = next(iter(colors))
    for index in range(len(found)):
        output[1][index] = color
    return output


def object_extraction(grid: Grid) -> Grid:
    found = components(grid)
    if not found:
        raise ValueError("no object to extract")
    largest_size = max(len(component["cells"]) for component in found)
    largest = [component for component in found if len(component["cells"]) == largest_size]
    if len(largest) != 1 or largest_size <= 1:
        raise ValueError("object extraction requires one unique non-singleton largest component")
    return _render_component(largest[0])


def count_then_object_extraction(grid: Grid) -> Grid:
    """Count marker components, then extract that indexed candidate object.

    Markers are the only non-background components entirely in the top two
    rows.  Candidate objects are the non-singleton components below them and
    are ordered left-to-right by bounding-box position.
    """
    found = components(grid)
    markers = [component for component in found if max(y for y, _ in component["cells"]) < 2]
    candidates = [component for component in found if min(y for y, _ in component["cells"]) >= 3 and len(component["cells"]) > 1]
    if not 1 <= len(markers) <= 3:
        raise ValueError("composition grid must contain one to three marker components")
    if len({component["color"] for component in markers}) != 1:
        raise ValueError("composition markers must share one color")
    if len(candidates) != 3:
        raise ValueError("composition grid must contain exactly three candidates")
    candidates.sort(key=lambda component: (min(x for _, x in component["cells"]), min(y for y, _ in component["cells"])))
    return _render_component(candidates[len(markers) - 1])


def target_for_family(family: str, grid: Grid) -> Grid:
    if family == "CONNECTED_COMPONENT_COUNT":
        return connected_component_count(grid)
    if family in {"OBJECT_EXTRACTION", "NO_TRANSFORM_RETENTION"}:
        return object_extraction(grid)
    if family == "COUNT_THEN_OBJECT_EXTRACTION":
        return count_then_object_extraction(grid)
    raise ValueError(f"unsupported semantic family: {family}")


def validate_demonstrations(task: dict, family: str) -> None:
    for index, pair in enumerate(task["train"]):
        expected = target_for_family(family, pair["input"])
        if pair.get("output") != expected:
            raise ValueError(f"demonstration {index} violates {family} semantic oracle")
