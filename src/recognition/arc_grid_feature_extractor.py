"""Train-pair-only deterministic observations for semantic recognition."""
from __future__ import annotations

from collections import Counter, deque
from typing import Any

import numpy as np

from arc.task import ARCTask


def _components(grid: np.ndarray) -> list[dict[str, Any]]:
    seen: set[tuple[int, int]] = set(); result: list[dict[str, Any]] = []
    for row, col in np.ndindex(grid.shape):
        if (row, col) in seen: continue
        color = int(grid[row, col]); queue = deque([(row, col)]); seen.add((row, col)); cells = []
        while queue:
            r, c = queue.popleft(); cells.append((r, c))
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and (nr, nc) not in seen and int(grid[nr, nc]) == color:
                    seen.add((nr, nc)); queue.append((nr, nc))
        if color != int(Counter(map(int, grid.ravel())).most_common(1)[0][0]):
            rows, cols = zip(*cells); result.append({"color": color, "size": len(cells), "bbox": [min(rows), min(cols), max(rows), max(cols)]})
    return sorted(result, key=lambda item: (item["color"], item["bbox"]))


def _grid(values: np.ndarray) -> list[list[int]]:
    return np.asarray(values).astype(int).tolist()


def task_payload(task: ARCTask) -> dict[str, Any]:
    """Full train grids plus compact observations; intentionally no test pair."""
    pairs = []
    for example in task.train:
        source, target = example.input.values, example.output.values
        pairs.append({
            "input": _grid(source), "output": _grid(target),
            "input_shape": list(source.shape), "output_shape": list(target.shape),
            "input_colors": sorted(map(int, np.unique(source))), "output_colors": sorted(map(int, np.unique(target))),
            "changed_cells": None if source.shape != target.shape else int(np.count_nonzero(source != target)),
            "input_components": _components(source), "output_components": _components(target),
        })
    return {"train_pairs": pairs, "observation_policy": "all observations are deterministic functions of train input/output pairs only"}
