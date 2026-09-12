"""Deterministic, solution-blind structured ARC train-pair observations."""
from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from arc.task import ARCTask
from representations.objects import extract_objects


def _background(grid: np.ndarray) -> int:
    return int(Counter(np.asarray(grid).ravel().tolist()).most_common(1)[0][0])


def _objects(grid: np.ndarray) -> list[dict[str, Any]]:
    result = []
    for item in extract_objects(grid, 4, background=_background(grid)):
        result.append({"color": int(item.color), "area": int(item.area), "bbox": [int(item.top), int(item.left), int(item.height), int(item.width)]})
    return result


def task_features(task: ARCTask) -> dict[str, Any]:
    """Return only features deterministically obtainable from train pairs."""
    pairs = []
    for example in task.train:
        source, target = example.input.values, example.output.values
        changed = np.argwhere(source != target) if source.shape == target.shape else np.empty((0, 2), dtype=int)
        pairs.append({
            "input_shape": list(source.shape), "output_shape": list(target.shape),
            "input_colors": sorted(map(int, np.unique(source))), "output_colors": sorted(map(int, np.unique(target))),
            "changed_cell_count": int(len(changed)) if source.shape == target.shape else None,
            "input_objects": _objects(source), "output_objects": _objects(target),
            "shape_preserved": bool(source.shape == target.shape),
        })
    return {"pair_count": len(pairs), "pairs": pairs, "all_shapes_preserved": all(item["shape_preserved"] for item in pairs)}
