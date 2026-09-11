"""Leakage-safe, task-ID-free ARC context for language-model prompting."""
from __future__ import annotations

from typing import Any

import numpy as np

from arc.task import ARCTask
from capabilities import counting, lines, paths, regions


def _grid_summary(grid: np.ndarray) -> dict[str, Any]:
    background = int(np.bincount(grid.ravel()).argmax())
    graph = paths.graph_pixels(grid)
    return {
        "shape": list(grid.shape),
        "palette": sorted(int(value) for value in np.unique(grid)),
        "background_candidate": background,
        "object_count": counting.count_objects(grid),
        "region_count_4": counting.count_regions(grid, 4),
        "enclosed_region_count": len(regions.enclosed_regions(grid)),
        "full_horizontal_separators": lines.full_horizontal(grid),
        "full_vertical_separators": lines.full_vertical(grid),
        "graph_pixel_count": len(graph),
        "graph_is_simple_path": paths.simple_path(graph),
    }


def build_task_context(task: ARCTask, test_index: int = 0) -> dict[str, Any]:
    """Return only train labels and one unlabeled test input; never expose task ID."""
    if not 0 <= test_index < len(task.test):
        raise ValueError("test_index out of range")
    train = []
    for example in task.train:
        source, target = example.input.values, example.output.values
        train.append(
            {
                "input": source.astype(int).tolist(),
                "output": target.astype(int).tolist(),
                "input_summary": _grid_summary(source),
                "output_summary": _grid_summary(target),
                "output_dimension_relation": {"same_shape": source.shape == target.shape, "input_shape": list(source.shape), "output_shape": list(target.shape)},
            }
        )
    test = task.test[test_index].input.values
    return {
        "context_version": "arc_llm_task_context_v1",
        "train": train,
        "test_input": test.astype(int).tolist(),
        "test_summary": _grid_summary(test),
        "task_id_included": False,
    }
