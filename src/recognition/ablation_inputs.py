"""Deterministic, train-pair-only input representations for Track A.

The functions here deliberately have no knowledge of oracle IR, solutions,
macro libraries, compilers, or execution.  They are reusable observations of
the displayed train grids, not hypotheses about a particular ARC task.
"""
from __future__ import annotations

from collections import Counter, deque
from typing import Any, Iterable

import numpy as np

from arc.task import ARCTask
from .arc_grid_feature_extractor import task_payload as current_feature_payload


RAW_GRID_ONLY = "A0_RAW_GRID_ONLY"
FEATURES_ONLY = "A1_DETERMINISTIC_FEATURES_ONLY"
RAW_CURRENT = "A2_RAW_PLUS_CURRENT_FEATURES"
RAW_RELATION_GRAPH = "A3_RAW_PLUS_OBJECT_RELATION_GRAPH"
TRACK_A_CONDITIONS = (RAW_GRID_ONLY, FEATURES_ONLY, RAW_CURRENT, RAW_RELATION_GRAPH)


def _background(grid: np.ndarray) -> int:
    counts = Counter(map(int, np.asarray(grid).ravel()))
    return min((-count, color) for color, count in counts.items())[1]


def _components(grid: np.ndarray) -> list[dict[str, Any]]:
    """Return deterministic non-background 4-connected objects."""
    grid = np.asarray(grid)
    background = _background(grid)
    seen: set[tuple[int, int]] = set()
    output: list[dict[str, Any]] = []
    for row, col in np.ndindex(grid.shape):
        if (row, col) in seen or int(grid[row, col]) == background:
            continue
        color = int(grid[row, col])
        queue = deque([(row, col)])
        seen.add((row, col))
        cells: list[tuple[int, int]] = []
        while queue:
            r, c = queue.popleft()
            cells.append((r, c))
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if (
                    0 <= nr < grid.shape[0]
                    and 0 <= nc < grid.shape[1]
                    and (nr, nc) not in seen
                    and int(grid[nr, nc]) == color
                ):
                    seen.add((nr, nc))
                    queue.append((nr, nc))
        rows, cols = zip(*cells)
        top, left, bottom, right = min(rows), min(cols), max(rows), max(cols)
        normalized = tuple(sorted((r - top, c - left) for r, c in cells))
        output.append(
            {
                "color": color,
                "area": len(cells),
                "bbox": [top, left, bottom, right],
                "centroid": [round(sum(rows) / len(rows), 3), round(sum(cols) / len(cols), 3)],
                "width": right - left + 1,
                "height": bottom - top + 1,
                "shape_signature": ";".join(f"{r},{c}" for r, c in normalized),
            }
        )
    return sorted(output, key=lambda item: (item["bbox"], item["color"], item["shape_signature"]))


def _symmetry(grid: np.ndarray) -> dict[str, bool]:
    grid = np.asarray(grid)
    return {
        "horizontal": bool(np.array_equal(grid, np.flipud(grid))),
        "vertical": bool(np.array_equal(grid, np.fliplr(grid))),
        "main_diagonal": bool(grid.shape[0] == grid.shape[1] and np.array_equal(grid, grid.T)),
    }


def _periods(grid: np.ndarray) -> dict[str, list[int]]:
    grid = np.asarray(grid)
    rows, cols = grid.shape
    return {
        "row": [period for period in range(1, rows) if rows % period == 0 and np.array_equal(grid, np.tile(grid[:period], (rows // period, 1)))],
        "column": [period for period in range(1, cols) if cols % period == 0 and np.array_equal(grid, np.tile(grid[:, :period], (1, cols // period)))],
    }


def _grid_observation(grid: np.ndarray, *, include_objects: bool) -> dict[str, Any]:
    grid = np.asarray(grid)
    colors, counts = np.unique(grid, return_counts=True)
    observation: dict[str, Any] = {
        "shape": list(grid.shape),
        "background": _background(grid),
        "color_counts": [[int(color), int(count)] for color, count in zip(colors, counts)],
        "symmetry": _symmetry(grid),
        "periodicity": _periods(grid),
    }
    if include_objects:
        observation["objects"] = _components(grid)
    return observation


def _edge(left: dict[str, Any], right: dict[str, Any]) -> list[dict[str, Any]]:
    """Finite generic pairwise relations; numeric delta is always explicit."""
    lbox, rbox = left["bbox"], right["bbox"]
    lr, lc = left["centroid"]
    rr, rc = right["centroid"]
    relations: list[str] = []
    if lc < rc:
        relations.append("LEFT_OF")
    if lc > rc:
        relations.append("RIGHT_OF")
    if lr < rr:
        relations.append("ABOVE")
    if lr > rr:
        relations.append("BELOW")
    if lr == rr:
        relations.append("ALIGNED_ROW")
    if lc == rc:
        relations.append("ALIGNED_COLUMN")
    if lbox[0] <= rbox[0] and lbox[1] <= rbox[1] and lbox[2] >= rbox[2] and lbox[3] >= rbox[3]:
        relations.append("CONTAINS")
    if rbox[0] <= lbox[0] and rbox[1] <= lbox[1] and rbox[2] >= lbox[2] and rbox[3] >= lbox[3]:
        relations.append("INSIDE")
    if not (lbox[2] < rbox[0] or rbox[2] < lbox[0] or lbox[3] < rbox[1] or rbox[3] < lbox[1]):
        relations.append("OVERLAPS")
    if left["shape_signature"] == right["shape_signature"]:
        relations.append("SAME_SHAPE")
    if left["color"] == right["color"]:
        relations.append("SAME_COLOR")
    if left["area"] == right["area"]:
        relations.append("SAME_SIZE")
    return [
        {
            "relation": relation,
            "delta": [round(rr - lr, 3), round(rc - lc, 3)],
        }
        for relation in relations
    ]


def _object_graph(grid: np.ndarray, name: str) -> dict[str, Any]:
    objects = _components(grid)
    # Dense multicolour grids can contain hundreds of isolated one-cell noise
    # components.  Full pixels remain in A3's raw-grid channel; the graph has
    # a deterministic bounded node budget: retain at most four canonical
    # components per colour, ordered by area then geometry.  This is
    # independent of task identity or output and is necessary to keep the
    # structured representation consumable by the fixed model context.
    by_color: dict[int, list[dict[str, Any]]] = {}
    for item in objects:
        by_color.setdefault(item["color"], []).append(item)
    retained: list[dict[str, Any]] = []
    suppressed: dict[int, int] = {}
    for color, values in sorted(by_color.items()):
        selected = sorted(values, key=lambda item: (-item["area"], item["bbox"], item["shape_signature"]))[:4]
        retained.extend(selected)
        suppressed[color] = len(values) - len(selected)
    retained.sort(key=lambda item: (item["bbox"], item["color"], item["shape_signature"]))
    nodes = [{"object_id": f"{name}_{index}", **item} for index, item in enumerate(retained)]
    # A complete graph is not an observation a bounded-context recognizer can
    # consume.  Preserve *all* object nodes, but retain only deterministic
    # nearest-neighbour relations (one nearest node for each cardinal sector)
    # plus containment/overlap/touching relations.  This is a graph encoding,
    # not a task-dependent lossy selection.
    edges: list[dict[str, Any]] = []
    seen_edges: set[tuple[str, str]] = set()
    for source_index, source in enumerate(nodes):
        sectors: dict[str, tuple[float, dict[str, Any]]] = {}
        for target_index, target in enumerate(nodes):
            if source_index == target_index:
                continue
            dr, dc = target["centroid"][0] - source["centroid"][0], target["centroid"][1] - source["centroid"][1]
            sector = "RIGHT" if abs(dc) >= abs(dr) and dc > 0 else "LEFT" if abs(dc) >= abs(dr) else "DOWN" if dr > 0 else "UP"
            distance = abs(dr) + abs(dc)
            if sector not in sectors or (distance, target["object_id"]) < (sectors[sector][0], sectors[sector][1]["object_id"]):
                sectors[sector] = (distance, target)
        for _distance, target in sectors.values():
            relationships = _edge(source, target)
            key = (source["object_id"], target["object_id"])
            if relationships and key not in seen_edges:
                seen_edges.add(key)
                edges.append({"source": source["object_id"], "target": target["object_id"], "relations": [item["relation"] for item in relationships], "delta": relationships[0]["delta"]})
    return {
        "nodes": nodes,
        "edge_fields": ["source", "target", "relations", "delta"],
        "edges": [[edge["source"], edge["target"], edge["relations"], edge["delta"]] for edge in edges],
        "suppressed_component_count_by_color": [[color, count] for color, count in sorted(suppressed.items()) if count],
    }


def _correspondences(source_graph: dict[str, Any], target_graph: dict[str, Any]) -> list[dict[str, Any]]:
    links: list[dict[str, Any]] = []
    for source in source_graph["nodes"]:
        candidates: list[tuple[int, str, dict[str, Any]]] = []
        for target in target_graph["nodes"]:
            exact_shape = source["shape_signature"] == target["shape_signature"]
            size_match = source["area"] == target["area"]
            color_match = source["color"] == target["color"]
            score = 4 * int(exact_shape) + 2 * int(size_match) + int(color_match)
            if not score:
                continue
            delta = [round(target["centroid"][0] - source["centroid"][0], 3), round(target["centroid"][1] - source["centroid"][1], 3)]
            labels = []
            if exact_shape:
                labels.append("SAME_SHAPE")
            if size_match:
                labels.append("SAME_SIZE")
            if color_match:
                labels.append("SAME_COLOR")
            if delta != [0.0, 0.0]:
                labels.append("MOVED_RELATIVE")
            if exact_shape and not color_match:
                labels.append("RECOLORED")
            candidates.append((score, target["object_id"], {"source": source["object_id"], "target": target["object_id"], "relations": labels, "translation_delta": delta, "color_delta": target["color"] - source["color"], "size_delta": target["area"] - source["area"]}))
        # Keep a single best candidate per source, avoiding quadratic
        # same-color singleton correspondences on dense ARC grids.
        links.extend(item for _score, _target, item in sorted(candidates, key=lambda item: (-item[0], item[1]))[:1])
    return links


def deterministic_features(task: ARCTask) -> dict[str, Any]:
    """Detailed train-pair features, without serializing any grid cells."""
    pairs = []
    for example in task.train:
        source, target = example.input.values, example.output.values
        source_observation = _grid_observation(source, include_objects=True)
        target_observation = _grid_observation(target, include_objects=True)
        pairs.append(
            {
                "input": source_observation,
                "output": target_observation,
                "changed_cells": int(np.count_nonzero(source != target)) if source.shape == target.shape else None,
                "shape_delta": [int(target.shape[0] - source.shape[0]), int(target.shape[1] - source.shape[1])],
            }
        )
    return {"train_pair_features": pairs, "policy": "deterministic functions of train input/output grids; no grid serialization"}


def relation_graph_features(task: ARCTask) -> dict[str, Any]:
    """Pixel evidence plus an object/relation graph for each train pair."""
    pairs = []
    for pair_index, example in enumerate(task.train):
        source, target = example.input.values, example.output.values
        source_graph, target_graph = _object_graph(source, f"p{pair_index}_in"), _object_graph(target, f"p{pair_index}_out")
        pairs.append(
            {
                "input_graph": source_graph,
                "output_graph": target_graph,
                "correspondence_fields": ["source", "target", "relations", "translation_delta", "color_delta", "size_delta"],
                "candidate_correspondences": [
                    [item["source"], item["target"], item["relations"], item["translation_delta"], item["color_delta"], item["size_delta"]]
                    for item in _correspondences(source_graph, target_graph)
                ],
            }
        )
    return {"train_pair_relation_graphs": pairs, "policy": "deterministic train-pair object graph; object IDs are local temporary indexes"}


def raw_grids(task: ARCTask) -> dict[str, Any]:
    return {
        "train_pairs": [
            {"input": np.asarray(example.input.values).astype(int).tolist(), "output": np.asarray(example.output.values).astype(int).tolist()}
            for example in task.train
        ],
        "policy": "complete train input/output grids only",
    }


def payload_for_condition(task: ARCTask, condition: str) -> dict[str, Any]:
    if condition == RAW_GRID_ONLY:
        return raw_grids(task)
    if condition == FEATURES_ONLY:
        return deterministic_features(task)
    if condition == RAW_CURRENT:
        return current_feature_payload(task)
    if condition == RAW_RELATION_GRAPH:
        return {"raw_grids": raw_grids(task), "relation_graph": relation_graph_features(task)}
    raise ValueError(f"unknown input-ablation condition: {condition}")


def no_test_oracle_terms(value: object) -> bool:
    """Structural guard used by tests; feature payloads contain no test/oracle key."""
    if isinstance(value, dict):
        return all("test" not in str(key).lower() and "oracle" not in str(key).lower() and no_test_oracle_terms(item) for key, item in value.items())
    if isinstance(value, (tuple, list)):
        return all(no_test_oracle_terms(item) for item in value)
    return True
