"""Deterministic, non-learned capability retrieval for the V1 diagnostic.

This module is deliberately a *prompt* catalogue selector.  It never changes
the registry, executor, structured schema, or legal program space.
"""
from __future__ import annotations

from collections import defaultdict

from arc.task import ARCTask
from primitives.registry import REGISTRY
from .context import build_task_context


FAMILY_BY_CATEGORY = {
    "region": "Region", "count": "Counting", "generation": "Generation",
    "graph": "Graph/Path", "line": "Relation", "sequence": "Sequence",
    "iteration": "Iteration", "pattern": "Pattern", "transform": "Geometry",
    "render": "Object", "selection": "Object", "relation": "Relation",
    "composition": "Relation",
}


def primitive_family(primitive_id: str) -> str:
    return FAMILY_BY_CATEGORY.get(REGISTRY[primitive_id].category, "Other")


def family_scores(task: ARCTask) -> dict[str, int]:
    """Score families from train/test structural evidence only, never task ID."""
    context = build_task_context(task)
    scores: dict[str, int] = defaultdict(int)
    for example in context["train"]:
        source, target = example["input_summary"], example["output_summary"]
        dimensions = example["output_dimension_relation"]
        scores["Geometry"] += int(not dimensions["same_shape"]) * 4
        # Context V1 deliberately exposes only compact, task-ID-free features.
        # A small palette with a dimensional change is the available pattern
        # proxy; do not assume richer feature fields exist.
        scores["Pattern"] += int(len(source["palette"]) <= 3 and not dimensions["same_shape"]) * 3
        scores["Region"] += int(source["enclosed_region_count"] > 0) * 5
        scores["Graph/Path"] += int(source["graph_is_simple_path"]) * 5
        scores["Relation"] += int(bool(source["full_horizontal_separators"] or source["full_vertical_separators"])) * 5
        scores["Object"] += int(source["object_count"] != target["object_count"]) * 3
        scores["Counting"] += int(len(target["palette"]) <= 3) * 1
        scores["Generation"] += int(not dimensions["same_shape"]) * 3
    # Keep all core families eligible under ties; this is not hard exclusion.
    for family in set(FAMILY_BY_CATEGORY.values()) | {"Other"}:
        scores.setdefault(family, 0)
    return dict(scores)


def retrieve_capability_ids(task: ARCTask, limit: int) -> tuple[str, ...]:
    if limit <= 0 or limit > len(REGISTRY):
        raise ValueError("limit must be within the registry size")
    scores = family_scores(task)
    ranked = sorted(
        REGISTRY,
        key=lambda primitive_id: (-scores[primitive_family(primitive_id)], primitive_family(primitive_id), primitive_id),
    )
    return tuple(ranked[:limit])
