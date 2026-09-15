"""Reversible, task-agnostic transformations for native ARC prompting.

This module only transforms representations.  It never inspects a task for a
rule, scores an ARC answer, or uses test targets.  Every operation is applied
to all grids in a task, and generated grids are inverted before they leave the
native inference condition.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from arc.task import ARCExample, ARCGrid, ARCTask

Geometry = Literal[
    "identity", "rot90", "rot180", "rot270", "flip_lr", "flip_ud", "transpose", "anti_transpose",
]
PairOrder = Literal["canonical", "reversed"]

_GEOMETRIES: tuple[Geometry, ...] = (
    "identity", "rot90", "rot180", "rot270", "flip_lr", "flip_ud", "transpose", "anti_transpose",
)


def _geometric(grid: np.ndarray, geometry: Geometry) -> np.ndarray:
    if geometry == "identity":
        return grid
    if geometry == "rot90":
        return np.rot90(grid, 1)
    if geometry == "rot180":
        return np.rot90(grid, 2)
    if geometry == "rot270":
        return np.rot90(grid, 3)
    if geometry == "flip_lr":
        return np.fliplr(grid)
    if geometry == "flip_ud":
        return np.flipud(grid)
    if geometry == "transpose":
        return grid.T
    if geometry == "anti_transpose":
        return np.rot90(grid.T, 2)
    raise ValueError(f"unsupported native geometry: {geometry}")


_INVERSE: dict[Geometry, Geometry] = {
    "identity": "identity", "rot90": "rot270", "rot180": "rot180", "rot270": "rot90",
    "flip_lr": "flip_lr", "flip_ud": "flip_ud", "transpose": "transpose", "anti_transpose": "anti_transpose",
}


def _transform_task_geometry_and_color(task: ARCTask, geometry: Geometry, color_offset: int) -> ARCTask:
    """Transform pixels once, deliberately leaving train-pair order canonical."""
    augmentation = NativeAugmentation(geometry=geometry, color_offset=color_offset)
    train = tuple(
        ARCExample(
            ARCGrid(augmentation.transform_grid(example.input.values)),
            ARCGrid(augmentation.transform_grid(example.output.values)),
        )
        for example in task.train
    )
    test = tuple(ARCExample(ARCGrid(augmentation.transform_grid(example.input.values))) for example in task.test)
    return ARCTask(task.task_id, train, test)


def _with_pair_order(task: ARCTask, pair_order: PairOrder) -> ARCTask:
    """Return the canonical transformed task or its exact reversed train view."""
    if pair_order == "canonical":
        return task
    if pair_order == "reversed":
        return ARCTask(task.task_id, tuple(reversed(task.train)), task.test)
    raise ValueError(f"unsupported pair order: {pair_order}")


@dataclass(frozen=True)
class NativeAugmentation:
    """A fully invertible global augmentation, independent of task identity."""

    geometry: Geometry = "identity"
    color_offset: int = 0
    pair_order: PairOrder = "canonical"

    def __post_init__(self) -> None:
        if self.geometry not in _GEOMETRIES:
            raise ValueError(f"unsupported geometry: {self.geometry}")
        if not 0 <= self.color_offset < 10:
            raise ValueError("color_offset must be in 0..9")
        if self.pair_order not in ("canonical", "reversed"):
            raise ValueError(f"unsupported pair order: {self.pair_order}")

    def transform_grid(self, grid: np.ndarray) -> np.ndarray:
        values = _geometric(np.asarray(grid, dtype=np.int16), self.geometry)
        return ((values + self.color_offset) % 10).astype(np.int16, copy=False)

    def inverse_grid(self, grid: list[list[int]] | np.ndarray) -> list[list[int]]:
        values = (np.asarray(grid, dtype=np.int16) - self.color_offset) % 10
        return _geometric(values, _INVERSE[self.geometry]).astype(int).tolist()

    def transform_task(self, task: ARCTask) -> ARCTask:
        return _with_pair_order(_transform_task_geometry_and_color(task, self.geometry, self.color_offset), self.pair_order)

    def to_dict(self) -> dict[str, object]:
        return {"geometry": self.geometry, "color_offset": self.color_offset, "pair_order": self.pair_order}


def bounded_native_augmentations(*, color_offsets: tuple[int, ...] = (0, 1), pair_orders: tuple[PairOrder, ...] = ("canonical", "reversed")) -> tuple[NativeAugmentation, ...]:
    """Return a deterministic, fixed 8×color×order pool (32 by default)."""
    if not color_offsets or not pair_orders:
        raise ValueError("native augmentation pool cannot be empty")
    # Geometry varies first so 4/8/32 stage prefixes are meaningful nested
    # pools: first 4 are rotations, first 8 all dihedral geometries, then the
    # same geometry set for the next reversible color/order condition.
    return tuple(NativeAugmentation(geometry, offset, order) for offset in color_offsets for order in pair_orders for geometry in _GEOMETRIES)


def transform_tasks_for_augmentations(task: ARCTask, augmentations: tuple[NativeAugmentation, ...]) -> tuple[ARCTask, ...]:
    """Build views in input order, sharing geometry/color work across pair orders."""
    bases: dict[tuple[Geometry, int], ARCTask] = {}
    transformed: list[ARCTask] = []
    for augmentation in augmentations:
        key = (augmentation.geometry, augmentation.color_offset)
        base = bases.get(key)
        if base is None:
            base = _transform_task_geometry_and_color(task, *key)
            bases[key] = base
        transformed.append(_with_pair_order(base, augmentation.pair_order))
    return tuple(transformed)
