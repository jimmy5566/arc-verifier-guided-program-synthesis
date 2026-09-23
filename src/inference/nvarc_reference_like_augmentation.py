"""Target-blind reversible NVARC-style representation views.

This module is deliberately separate from the frozen production Aug8
transport.  It represents only the public NVARC augmentation family:
full train-pair permutations, D4 geometry, and arbitrary colour
permutations.  It contains no solver, scorer, or target-dependent logic.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any

import numpy as np

from arc.task import ARCExample, ARCGrid, ARCTask
from inference.nvarc_native_augmentation import Geometry, _INVERSE, _geometric


GEOMETRIES: tuple[Geometry, ...] = (
    "identity", "rot90", "rot180", "rot270", "flip_lr", "flip_ud", "transpose", "anti_transpose",
)


def _seed(task_id: str, variant_id: int) -> int:
    raw = sha256(f"{task_id}:{variant_id}".encode("utf-8")).digest()
    return int.from_bytes(raw[:8], "big") % (2**32)


@dataclass(frozen=True)
class ReferenceLikeAugmentation:
    """One fully reversible public-family representation instance."""

    variant_id: int
    geometry: Geometry
    color_mapping: tuple[int, ...]
    train_pair_order: tuple[int, ...]
    seed: int

    def __post_init__(self) -> None:
        if self.geometry not in GEOMETRIES:
            raise ValueError(f"unsupported geometry: {self.geometry}")
        if len(self.color_mapping) != 10 or set(self.color_mapping) != set(range(10)):
            raise ValueError("color_mapping must be a permutation of 0..9")
        if len(self.train_pair_order) != len(set(self.train_pair_order)) or set(self.train_pair_order) != set(range(len(self.train_pair_order))):
            raise ValueError("train_pair_order must be a full permutation")

    @property
    def inverse_color_mapping(self) -> tuple[int, ...]:
        inverse = np.argsort(np.asarray(self.color_mapping, dtype=np.int16))
        return tuple(int(value) for value in inverse.tolist())

    def transform_grid(self, grid: Any) -> np.ndarray:
        geometric = _geometric(np.asarray(grid, dtype=np.int16), self.geometry)
        return np.asarray(self.color_mapping, dtype=np.int16)[geometric]

    def inverse_grid(self, grid: list[list[int]] | np.ndarray) -> list[list[int]]:
        recolored = np.asarray(self.inverse_color_mapping, dtype=np.int16)[np.asarray(grid, dtype=np.int16)]
        return _geometric(recolored, _INVERSE[self.geometry]).astype(int).tolist()

    def transform_task(self, task: ARCTask) -> ARCTask:
        if len(task.train) != len(self.train_pair_order):
            raise ValueError("variant train_pair_order does not match task")
        transformed = tuple(
            ARCExample(ARCGrid(self.transform_grid(example.input.values)), ARCGrid(self.transform_grid(example.output.values)))
            for example in task.train
        )
        train = tuple(transformed[index] for index in self.train_pair_order)
        test = tuple(ARCExample(ARCGrid(self.transform_grid(example.input.values))) for example in task.test)
        return ARCTask(task.task_id, train, test)

    def to_dict(self) -> dict[str, object]:
        return {
            "family": "reference_like_d4_color_permutation_pair_order",
            "variant_id": self.variant_id,
            "geometry": self.geometry,
            "color_mapping": list(self.color_mapping),
            "inverse_color_mapping": list(self.inverse_color_mapping),
            "train_pair_order": list(self.train_pair_order),
            "seed": self.seed,
        }


def reference_like_variants(task_id: str, train_pair_count: int, *, count: int = 8) -> tuple[ReferenceLikeAugmentation, ...]:
    """Create exactly ``count`` stable views without inspecting ARC targets."""
    if count != 8:
        raise ValueError("the Smoke30 representation condition is fixed to exactly eight variants")
    if train_pair_count < 1:
        raise ValueError("ARC task must have at least one train pair")
    variants: list[ReferenceLikeAugmentation] = []
    for variant_id, geometry in enumerate(GEOMETRIES):
        seed = _seed(task_id, variant_id)
        rng = np.random.RandomState(seed)
        # This mirrors NVARC's public ``np.argsort(rng.permutation(10))``
        # colour representation, while retaining the mapping for inversion.
        mapping = tuple(int(value) for value in np.argsort(rng.permutation(10)).tolist())
        pair_order = tuple(int(value) for value in rng.permutation(train_pair_count).tolist())
        variants.append(ReferenceLikeAugmentation(variant_id, geometry, mapping, pair_order, seed))
    return tuple(variants)
