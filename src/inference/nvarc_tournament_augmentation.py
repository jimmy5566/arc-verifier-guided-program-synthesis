"""Frozen, reversible challenger views for the augmentation tournament only."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any

import numpy as np

from arc.task import ARCExample, ARCGrid, ARCTask
from inference.nvarc_native_augmentation import Geometry, _INVERSE, _geometric


def seed_for(task_id: str, aug_id: str) -> int:
    return int.from_bytes(sha256(f"{task_id}:{aug_id}".encode()).digest()[:8], "big") % (2**32)


def _permutation_preserving_zero(seed: int) -> tuple[int, ...]:
    rng = np.random.RandomState(seed)
    values = rng.permutation(np.arange(1, 10)).tolist()
    return (0, *[int(value) for value in values])


def _frequency_mapping(task: ARCTask) -> tuple[int, ...]:
    counts = np.zeros(10, dtype=np.int64)
    for pair in task.train:
        for grid in (pair.input.values, pair.output.values):
            counts += np.bincount(np.asarray(grid, dtype=np.int64).ravel(), minlength=10)
    # Keep background transport fixed; rank foreground by train-only frequency.
    ordered = sorted(range(1, 10), key=lambda color: (-int(counts[color]), color))
    mapping = [0] * 10
    for canonical, source in enumerate(ordered, start=1):
        mapping[source] = canonical
    return tuple(mapping)


@dataclass(frozen=True)
class TournamentAugmentation:
    aug_id: str
    geometry: Geometry
    color_mapping: tuple[int, ...]
    train_pair_order: tuple[int, ...]
    seed: int
    family: str
    duplicate_pair_order: bool

    @property
    def inverse_color_mapping(self) -> tuple[int, ...]:
        return tuple(int(value) for value in np.argsort(np.asarray(self.color_mapping, dtype=np.int16)).tolist())

    def transform_grid(self, grid: Any) -> np.ndarray:
        values = _geometric(np.asarray(grid, dtype=np.int16), self.geometry)
        return np.asarray(self.color_mapping, dtype=np.int16)[values]

    def inverse_grid(self, grid: list[list[int]] | np.ndarray) -> list[list[int]]:
        values = np.asarray(self.inverse_color_mapping, dtype=np.int16)[np.asarray(grid, dtype=np.int16)]
        return _geometric(values, _INVERSE[self.geometry]).astype(int).tolist()

    def transform_task(self, task: ARCTask) -> ARCTask:
        if tuple(sorted(self.train_pair_order)) != tuple(range(len(task.train))):
            raise ValueError("train pair order is not a full permutation")
        pairs = tuple(ARCExample(ARCGrid(self.transform_grid(item.input.values)), ARCGrid(self.transform_grid(item.output.values))) for item in task.train)
        return ARCTask(task.task_id, tuple(pairs[index] for index in self.train_pair_order), tuple(ARCExample(ARCGrid(self.transform_grid(item.input.values))) for item in task.test))

    def to_dict(self) -> dict[str, Any]:
        return {"aug_id": self.aug_id, "family": self.family, "geometry": self.geometry, "color_mapping": list(self.color_mapping), "inverse_color_mapping": list(self.inverse_color_mapping), "train_pair_order": list(self.train_pair_order), "seed": self.seed, "duplicate_pair_order": self.duplicate_pair_order}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "TournamentAugmentation":
        return cls(str(raw["aug_id"]), str(raw["geometry"]), tuple(int(x) for x in raw["color_mapping"]), tuple(int(x) for x in raw["train_pair_order"]), int(raw["seed"]), str(raw["family"]), bool(raw["duplicate_pair_order"]))


def challenger_views(task: ARCTask) -> tuple[TournamentAugmentation, ...]:
    canonical = tuple(range(len(task.train)))
    reversed_order = tuple(reversed(canonical))
    seed3, seed5 = seed_for(task.task_id, "A3"), seed_for(task.task_id, "A5")
    permuted = tuple(int(value) for value in np.random.RandomState(seed3).permutation(len(task.train)).tolist())
    freq = _frequency_mapping(task)
    identity = tuple(range(10))
    specs = (
        ("A1", "pair_order_reversed", "identity", identity, reversed_order),
        ("A2", "pair_order_reversed", "anti_transpose", identity, reversed_order),
        ("A3", "pair_order_permuted", "identity", identity, permuted),
        ("A4", "pair_order_permuted", "anti_transpose", identity, permuted),
        ("A5", "color_permutation_zero_fixed", "identity", _permutation_preserving_zero(seed5), canonical),
        ("A6", "color_permutation_zero_fixed", "anti_transpose", _permutation_preserving_zero(seed_for(task.task_id, "A6")), canonical),
        ("A7", "color_frequency_canonicalization_zero_fixed", "identity", freq, canonical),
        ("A8", "color_frequency_canonicalization_zero_fixed", "anti_transpose", freq, canonical),
    )
    return tuple(TournamentAugmentation(aug_id, geometry, mapping, order, seed_for(task.task_id, aug_id), family, family.startswith("pair_order") and order == canonical) for aug_id, family, geometry, mapping, order in specs)


def legal_spec_templates() -> tuple[dict[str, str], ...]:
    """Finite reversible grammar, before target-blind redundancy reduction."""
    specs: list[dict[str, str]] = []
    for geometry in ("identity", "rot90", "rot180", "rot270", "flip_lr", "flip_ud", "transpose", "anti_transpose"):
        for pair_order in ("canonical", "reversed", "permuted"):
            for color in ("identity", "arbitrary", "background_preserving", "frequency_canonical"):
                if pair_order == "canonical" and color == "identity":
                    continue  # exactly an existing Aug8 view
                spec_id = f"g={geometry};p={pair_order};c={color}"
                specs.append({"augmentation_id": spec_id, "geometry": geometry, "pair_order": pair_order, "color": color})
    return tuple(specs)


def view_from_template(task: ARCTask, template: dict[str, str]) -> TournamentAugmentation:
    """Materialize a legal template with task-local SHA256 seed transport."""
    aug_id, geometry = str(template["augmentation_id"]), str(template["geometry"])
    pair_kind, color_kind = str(template["pair_order"]), str(template["color"])
    canonical = tuple(range(len(task.train)))
    if pair_kind == "canonical":
        order = canonical
    elif pair_kind == "reversed":
        order = tuple(reversed(canonical))
    elif pair_kind == "permuted":
        order = tuple(int(value) for value in np.random.RandomState(seed_for(task.task_id, aug_id + ":pairs")).permutation(len(task.train)).tolist())
    else:
        raise ValueError(f"unsupported pair order: {pair_kind}")
    if color_kind == "identity":
        mapping = tuple(range(10))
    elif color_kind == "arbitrary":
        mapping = tuple(int(value) for value in np.random.RandomState(seed_for(task.task_id, aug_id + ":colors")).permutation(10).tolist())
    elif color_kind == "background_preserving":
        mapping = _permutation_preserving_zero(seed_for(task.task_id, aug_id + ":colors"))
    elif color_kind == "frequency_canonical":
        mapping = _frequency_mapping(task)
    else:
        raise ValueError(f"unsupported color transport: {color_kind}")
    return TournamentAugmentation(aug_id, geometry, mapping, order, seed_for(task.task_id, aug_id), f"pair={pair_kind};color={color_kind}", pair_kind != "canonical" and order == canonical)
