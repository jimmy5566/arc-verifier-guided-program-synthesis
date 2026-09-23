"""Frozen, target-blind hybrid representation specifications for native ARC.

This module transforms an ARC task's presentation only.  It deliberately has
no rule inference, candidate ranking, or access to target grids.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from inference.nvarc_native_augmentation import NativeAugmentation
from inference.nvarc_reference_like_augmentation import ReferenceLikeAugmentation, reference_like_variants


_GEOMETRIES = ("identity", "rot90", "rot180", "rot270", "flip_lr", "flip_ud", "transpose", "anti_transpose")


def redundant_geometry_indices(baseline_record: dict[str, Any]) -> tuple[int, int]:
    """Choose two non-identity geometry slots using frozen duplicate support.

    A candidate shared by two or more baseline geometry views contributes one
    redundancy unit to every view in that support set.  Ties use the original
    Aug8 order, keeping the choice independent of target correctness.
    """
    scores = {geometry: 0 for geometry in _GEOMETRIES}
    for candidate in baseline_record.get("candidates", ()):
        support = {
            str(item.get("geometry"))
            for item in candidate.get("support_augmentations", ())
            if int(item.get("color_offset", -1)) == 0 and item.get("pair_order") == "canonical"
        }
        overlap = max(0, len(support) - 1)
        for geometry in support:
            if geometry in scores:
                scores[geometry] += overlap
    ranked = sorted(range(1, len(_GEOMETRIES)), key=lambda index: (-scores[_GEOMETRIES[index]], index))
    return tuple(ranked[:2])  # type: ignore[return-value]


@dataclass(frozen=True)
class HybridRepresentationView:
    """One reversible view, backed by an existing native transform primitive."""

    slot: int
    kind: str
    source_index: int
    augmentation: NativeAugmentation | ReferenceLikeAugmentation

    def transform_task(self, task: Any) -> Any:
        return self.augmentation.transform_task(task)

    def inverse_grid(self, grid: list[list[int]]) -> list[list[int]]:
        return self.augmentation.inverse_grid(grid)

    def to_dict(self) -> dict[str, Any]:
        return {
            "slot": self.slot,
            "kind": self.kind,
            "source_index": self.source_index,
            "seed_namespace": f"hybrid:{self.kind}:{self.source_index}",
            "augmentation": self.augmentation.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "HybridRepresentationView":
        kind = str(value["kind"])
        raw = dict(value["augmentation"])
        if kind == "geometry":
            augmentation: NativeAugmentation | ReferenceLikeAugmentation = NativeAugmentation(
                geometry=str(raw["geometry"]), color_offset=int(raw["color_offset"]), pair_order=str(raw["pair_order"]),
            )
        elif kind == "reference_like":
            augmentation = ReferenceLikeAugmentation(
                variant_id=int(raw["variant_id"]), geometry=str(raw["geometry"]),
                color_mapping=tuple(int(item) for item in raw["color_mapping"]),
                train_pair_order=tuple(int(item) for item in raw["train_pair_order"]), seed=int(raw["seed"]),
            )
        else:
            raise ValueError(f"unsupported hybrid representation kind: {kind}")
        return cls(slot=int(value["slot"]), kind=kind, source_index=int(value["source_index"]), augmentation=augmentation)


def hybrid_views(task_id: str, train_pair_count: int, baseline_record: dict[str, Any]) -> tuple[HybridRepresentationView, ...]:
    """Return six original geometry views and two deterministic reference views."""
    replace = set(redundant_geometry_indices(baseline_record))
    result: list[HybridRepresentationView] = []
    for index, geometry in enumerate(_GEOMETRIES):
        if index not in replace:
            result.append(HybridRepresentationView(len(result), "geometry", index, NativeAugmentation(geometry=geometry)))
    reference = reference_like_variants(task_id, train_pair_count)
    for index in (6, 7):
        result.append(HybridRepresentationView(len(result), "reference_like", index, reference[index]))
    if len(result) != 8 or result[0].augmentation.to_dict() != NativeAugmentation().to_dict():
        raise AssertionError("hybrid contract must retain the canonical identity view")
    return tuple(result)
