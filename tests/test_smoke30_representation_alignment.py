from __future__ import annotations

from arc.task import ARCExample, ARCGrid, ARCTask
from inference.nvarc_reference_like_augmentation import reference_like_variants
from scripts.run_smoke30_representation_alignment import _deduplicate


def test_reference_like_variants_are_stable_and_reversible() -> None:
    task = ARCTask("example", (ARCExample(ARCGrid([[0, 1], [2, 3]]), ARCGrid([[3, 2], [1, 0]])),), (ARCExample(ARCGrid([[4, 5]])),))
    first, second = reference_like_variants(task.task_id, len(task.train)), reference_like_variants(task.task_id, len(task.train))
    assert [item.to_dict() for item in first] == [item.to_dict() for item in second]
    assert len(first) == 8
    for variant in first:
        assert variant.inverse_grid(variant.transform_grid([[0, 1], [2, 3]])) == [[0, 1], [2, 3]]
        transformed = variant.transform_task(task)
        assert len(transformed.train) == len(task.train)


def test_reference_like_dedup_preserves_raw_variant_support() -> None:
    common = {"prediction": [[[1]]], "completion_tokens": 4, "generation_seconds": 1.0}
    unique = _deduplicate([
        {**common, "representation": {"variant_id": 0}},
        {**common, "representation": {"variant_id": 1}},
        {**common, "prediction": [[[2]]], "representation": {"variant_id": 2}},
    ])
    assert [item["prediction"] for item in unique] == [[[[1]]], [[[2]]]]
    assert unique[0]["support_count"] == 2
    assert [item["variant_id"] for item in unique[0]["provenance"]] == [0, 1]
