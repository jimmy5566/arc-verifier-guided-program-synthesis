from __future__ import annotations

from arc.task import ARCExample, ARCGrid, ARCTask
from inference.nvarc_hybrid_representation import HybridRepresentationView, hybrid_views, redundant_geometry_indices


def _record() -> dict[str, object]:
    return {
        "candidates": [
            {"support_augmentations": [{"geometry": "rot90", "color_offset": 0, "pair_order": "canonical"}, {"geometry": "rot180", "color_offset": 0, "pair_order": "canonical"}]},
            {"support_augmentations": [{"geometry": "rot90", "color_offset": 0, "pair_order": "canonical"}, {"geometry": "flip_lr", "color_offset": 0, "pair_order": "canonical"}]},
        ],
    }


def test_hybrid_replaces_target_blind_redundant_nonidentity_views() -> None:
    assert redundant_geometry_indices(_record()) == (1, 2)
    task = ARCTask("hybrid-example", (ARCExample(ARCGrid([[0, 1]]), ARCGrid([[1, 0]])),), (ARCExample(ARCGrid([[2, 3]])),))
    first, second = hybrid_views(task.task_id, len(task.train), _record()), hybrid_views(task.task_id, len(task.train), _record())
    assert [item.to_dict() for item in first] == [item.to_dict() for item in second]
    assert len(first) == 8
    assert sum(item.kind == "geometry" for item in first) == 6
    assert sum(item.kind == "reference_like" for item in first) == 2
    assert first[0].to_dict()["augmentation"] == {"geometry": "identity", "color_offset": 0, "pair_order": "canonical"}
    for view in first:
        assert HybridRepresentationView.from_dict(view.to_dict()).to_dict() == view.to_dict()
        assert view.inverse_grid(view.augmentation.transform_grid([[0, 1]])) == [[0, 1]]
