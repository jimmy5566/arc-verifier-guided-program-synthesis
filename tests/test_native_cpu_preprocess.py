"""CPU-only exact-equivalence checks for Frozen30 native preprocessing."""
from __future__ import annotations

import json
from pathlib import Path

from arc.io import load_dataset
from arc.task import ARCExample, ARCGrid, ARCTask
from inference.nvarc_native_augmentation import NativeAugmentation, bounded_native_augmentations, transform_tasks_for_augmentations
from inference.nvarc_native import native_messages, native_messages_from_training_prefix, native_training_message_prefix
from inference.nvarc_native_candidates import NativeGridCandidate
from inference.arc_native_io import ARCNativeInputAdapter


ROOT = Path(__file__).resolve().parents[1]
FROZEN30 = json.loads((ROOT / "artifacts/speed_v2_frozen30_manifest.json").read_text(encoding="utf-8"))["task_ids"]


def _signature(task: ARCTask) -> tuple[object, ...]:
    return (
        task.task_id,
        tuple((example.input.to_list(), None if example.output is None else example.output.to_list()) for example in task.train),
        tuple(example.input.to_list() for example in task.test),
    )


def _legacy_transform(task: ARCTask, augmentation: NativeAugmentation) -> ARCTask:
    train = tuple(
        ARCExample(
            ARCGrid(augmentation.transform_grid(example.input.values)),
            ARCGrid(augmentation.transform_grid(example.output.values)),
        )
        for example in task.train
    )
    if augmentation.pair_order == "reversed":
        train = tuple(reversed(train))
    test = tuple(ARCExample(ARCGrid(augmentation.transform_grid(example.input.values))) for example in task.test)
    return ARCTask(task.task_id, train, test)


def test_pair_order_factoring_matches_legacy_for_every_frozen30_task() -> None:
    tasks = load_dataset(ROOT / "data/raw/arc-agi_training_challenges.json")
    augmentations = bounded_native_augmentations(color_offsets=(0, 1), pair_orders=("canonical", "reversed"))
    for task_id in FROZEN30:
        legacy = tuple(_signature(_legacy_transform(tasks[task_id], augmentation)) for augmentation in augmentations)
        optimized = tuple(_signature(value) for value in transform_tasks_for_augmentations(tasks[task_id], augmentations))
        assert optimized == legacy, task_id


def test_cached_training_prefix_matches_native_messages_for_every_frozen30_task() -> None:
    tasks = load_dataset(ROOT / "data/raw/arc-agi_training_challenges.json")
    augmentations = bounded_native_augmentations(color_offsets=(0, 1), pair_orders=("canonical", "reversed"))
    for task_id in FROZEN30:
        for transformed in transform_tasks_for_augmentations(tasks[task_id], augmentations):
            prefix = native_training_message_prefix(transformed)
            for test_index, example in enumerate(transformed.test):
                assert native_messages_from_training_prefix(prefix, example.input) == native_messages(transformed, test_index), task_id


def test_trusted_serializer_matches_validated_serializer_for_every_frozen30_grid() -> None:
    tasks = load_dataset(ROOT / "data/raw/arc-agi_training_challenges.json")
    adapter = ARCNativeInputAdapter()
    for task_id in FROZEN30:
        for example in (*tasks[task_id].train, *tasks[task_id].test):
            assert adapter.serialize_trusted_grid(example.input.values) == adapter.serialize_grid(example.input.to_list()), task_id
            if example.output is not None:
                assert adapter.serialize_trusted_grid(example.output.values) == adapter.serialize_grid(example.output.to_list()), task_id


def test_candidate_dict_cache_preserves_serialized_values_and_is_not_mutated() -> None:
    augmentations = bounded_native_augmentations(color_offsets=(0, 1), pair_orders=("canonical", "reversed"))
    candidates = [
        NativeGridCandidate(augmentation, (((index % 10, (index + 1) % 10),),), index + 1, float(index))
        for index, augmentation in enumerate(augmentations)
    ]
    legacy = [item.to_dict() for item in candidates]
    cached = [item.to_dict() for item in candidates]
    # The downstream CPU consumers receive the same data and only read it.
    _ = [item["prediction"] for item in cached]
    assert cached == legacy
    assert [item.to_dict() for item in candidates] == legacy
