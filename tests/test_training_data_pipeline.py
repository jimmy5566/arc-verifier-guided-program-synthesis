from __future__ import annotations

import json
from pathlib import Path

from training_data.pipeline import (
    CPU_WORKERS,
    EXPECTED_TOKENS,
    IGNORE_INDEX,
    canonical_hash,
    collate,
    deduplicate,
    exclude_leakage,
    native_tokenize_with_labels,
    normalize_task,
    parallel_canonicalize,
    split_tasks,
    structural_signature,
    task_content,
    task_observation,
    task_to_sample,
    training_benchmark_protocol,
)


def _task(task_id: str, color: int = 1) -> dict:
    return normalize_task({"train": [{"input": [[0, color], [color, 0]], "output": [[color, 0], [0, color]]}], "test": [{"input": [[color, color], [0, 0]], "output": [[0, 0], [color, color]]}]}, source_id=task_id, require_test_outputs=True)


def _canonical(items):
    return [{"task": task, "canonical_content_sha256": canonical_hash(task_content(task)), "observation_sha256": canonical_hash(task_observation(task)), "train_pair_sha256": canonical_hash(task_observation(task)["train"]), "structural_signature_sha256": structural_signature(task)[0]} for task in items]


def test_canonical_task_hash_is_stable():
    assert canonical_hash(task_content(_task("aaaaaaaa"))) == canonical_hash(task_content(_task("bbbbbbbb")))


def test_id_blacklist_exclusion():
    kept, report = exclude_leakage(_canonical([_task("aaaaaaaa")]), blacklist_entries=[{"task_id": "aaaaaaaa"}], evaluation_tasks=[])
    assert not kept and report["excluded_by_id"] == 1


def test_hash_blacklist_exclusion():
    used = _task("aaaaaaaa")
    candidate = _task("bbbbbbbb")
    kept, report = exclude_leakage(_canonical([candidate]), blacklist_entries=[{"task_id": "aaaaaaaa"}], evaluation_tasks=[used])
    assert not kept and report["excluded_by_exact_hash"] == 1


def test_geometric_equivalence_signature():
    first = _task("aaaaaaaa", color=1)
    second = _task("bbbbbbbb", color=7)
    assert structural_signature(first)[0] == structural_signature(second)[0]


def test_deterministic_parallel_canonicalization():
    tasks = [_task(f"{index:08x}", color=(index % 8) + 1) for index in range(CPU_WORKERS)]
    first, _ = parallel_canonicalize(tasks, workers=CPU_WORKERS)
    second, _ = parallel_canonicalize(tasks, workers=CPU_WORKERS)
    assert [(x["task"]["source_id"], x["canonical_content_sha256"]) for x in first] == [(x["task"]["source_id"], x["canonical_content_sha256"]) for x in second]


def test_puzzle_split_keeps_structural_family_together():
    tasks = []
    for index in range(16):
        width = index + 1
        color = (index % 9) + 1
        tasks.append(normalize_task({"train": [{"input": [[color] * width], "output": [[0] * width]}], "test": [{"input": [[0] * width], "output": [[color] * width]}]}, source_id=f"{index:08x}", require_test_outputs=True))
    items = _canonical(tasks)
    train, validation, manifest = split_tasks(items, seed=8, validation_fraction=.4)
    assert manifest["family_overlap_count"] == 0
    assert {row["structural_signature_sha256"] for row in train}.isdisjoint({row["structural_signature_sha256"] for row in validation})


def test_native_serialization_and_assistant_masking():
    messages = [{"role": "user", "content": "01\n23"}, {"role": "assistant", "content": "32\n10"}]
    text, ids, labels = native_tokenize_with_labels(messages)
    assert text == "<|im_start|>user\n01\n23<|im_end|><|im_start|>assistant\n32\n10<|im_end|>"
    assert ids[0] == EXPECTED_TOKENS["<|im_start|>"] and ids[-1] == 15
    assert all(value == IGNORE_INDEX for value in labels[: len(ids) // 2])
    assert any(value != IGNORE_INDEX for value in labels)


def test_collator_padding_and_supervision():
    one, two = task_to_sample(_task("aaaaaaaa")), task_to_sample(normalize_task({"train": [{"input": [[1]], "output": [[2]]}], "test": [{"input": [[2]], "output": [[3]]}]}, source_id="bbbbbbbb", require_test_outputs=True))
    batch = collate([one, two])
    assert len(batch["input_ids"][0]) == len(batch["input_ids"][1])
    assert all(any(value != IGNORE_INDEX for value in labels) for labels in batch["labels"])


def test_shard_hash_reproducibility(tmp_path: Path):
    payload = {"a": 1, "b": [2, 3]}
    first = canonical_hash(payload)
    (tmp_path / "x.json").write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    second = canonical_hash(json.loads((tmp_path / "x.json").read_text(encoding="utf-8")))
    assert first == second


def test_internal_dedup_maps_duplicate():
    first, second = _canonical([_task("aaaaaaaa"), _task("bbbbbbbb")])
    kept, report = deduplicate([first, second])
    assert len(kept) == 1 and report["exact_duplicate_count_removed"] == 1


def test_benchmark_protocol_is_explicit_when_no_eligible_tasks_remain():
    protocol = training_benchmark_protocol(tokenizer_summary={"status": "NOT_RUN_NO_ELIGIBLE_TASKS"})
    assert protocol["status"] == "PREPARED_NOT_RUN"
    assert protocol["protocol"]["training_started"] is False
