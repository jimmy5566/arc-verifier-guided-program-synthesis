from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from training_data_v2.pipeline import (
    Descriptor,
    IGNORE_INDEX,
    ROLE_HARD_EXCLUDE,
    ROLE_NOVEL,
    ROLE_QUARANTINE,
    ROLE_REPLAY,
    RECORD_REARC_PAIR_BANK,
    _read_rearc_pair_bank,
    _stable_seed,
    _process_descriptors,
    _group_duplicate_counts,
    assign_final_role,
    canonical_hash,
    native_tokenize_with_labels,
    normalize_task,
    pair_bank_signatures,
    signatures,
    validate_family_split,
)


def _task() -> dict:
    return normalize_task(
        {
            "train": [{"input": [[1, 0], [0, 0]], "output": [[0, 1], [0, 0]]}],
            "test": [{"input": [[2, 0], [0, 0]], "output": [[0, 2], [0, 0]]}],
        },
        source_native_id="fixture",
    )


def test_frozen_filename_does_not_imply_heldout() -> None:
    # Role assignment intentionally has no filename input: semantic evidence is required.
    assert assign_final_role(source="official_arc_agi_2", subset="training", sft139_training_seen=True) == ROLE_REPLAY


def test_official_training_is_replay_and_evaluation_is_hard_excluded() -> None:
    assert assign_final_role(source="official_arc_agi_2", subset="training", sft139_training_seen=True) == ROLE_REPLAY
    assert assign_final_role(source="official_arc_agi_2", subset="evaluation", sft139_training_seen=False) == ROLE_HARD_EXCLUDE


def test_explicit_future_gate_remains_hard_excluded() -> None:
    assert assign_final_role(source="licensed_new_source", subset="train", sft139_training_seen=False, explicit_future_reserved=True) == ROLE_HARD_EXCLUDE


def test_sft139_replay_and_unknown_quarantine() -> None:
    assert assign_final_role(source="miniarc", subset="MiniARC", sft139_training_seen=True) == ROLE_REPLAY
    assert assign_final_role(source="unknown", subset="unknown", sft139_training_seen=False, provenance_resolved=False) == ROLE_QUARANTINE
    assert assign_final_role(source="licensed_new", subset="train", sft139_training_seen=False) == ROLE_NOVEL


def test_d4_color_equivalent_duplicate_detection() -> None:
    base = _task()
    recolored = normalize_task(
        {
            "train": [{"input": [[4, 0], [0, 0]], "output": [[0, 4], [0, 0]]}],
            "test": [{"input": [[9, 0], [0, 0]], "output": [[0, 9], [0, 0]]}],
        },
        source_native_id="recolored",
    )
    assert signatures(base)["d4_color_signature_sha256"] == signatures(recolored)["d4_color_signature_sha256"]
    records = [
        {"source": "a", "source_native_id": "one", **signatures(base)},
        {"source": "b", "source_native_id": "two", **signatures(recolored)},
    ]
    count, groups = _group_duplicate_counts(records, "d4_color_signature_sha256")
    assert count == 1
    assert len(groups) == 1


def test_d4_and_color_signatures_are_independently_stable() -> None:
    base = _task()
    rotated = normalize_task(
        {
            "train": [{"input": [[0, 1], [0, 0]], "output": [[0, 0], [0, 1]]}],
            "test": [{"input": [[0, 2], [0, 0]], "output": [[0, 0], [0, 2]]}],
        },
        source_native_id="rotated",
    )
    recolored = normalize_task(
        {
            "train": [{"input": [[9, 0], [0, 0]], "output": [[0, 9], [0, 0]]}],
            "test": [{"input": [[7, 0], [0, 0]], "output": [[0, 7], [0, 0]]}],
        },
        source_native_id="recolored",
    )
    assert signatures(base)["d4_signature_sha256"] == signatures(rotated)["d4_signature_sha256"]
    assert signatures(base)["color_signature_sha256"] == signatures(recolored)["color_signature_sha256"]
    assert signatures(base)["d4_color_signature_sha256"] == signatures(recolored)["d4_color_signature_sha256"]


def test_native_labels_and_message_boundaries() -> None:
    ids, labels = native_tokenize_with_labels(
        [
            {"role": "user", "content": "12\n34"},
            {"role": "assistant", "content": "43\n21"},
        ]
    )
    assert len(ids) == len(labels)
    assert any(item == IGNORE_INDEX for item in labels)
    assert any(item != IGNORE_INDEX for item in labels)
    assert ids[:3] == [14, 11, 10]
    assert labels[:3] == [IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX]


def test_registry_hash_inputs_are_deterministic() -> None:
    assert signatures(_task()) == signatures(_task())
    assert canonical_hash({"shard": ["a", "b"], "policy": "family-first"}) == canonical_hash({"policy": "family-first", "shard": ["a", "b"]})


def test_family_split_integrity_rejects_augmented_cross_split_leakage() -> None:
    assert validate_family_split({"train": ["rearc:abc"], "validation": [], "holdout": ["official:def"]})["status"] == "PASS"
    leaked = validate_family_split({"train": ["rearc:abc"], "validation": ["rearc:abc"], "holdout": []})
    assert leaked["status"] == "FAIL"
    assert leaked["cross_split_overlaps"] == {"rearc:abc": ["train", "validation"]}


def test_rearc_pair_bank_is_not_coerced_to_arc_train_test() -> None:
    pairs = [
        {"input": [[index % 10]], "output": [[(index + 1) % 10]]}
        for index in range(7)
    ]
    result = pair_bank_signatures(pairs)
    assert set(result) == {
        "canonical_observation_sha256",
        "full_content_sha256",
        "train_pair_sha256",
        "d4_signature_sha256",
        "color_signature_sha256",
        "d4_color_signature_sha256",
    }


def test_rearc_adapter_records_invalid_pair_validation_without_silent_skip(tmp_path: Path) -> None:
    descriptor = Descriptor("rearc", "v1", "tasks", "fixture", "rearc:fixture", str(tmp_path / "unused.zip"), "re_arc/tasks/fixture.json", ROLE_REPLAY, "BASE_MODEL_TRAIN_SEEN", ("BASE_MODEL_TRAIN_SEEN",), RECORD_REARC_PAIR_BANK)
    raw = [{"input": [[index % 10]], "output": [[(index + 1) % 10]]} for index in range(7)]
    raw.append({"input": [[0] for _ in range(31)], "output": [[0]]})
    record = _read_rearc_pair_bank(descriptor, raw)
    assert record["record_kind"] == RECORD_REARC_PAIR_BANK
    assert record["pair_bank_observed_count"] == 8
    assert record["pair_bank_valid_count"] == 7
    assert record["pair_bank_invalid_count"] == 1
    assert len(record["samples"]) == 1
    assert 0 <= _stable_seed("rearc", "fixture", "recovery-v2") < 2**63


def test_holdout_cannot_enter_training_sampler_contract() -> None:
    train_role = assign_final_role(source="official_arc_agi_2", subset="training", sft139_training_seen=True)
    holdout_role = assign_final_role(source="official_arc_agi_2", subset="evaluation", sft139_training_seen=False)
    assert train_role == ROLE_REPLAY
    assert holdout_role == ROLE_HARD_EXCLUDE
    assert holdout_role != train_role


def test_numpy_available_for_percentile_audit() -> None:
    # Explicitly guards the requested NumPy-backed path rather than letting it be skipped.
    import numpy as np

    assert float(np.percentile(np.array([1, 2, 3]), 50)) == 2.0


def test_multiprocessing_registry_order_is_deterministic(tmp_path: Path) -> None:
    raw = {
        "train": [{"input": [[1]], "output": [[2]]}],
        "test": [{"input": [[3]], "output": [[4]]}],
    }
    first, second = tmp_path / "b.json", tmp_path / "a.json"
    first.write_text(json.dumps(raw), encoding="utf-8")
    second.write_text(json.dumps(raw), encoding="utf-8")
    descriptors = [
        Descriptor("fixture", "v1", "train", "b", "fixture:b", str(first), None, ROLE_REPLAY, "BASE_MODEL_TRAIN_SEEN", ("BASE_MODEL_TRAIN_SEEN",)),
        Descriptor("fixture", "v1", "train", "a", "fixture:a", str(second), None, ROLE_REPLAY, "BASE_MODEL_TRAIN_SEEN", ("BASE_MODEL_TRAIN_SEEN",)),
    ]
    one = _process_descriptors(descriptors, workers=2)
    two = _process_descriptors(list(reversed(descriptors)), workers=2)
    assert [(item["source_native_id"], item["full_content_sha256"]) for item in one] == [(item["source_native_id"], item["full_content_sha256"]) for item in two]


@pytest.mark.integration
def test_actual_sft139_tokenizer_identity_when_assets_are_supplied() -> None:
    model_root = os.environ.get("ARC2_SFT139_MODEL_DIR")
    if not model_root or not Path(model_root).exists():
        pytest.skip("CPU integration asset is intentionally untracked; the pipeline gate runs this check when supplied")
    from transformers import AutoTokenizer

    vocab = AutoTokenizer.from_pretrained(model_root, local_files_only=True, trust_remote_code=False).get_vocab()
    assert {str(index): vocab[str(index)] for index in range(10)} == {str(index): index for index in range(10)}
    assert {"Ċ": vocab["Ċ"], "user": vocab["user"], "assistant": vocab["assistant"], "<|im_start|>": vocab["<|im_start|>"], "<|im_end|>": vocab["<|im_end|>"]} == {"Ċ": 10, "user": 11, "assistant": 12, "<|im_start|>": 14, "<|im_end|>": 15}
