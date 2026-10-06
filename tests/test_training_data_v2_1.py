from __future__ import annotations

import json
import os
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from scripts.audit_training_data_v2_1 import (
    IGNORE_INDEX,
    _api_tokenize_text,
    _duplicate_report,
    _init_tokenizer,
    _turns,
)


def test_complete_turn_parser_rejects_uncovered_text() -> None:
    valid = "<|im_start|>user\n01<|im_end|><|im_start|>assistant\n10<|im_end|>"
    assert [role for role, _ in _turns(valid)] == ["user", "assistant"]
    with pytest.raises(ValueError):
        _turns(valid + "trailing")


@pytest.mark.integration
def test_actual_tokenizer_api_and_assistant_masking() -> None:
    model = os.environ.get("ARC2_SFT139_MODEL_DIR")
    if not model:
        pytest.skip("ARC2_SFT139_MODEL_DIR is not supplied")
    _init_tokenizer(model)
    text = "<|im_start|>user\n01<|im_end|><|im_start|>assistant\n10<|im_end|>"
    ids, labels, details = _api_tokenize_text(text)
    assert ids == [14, 10, 0, 1, 15, 14, 10, 1, 0, 15]
    boundary = details["turns"][0]["end"]
    assert labels[:boundary] == [IGNORE_INDEX] * boundary
    assert labels[boundary:] == ids[boundary:]
    assert ids[-1] == details["eos_token_id"] == 15


@pytest.mark.integration
def test_v2_1_registry_contains_replay_only() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "data/processed/arc_training_v2_1/sample_registry.parquet"
    if not path.exists():
        pytest.skip("V2.1 local materialization is not present")
    table = pq.read_table(path, columns=["sample_id", "final_training_role"])
    assert table.num_rows == 62650
    assert table.column("sample_id").null_count == 0
    assert set(table.column("final_training_role").to_pylist()) == {"TRAIN_ELIGIBLE_REPLAY"}


@pytest.mark.integration
def test_train_pair_duplicate_is_replay_boundary_safe() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "data/processed/arc_training_v2/puzzle_registry.parquet"
    if not path.exists():
        pytest.skip("V2 puzzle registry is not present")
    report = _duplicate_report(path)
    assert report["status"] == "PASS"
    assert report["group_count"] == 1
    assert report["duplicate_groups"][0]["crosses_replay_hard_exclude_boundary"] is False


def test_gate_is_derived_from_checks() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "artifacts/training_data_v2_1/GPU_TRAINING_GATE_V2_1.json"
    if not path.exists():
        pytest.skip("V2.1 audit artifacts are not present")
    gate = json.loads(path.read_text(encoding="utf-8"))
    assert gate["DATA_ENGINEERING_READY"] == all(gate["checks"].values())
    assert gate["SCIENTIFIC_TRAINING_READY"] is False
    assert gate["GPU_TRAINING_STARTED"] is False
