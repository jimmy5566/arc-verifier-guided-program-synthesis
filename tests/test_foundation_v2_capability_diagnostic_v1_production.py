from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def module():
    path = ROOT / "scripts/run_foundation_v2_capability_diagnostic_v1_production.py"
    spec = importlib.util.spec_from_file_location("production_diagnostic", path)
    assert spec and spec.loader
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def test_production_path_has_no_forensic_or_calibration_generation() -> None:
    loaded = module()
    source = (ROOT / "scripts/run_foundation_v2_capability_diagnostic_v1_production.py").read_text(encoding="utf-8")
    assert loaded.PROTOCOL_ID == "MAX_BATCH32_TOKEN_CAP21568"
    assert 'run("calibration")' not in source
    assert "LogitsProcessor" not in source
    assert "torch.topk" not in source
    assert "output_scores" not in source
    assert "return_dict_in_generate" not in source
    assert "hidden_states" not in source
    assert "attentions" not in source
    assert "duplicate_self" not in source
    assert "batch ladder" not in source.lower()


def test_generation_has_only_state_level_synchronization_and_fail_closed_oom() -> None:
    source = (ROOT / "scripts/run_foundation_v2_capability_diagnostic_v1_production.py").read_text(encoding="utf-8")
    assert source.count("torch.cuda.synchronize(0)") == 2
    assert "PRODUCTION_BATCH_OOM" in source
    assert "count//2" not in source and "count // 2" not in source
    assert "gc.collect()" in source and "torch.cuda.empty_cache()" in source


def test_partial_jsonl_append_and_duplicate_rejection(tmp_path: Path) -> None:
    loaded = module()
    path = tmp_path / "base.partial.jsonl"
    loaded.append_records(path, [{"sample_id": "a", "value": 1}, {"sample_id": "b", "value": 2}])
    assert loaded.load_partial(path) == {"a": {"sample_id": "a", "value": 1}, "b": {"sample_id": "b", "value": 2}}
    loaded.append_records(path, [{"sample_id": "a", "value": 3}])
    with pytest.raises(loaded.DiagnosticError, match="DUPLICATE_PARTIAL_SAMPLE"):
        loaded.load_partial(path)


def test_scheduler_is_frozen_and_deterministic() -> None:
    loaded = module()
    rows = [
        {"sample_id": "z", "prompt_ids": [1] * 1000},
        {"sample_id": "a", "prompt_ids": [1] * 1000},
        {"sample_id": "b", "prompt_ids": [1] * 11000},
    ]
    groups = loaded.production_batches(rows)
    assert [[row["sample_id"] for row in group] for group in groups] == [["a", "z"], ["b"]]
    assert all(max(len(row["prompt_ids"]) for row in group) * len(group) <= loaded.TOKEN_CAP for group in groups)


def test_borderline_semantics() -> None:
    from foundation_v2_capability_diagnostic_v1.audit import pair_models

    rows = [
        {"model_state": "base", "capability": "x", "measurement_type": "DIRECT_ATOMIC", "exact_accuracy": 0.0},
        {"model_state": "foundation_v2", "capability": "x", "measurement_type": "DIRECT_ATOMIC", "exact_accuracy": 0.375},
    ]
    result = pair_models(rows, "exact_accuracy")
    assert result[0]["borderline"] is True
    assert result[0]["borderline_observation_width"] == 1 / 32
