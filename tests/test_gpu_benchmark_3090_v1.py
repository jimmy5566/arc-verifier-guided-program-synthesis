from __future__ import annotations

from pathlib import Path

import pytest

from gpu_benchmark_3090_v1.benchmark import (
    BenchmarkGateError,
    assert_lora_partition,
    derive_gate,
    make_family_schedule,
    require_single_cuda_device,
    runtime_extrapolation,
    truncate_example,
    validate_model_package,
)


class Param:
    def __init__(self, count: int, trainable: bool):
        self.count = count
        self.requires_grad = trainable

    def numel(self):
        return self.count


def test_model_package_requires_index_and_all_shards(tmp_path: Path) -> None:
    for name in ("config.json", "tokenizer_config.json", "tokenizer.json"):
        (tmp_path / name).write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors.index.json").write_text('{"weight_map":{"x":"one.safetensors"}}', encoding="utf-8")
    with pytest.raises(BenchmarkGateError):
        validate_model_package(tmp_path)
    (tmp_path / "one.safetensors").write_bytes(b"x")
    assert validate_model_package(tmp_path)["safetensor_shards"] == ["one.safetensors"]


def test_exact_single_3090_requirement() -> None:
    require_single_cuda_device(1, "NVIDIA GeForce RTX 3090")
    with pytest.raises(BenchmarkGateError):
        require_single_cuda_device(2, "NVIDIA GeForce RTX 3090")


def test_only_lora_parameters_trainable() -> None:
    result = assert_lora_partition([("base.weight", Param(10, False)), ("layer.lora_A.weight", Param(2, True))])
    assert result["base_trainable_parameters"] == 0
    with pytest.raises(BenchmarkGateError):
        assert_lora_partition([("base.weight", Param(10, True))])


def _rows(prefix: str, role: str = "TRAIN_ELIGIBLE"):
    return [
        {"sample_id": f"{prefix}-1", "source": prefix, "generator_family": f"{prefix}:a", "sequence_length": 10, "supervised_token_count": 4, "final_training_role": role, "split": "train"},
        {"sample_id": f"{prefix}-2", "source": prefix, "generator_family": f"{prefix}:b", "sequence_length": 10, "supervised_token_count": 4, "final_training_role": role, "split": "train"},
    ]


def test_schedule_is_deterministic_and_excludes_holdout() -> None:
    novel = _rows("novel") + _rows("holdout", "HOLDOUT")
    first = make_family_schedule(novel, _rows("replay"), 100, seed=7)
    second = make_family_schedule(novel, _rows("replay"), 100, seed=7)
    assert first == second
    assert not any(row["source"] == "holdout" for row in first)
    assert sum(row["pool"] == "POOL_NOVEL_V1_1" for row in first) == 75


def test_context_truncation_accounting_and_reset_inputs() -> None:
    ids = list(range(8512))
    labels = [-100] * 4096 + list(range(4416))
    a = truncate_example(ids, labels, 8192)
    b = truncate_example(ids, labels, 8704)
    assert a["truncated"] and a["supervised_tokens_lost"] == 320
    assert not b["truncated"] and b["supervised_tokens_lost"] == 0
    assert a["input_ids"] is not b["input_ids"]


def test_runtime_extrapolation_math() -> None:
    assert runtime_extrapolation(1000)["2000000"]["seconds"] == 2000


def test_gate_derivation() -> None:
    checks = {"data_access_policy": True, "infrastructure": True, "numerical": True}
    assert derive_gate(checks, {"status": "PASS"}, {"status": "PASS", "vram_safety": "SAFE"}) == "PASS_8704_RECOMMENDED"
    assert derive_gate(checks, {"status": "PASS"}, {"status": "FAIL", "failure_class": "OOM"}) == "PASS_8192_ONLY"
    assert derive_gate({**checks, "data_access_policy": False}, None, None) == "FAIL_DATA_ACCESS_POLICY"
    assert derive_gate({**checks, "infrastructure": False}, {"status": "PASS"}, None) == "FAIL_INFRASTRUCTURE"
    assert derive_gate({**checks, "numerical": False}, {"status": "PASS"}, None) == "FAIL_NUMERICAL"
