from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from inference.kaggle_l4_parallel_runner import (
    CheckpointIdentity,
    RunnerStatus,
    TaskCheckpointStore,
    atomic_write_json,
    import_legacy_aggregate,
    inspect_hardware,
    locate_offline_model,
    materialize_offline_ollama,
    merge_peak_observation,
    merge_task_checkpoints,
    summarize_worker_timings,
    stable_round_robin,
    _retryable_model_result,
)


IDENTITY = CheckpointIdentity("LLM_PROGRAM_SYNTHESIS_V2", "qwen3:14b", "modelhash", "frozenhash", "prompt.1", "schema.1", "symbolic")


def l4_command(command, **_):
    if "cuda_version" in command[1]:
        return "12.8\n"
    return "NVIDIA L4, 23034, 570.00\nNVIDIA L4, 23034, 570.00\nNVIDIA L4, 23034, 570.00\nNVIDIA L4, 23034, 570.00\n"


def test_hardware_gate_requires_exactly_four_l4_gpus():
    report = inspect_hardware(l4_command)
    assert report.status is RunnerStatus.SUCCESS
    assert [gpu.index for gpu in report.gpus] == [0, 1, 2, 3]
    invalid = inspect_hardware(lambda *_, **__: "NVIDIA Tesla T4, 15360, 555\n")
    assert invalid.status is RunnerStatus.INVALID_GPU_RUNTIME


def test_deterministic_sharding_has_no_duplicate_task_assignment():
    assignments = stable_round_robin(tuple("abcdefgh"), 4)
    assert [assignment.gpu_id for assignment in assignments] == [0, 1, 2, 3]
    assert [assignment.task_ids for assignment in assignments] == [("a", "e"), ("b", "f"), ("c", "g"), ("d", "h")]
    assert sorted(task for assignment in assignments for task in assignment.task_ids) == list("abcdefgh")
    with pytest.raises(ValueError, match="unique"):
        stable_round_robin(("a", "a"), 4)


def test_atomic_checkpoint_resume_and_config_mismatch(tmp_path: Path):
    store = TaskCheckpointStore(tmp_path / "checkpoints", IDENTITY)
    store.write("abc", {"status": "SUCCESS", "worker_id": 0, "gpu_id": 0, "total_task_seconds": 1.0})
    compatible, record, reason = store.compatible("abc")
    assert compatible and record and reason is None
    payload = json.loads(store.path_for("abc").read_text(encoding="utf-8"))
    assert payload["identity"] == IDENTITY.to_dict()
    changed = TaskCheckpointStore(tmp_path / "checkpoints", CheckpointIdentity(**(IDENTITY.to_dict() | {"experiment_config_sha256": "other"})))
    assert changed.compatible("abc")[2] == RunnerStatus.CHECKPOINT_CONFIG_MISMATCH.value
    assert not list((tmp_path / "checkpoints").glob("*.tmp"))


def test_merge_detects_missing_and_merges_complete_worker_records(tmp_path: Path):
    store = TaskCheckpointStore(tmp_path, IDENTITY)
    store.write("a", {"status": "SUCCESS", "worker_id": 0, "gpu_id": 0, "total_task_seconds": 1.0})
    with pytest.raises(RuntimeError, match="incomplete"):
        merge_task_checkpoints(store, ("a", "b"))
    store.write("b", {"status": "SUCCESS", "worker_id": 1, "gpu_id": 1, "total_task_seconds": 2.0})
    merged = merge_task_checkpoints(store, ("a", "b"))
    assert merged["duplicate_task_count"] == 0
    assert merged["worker_task_counts"] == {"0": 1, "1": 1}


def test_one_task_failure_is_checkpointed_without_blocking_other_tasks(tmp_path: Path):
    store = TaskCheckpointStore(tmp_path, IDENTITY)
    store.write("good", {"status": "SUCCESS", "worker_id": 0, "gpu_id": 0, "total_task_seconds": 1.0})
    store.write("bad", {"status": "EXECUTION_ERROR", "worker_id": 1, "gpu_id": 1, "reason": "isolated failure", "total_task_seconds": None})
    merged = merge_task_checkpoints(store, ("good", "bad"))
    assert merged["records"]["bad"]["status"] == "EXECUTION_ERROR"
    assert merged["records"]["good"]["status"] == "SUCCESS"


def test_compatible_legacy_aggregate_imports_once(tmp_path: Path):
    legacy = tmp_path / "legacy.json"
    atomic_write_json(legacy, {"frozen_config_sha256": "frozenhash", "records": {"abc": {"candidate_results": [], "prediction": None}}})
    store = TaskCheckpointStore(tmp_path / "tasks", IDENTITY)
    assert import_legacy_aggregate(legacy, store) == ["abc"]
    assert store.compatible("abc")[0]
    assert import_legacy_aggregate(legacy, store) == []
    incompatible = tmp_path / "incompatible.json"
    atomic_write_json(incompatible, {"frozen_config_sha256": "different", "records": {}})
    with pytest.raises(RuntimeError, match="CHECKPOINT_CONFIG_MISMATCH"):
        import_legacy_aggregate(incompatible, store)


def test_offline_artifact_requires_gguf_modelfile_and_executable_ollama(tmp_path: Path):
    model_root = tmp_path / "qwen"
    model_root.mkdir()
    gguf = model_root / "qwen3-14b.Q4_K_M.gguf"; gguf.write_bytes(b"gguf")
    modelfile = model_root / "Modelfile"; modelfile.write_text("FROM ./qwen3-14b.Q4_K_M.gguf\n", encoding="utf-8")
    binary = model_root / "ollama"; binary.write_text("binary", encoding="utf-8"); os.chmod(binary, 0o755)
    artifact = locate_offline_model(model_root)
    assert artifact.status is RunnerStatus.SUCCESS
    assert artifact.tokenizer == "embedded_in_gguf"
    assert artifact.quantization == "Q4_K_M"
    materialized = materialize_offline_ollama(artifact, tmp_path / "working")
    assert Path(materialized.ollama_binary_path).is_file()
    assert os.access(materialized.ollama_binary_path, os.X_OK)
    binary.unlink()
    assert locate_offline_model(model_root).status is RunnerStatus.MODEL_ARTIFACT_MISSING


def test_only_transport_failures_are_eligible_for_one_retry():
    assert _retryable_model_result({"candidate_results": [{"status": "PROVIDER_FAILED"}]})
    assert _retryable_model_result({"candidate_results": [{"status": "TIMEOUT"}]})
    assert not _retryable_model_result({"candidate_results": [{"status": "TRAIN_INCONSISTENT"}]})
    assert not _retryable_model_result({"candidate_results": [{"status": "PROVIDER_FAILED"}, {"status": "EXECUTION_ERROR"}]})


def test_peak_gpu_observation_keeps_largest_valid_values():
    peak = merge_peak_observation({"gpu_utilization_percent": 20, "vram_used_mib": 100, "vram_total_mib": 23034}, {"gpu_utilization_percent": 80, "vram_used_mib": 90, "vram_total_mib": 23034})
    assert peak == {"gpu_utilization_percent": 80, "vram_used_mib": 100, "vram_total_mib": 23034}


def test_worker_timing_p90_uses_nearest_rank_in_small_parallel_shards():
    summary = summarize_worker_timings(({"total_task_seconds": 10.0}, {"total_task_seconds": 40.0}))
    assert summary["median_task_seconds"] == 25.0
    assert summary["p90_task_seconds"] == 40.0
