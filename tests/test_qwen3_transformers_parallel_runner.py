from __future__ import annotations

from pathlib import Path

import pytest

from inference.kaggle_l4_parallel_runner import CheckpointIdentity
from inference.qwen3_transformers_parallel_runner import MODEL_LOAD_WATCHDOG_SECONDS, Qwen3TransformersParallelRunner, _generation_config, warm_model_safetensors


IDENTITY = CheckpointIdentity("LLM_PROGRAM_SYNTHESIS_V2", "qwen-lm/qwen-3/Transformers/8b/1", "modelhash", "configsha", "prompt.1", "schema.1", "symbolic")


def test_qwen_runner_uses_deterministic_task_shards_and_only_valid_worker_counts(tmp_path: Path):
    runner = Qwen3TransformersParallelRunner(identity=IDENTITY, checkpoint_root=tmp_path, worker_count=4)
    assignments = runner.assignments(("a", "b", "c", "d", "e"))
    assert [(item.worker_id, item.gpu_id, item.task_ids) for item in assignments] == [
        (0, 0, ("a", "e")), (1, 1, ("b",)), (2, 2, ("c",)), (3, 3, ("d",))
    ]
    with pytest.raises(ValueError, match="1, 2, or 4"):
        Qwen3TransformersParallelRunner(identity=IDENTITY, checkpoint_root=tmp_path, worker_count=3)


def test_qwen_generation_config_preserves_the_frozen_v2_semantics():
    frozen = {
        "model": {"model_source": "qwen-lm/qwen-3/Transformers/8b/1", "context_window": 12288},
        "sampling": {"temperature": 0, "top_p": 1, "seed": 0},
        "generation": {"candidate_budget": 5, "max_output_tokens": 800},
        "macro_dsl": {"prompt_version": "llm_program_synthesis_v2.macro_prompt.1"},
    }
    config = _generation_config(frozen)
    assert (config.model, config.temperature, config.top_p, config.seed) == ("qwen-lm/qwen-3/Transformers/8b/1", 0, 1, 0)
    assert (config.hypothesis_budget, config.max_output_tokens, config.context_window) == (5, 800, 12288)


def test_model_warmup_reads_all_safetensors_in_stable_order(tmp_path: Path):
    (tmp_path / "z.safetensors").write_bytes(b"z" * 3)
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "a.safetensors").write_bytes(b"abcd")
    report = warm_model_safetensors(tmp_path, chunk_bytes=2)
    assert report["shard_count"] == 2
    assert report["bytes_read"] == 7
    assert report["seconds"] >= 0


def test_model_warmup_rejects_missing_artifact_and_watchdog_is_five_minutes(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="no safetensors"):
        warm_model_safetensors(tmp_path)
    assert MODEL_LOAD_WATCHDOG_SECONDS == 300.0
