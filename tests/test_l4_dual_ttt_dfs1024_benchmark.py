from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts.build_l4_dual_ttt_dfs1024_benchmark_notebook import (
    BENCHMARK_ID,
    kernel_metadata,
    notebook_source,
)
from scripts.run_l4_dual_ttt_dfs1024_benchmark import (
    compact_cell,
    decoder_from_config,
    projections,
    validate_cohort,
    validate_config,
)


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "l4_dual_ttt_dfs1024_bench_v1"


def load(name: str) -> dict:
    return json.loads((EXPERIMENT / name).read_text(encoding="utf-8"))


def test_frozen_config_is_scalar_dfs1024_with_rerun_off() -> None:
    config = load("benchmark_config.json")
    validate_config(config)
    decoder = decoder_from_config(config)
    assert decoder.max_expanded_nodes == 1024
    assert decoder.max_completed_candidates == 32
    assert decoder.frontier_floor == 1
    assert decoder.independent_lane_budgets is False
    assert config["rerun"] == {
        "enabled": False,
        "auto_rerun": False,
        "retry_failed_task": False,
        "resume_completed": False,
    }
    assert config["augmentation_manifest"] == {
        "TTT24": [
            {"geometry": "flip_lr", "color_offset": 0, "pair_order": "canonical"},
            {"geometry": "flip_ud", "color_offset": 0, "pair_order": "canonical"},
            {"geometry": "transpose", "color_offset": 0, "pair_order": "canonical"},
            {"geometry": "anti_transpose", "color_offset": 0, "pair_order": "canonical"},
        ],
        "TTT48": [
            {"geometry": "identity", "color_offset": 0, "pair_order": "canonical"},
            {"geometry": "rot90", "color_offset": 0, "pair_order": "canonical"},
            {"geometry": "flip_ud", "color_offset": 0, "pair_order": "canonical"},
            {"geometry": "anti_transpose", "color_offset": 0, "pair_order": "canonical"},
        ],
    }


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("rerun", "retry_failed_task"), True),
        (("search", "max_expanded_nodes"), 2048),
        (("search", "lane_count"), 2),
        (("search", "batch2_cross_cell"), True),
        (("search", "batch4_regret"), True),
    ],
)
def test_contract_rejects_drift(path: tuple[str, str], value: object) -> None:
    config = copy.deepcopy(load("benchmark_config.json"))
    config[path[0]][path[1]] = value
    with pytest.raises(ValueError):
        validate_config(config)


def test_cohort_is_frozen_target_blind_and_counts_multi_output(tmp_path: Path) -> None:
    config = copy.deepcopy(load("benchmark_config.json"))
    cohort = load("BENCHMARK_TASK_IDS.json")
    assert len(cohort["task_ids"]) == 8
    assert cohort["solutions_accessed_for_selection"] is False
    expected = hashlib.sha256(
        json.dumps(cohort["task_ids"], separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert cohort["task_ids_canonical_sha256"] == expected
    challenges = {
        task_id: {
            "train": [{"input": [[0]], "output": [[0]]}],
            "test": [{"input": [[0]]}] * (2 if index == 0 else 1),
        }
        for index, task_id in enumerate(cohort["task_ids"])
    }
    challenge_path = tmp_path / "challenge.json"
    challenge_path.write_text(json.dumps(challenges), encoding="utf-8")
    config["challenge_sha256"] = hashlib.sha256(challenge_path.read_bytes()).hexdigest()
    result = validate_cohort(
        challenges,
        cohort,
        config,
        challenge_file_sha256=config["challenge_sha256"],
    )
    assert result == {"task_ids": cohort["task_ids"], "num_tasks": 8, "num_test_outputs": 9, "num_dfs_cells": 72}


def test_compact_cell_keeps_scalar_telemetry_and_empty_pool_visible() -> None:
    row = {
        "runtime_seconds": 2.0,
        "nodes_expanded": 1024,
        "model_forwards": 100,
        "model_forward_seconds": 1.5,
        "task_id": "task",
        "output_index": 2,
        "depth": 48,
        "view": "rot90",
        "tokens_advanced": 300,
        "complete_candidate_count": 0,
        "unique_grid_count": 0,
        "candidate_count": 4,
        "valid_grid_count": 0,
        "termination_reason": "node_budget",
        "budget_exhausted": True,
        "search_exhausted": False,
        "prompt_tokens": 123,
        "peak_vram_mb": 7000,
    }
    result = compact_cell(row, gpu_id=3, worker_id=3, candidate_cap=32)
    assert result["nodes_per_second"] == 512.0
    assert result["model_forward_sec_per_node"] == pytest.approx(1.5 / 1024)
    assert result["unique_candidates"] == 0
    assert result["invalid_candidates"] == 4
    assert result["budget_exhausted"] is True
    assert result["candidate_cap_reached"] is False


def test_projections_keep_task_and_output_denominators_separate(tmp_path: Path) -> None:
    config = copy.deepcopy(load("benchmark_config.json"))
    config["projection"]["bootstrap_samples"] = 10
    rows = [{"task_wall_s": value} for value in (10, 20, 30, 40, 50, 60, 70, 80)]
    result = projections(
        task_rows=rows,
        workload_wall=120.0,
        num_tasks=8,
        num_outputs=11,
        challenge=tmp_path / "arc-agi_evaluation_challenges.json",
        config=config,
    )
    assert result["60"]["task_based_seconds"] == 900.0
    assert result["60"]["output_based_seconds"] == pytest.approx(120.0 * 89 / 11)
    assert result["120"]["output_based_seconds"] is None
    assert result["confidence"] == "MEDIUM"


def test_notebook_is_one_shot_target_blind_and_calls_real_runner() -> None:
    source = notebook_source("jimmy5566", "arc2-l4-dual-ttt-dfs1024-bench-v1-source")
    lowered = source.lower()
    assert BENCHMARK_ID in source
    assert "RERUN_ENABLED = FALSE" in source
    assert "run_l4_dual_ttt_dfs1024_benchmark.py" in source
    assert "--resume" not in source
    assert "retry" not in lowered
    assert "solutions" not in lowered
    assert "submission.json" not in lowered
    assert "nvidia-smi" in source
    assert "requires exactly four NVIDIA L4 GPUs" in source
    assert "/kaggle/working/analysis/l4_dual_ttt_dfs1024_bench_v1" in source


def test_kernel_metadata_is_private_offline_l4_benchmark() -> None:
    metadata = kernel_metadata("jimmy5566", "arc2-l4-dual-ttt-dfs1024-bench-v1", "arc2-l4-dual-ttt-dfs1024-bench-v1-source")
    assert metadata["enable_gpu"] is True
    assert metadata["enable_internet"] is False
    assert metadata["machine_shape"] == "NvidiaL4"
    assert metadata["competition_sources"] == ["arc-prize-2026-arc-agi-2"]
    assert metadata["model_sources"] == ["sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1"]


def test_runner_has_required_report_fields_and_no_retry_route() -> None:
    source = (ROOT / "scripts" / "run_l4_dual_ttt_dfs1024_benchmark.py").read_text(encoding="utf-8")
    for field in (
        '"WORKLOAD_WALL_H"',
        '"MODEL_LOAD_S"',
        '"PROJECTED_240_TASK_H"',
        '"FAILED_CELLS"',
        '"OOM_COUNT"',
        '"NOTEBOOK_TEST_STATUS"',
    ):
        assert field in source
    assert "work.put(task_id)" in source
    assert "for task_id in cohort[\"task_ids\"]" in source
    assert source.count("work.put(task_id)") == 1
    assert '"retry_failed_task": False' in source
    assert '"resume_completed": False' in source
