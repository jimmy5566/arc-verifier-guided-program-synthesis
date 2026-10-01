from __future__ import annotations

import copy
import hashlib
import json
import pickle
import queue
from pathlib import Path

import pytest

from scripts.build_l4_dual_ttt_dfs1024_benchmark_notebook import (
    BENCHMARK_ID,
    kernel_metadata,
    notebook_source,
)
from scripts.run_l4_dual_ttt_dfs1024_benchmark import (
    EXPECTED_WORKER_IDS,
    STARTUP_MILESTONES,
    abort_startup,
    build_model_ready_payload,
    build_worker_failure,
    collect_startup_diagnostics,
    compact_cell,
    decoder_from_config,
    drain_startup_queue,
    persist_startup_failure,
    persist_startup_milestone,
    projections,
    record_startup_message,
    startup_failure_path,
    startup_status_path,
    validate_cohort,
    validate_config,
    validate_model_ready_payload,
)


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "l4_dual_ttt_dfs1024_bench_v1"


def load(name: str) -> dict:
    return json.loads((EXPERIMENT / name).read_text(encoding="utf-8"))


class UnpickleableUUID:
    def __str__(self) -> str:
        return "GPU-test-uuid"

    def __reduce__(self) -> object:
        raise TypeError("UUID-like object must not enter the queue payload")


def model_ready_payload(properties: object) -> dict:
    return build_model_ready_payload(
        worker_id=2,
        properties=properties,
        gpu_name="NVIDIA L4",
        compute_capability=(8, 9),
        model_load_seconds=12.5,
        model_vram_mb=7123.0,
        tokenizer_metadata={"vocab_size": 16},
        torch_version="2.8.0+cu128",
        cuda_runtime="12.8",
    )


def ready_for(worker_id: int) -> dict:
    return build_model_ready_payload(
        worker_id=worker_id,
        properties=type("Properties", (), {"uuid": f"GPU-{worker_id}"})(),
        gpu_name=f"NVIDIA L4 worker {worker_id}",
        compute_capability=(8, 9),
        model_load_seconds=10.0 + worker_id,
        model_vram_mb=7000.0,
        tokenizer_metadata={"vocab_size": 16},
        torch_version="2.8.0+cu128",
        cuda_runtime="12.8",
    )


class FakeBarrier:
    def __init__(self) -> None:
        self.was_set = False

    def set(self) -> None:
        self.was_set = True


class FakeProcess:
    def __init__(self, worker_id: int, *, alive: bool = True, exitcode: int | None = None) -> None:
        self.pid = 1000 + worker_id
        self._alive = alive
        self.exitcode = exitcode
        self.terminated = False
        self.killed = False

    def is_alive(self) -> bool:
        return self._alive

    def terminate(self) -> None:
        self.terminated = True
        self._alive = False
        self.exitcode = -15

    def join(self, timeout: float | None = None) -> None:
        del timeout

    def kill(self) -> None:
        self.killed = True
        self._alive = False
        self.exitcode = -9


def test_model_ready_payload_converts_uuid_and_is_pickleable() -> None:
    properties = type("Properties", (), {"uuid": UnpickleableUUID()})()
    payload = model_ready_payload(properties)
    assert payload["gpu_uuid"] == "GPU-test-uuid"
    assert payload["gpu_name"] == "NVIDIA L4"
    assert pickle.loads(pickle.dumps(payload)) == payload


def test_model_ready_payload_keeps_missing_uuid_provenance() -> None:
    payload = model_ready_payload(object())
    assert "gpu_uuid" in payload
    assert payload["gpu_uuid"] is None


def test_model_ready_pickle_guard_fails_synchronously_for_other_bad_evidence() -> None:
    with pytest.raises(TypeError, match="must not enter"):
        build_model_ready_payload(
            worker_id=0,
            properties=object(),
            gpu_name="NVIDIA L4",
            compute_capability=(8, 9),
            model_load_seconds=1.0,
            model_vram_mb=1.0,
            tokenizer_metadata={"bad_evidence": UnpickleableUUID()},
            torch_version="2.8.0+cu128",
            cuda_runtime="12.8",
        )


def test_model_ready_pickle_guard_precedes_queue_and_startup_completion() -> None:
    source = (ROOT / "scripts" / "run_l4_dual_ttt_dfs1024_benchmark.py").read_text(encoding="utf-8")
    guard = source.index("pickle.dumps(ready_payload)")
    queue_send = source.index("ready.put(ready_payload)")
    startup_complete = source.index("startup_complete = True", queue_send)
    assert guard < queue_send < startup_complete


def test_a_four_unique_ready_messages_allow_startup() -> None:
    ready: dict[int, dict] = {}
    failures: dict[int, dict] = {}
    duplicates: dict[int, int] = {}
    for worker_id in range(4):
        record_startup_message(
            ready_for(worker_id),
            ready_by_worker=ready,
            failures_by_worker=failures,
            duplicate_ready_counts=duplicates,
        )
    assert set(ready) == EXPECTED_WORKER_IDS
    assert failures == {}
    assert duplicates == {}
    assert all(validate_model_ready_payload(row)["gpu_uuid"].startswith("GPU-") for row in ready.values())


def test_b_worker3_startup_exception_persists_and_aborts_peers(tmp_path: Path) -> None:
    (tmp_path / "checkpoints" / "startup").mkdir(parents=True)
    (tmp_path / "failures.jsonl").write_text("", encoding="utf-8")
    for worker_id in range(4):
        persist_startup_milestone(tmp_path, worker_id, "PROCESS_STARTED")
    failure = build_worker_failure(
        worker_id=3,
        phase="STARTUP",
        last_startup_milestone="MODEL_LOAD_STARTED",
        exc=RuntimeError("synthetic worker3 startup failure"),
        traceback_text="Traceback: synthetic worker3 startup failure",
    )
    persist_startup_failure(tmp_path, failure)
    processes = [FakeProcess(index) for index in range(3)] + [FakeProcess(3, alive=False, exitcode=1)]
    barrier = FakeBarrier()
    report = abort_startup(
        output=tmp_path,
        processes=processes,
        start_barrier=barrier,
        ready_by_worker={index: ready_for(index) for index in range(3)},
        failures_by_worker={3: failure},
        duplicate_ready_counts={},
        reason="synthetic startup failure",
    )
    assert startup_failure_path(tmp_path, 3).is_file()
    assert "synthetic worker3" in report["workers"][3]["traceback"]
    assert report["workers"][3]["exit_code"] == 1
    assert barrier.was_set is True
    assert all(process.terminated for process in processes[:3])
    assert report["cleanup_duration_s"] < 30
    assert (tmp_path / "STARTUP_FAILURE_REPORT.json").is_file()
    assert (tmp_path / "startup_summary.json").is_file()
    assert "synthetic worker3 startup failure" in (tmp_path / "failures.jsonl").read_text(encoding="utf-8")


def test_c_dead_process_drain_captures_pending_worker_failed() -> None:
    pending: queue.Queue[dict] = queue.Queue()
    failure = build_worker_failure(
        worker_id=3,
        phase="STARTUP",
        last_startup_milestone="TOKENIZER_VERIFIED",
        exc=RuntimeError("pending failure"),
        traceback_text="pending traceback",
    )
    pending.put(failure)
    ready: dict[int, dict] = {}
    failures: dict[int, dict] = {}
    duplicates: dict[int, int] = {}
    drained = drain_startup_queue(pending)
    for item in drained:
        record_startup_message(
            item,
            ready_by_worker=ready,
            failures_by_worker=failures,
            duplicate_ready_counts=duplicates,
        )
    assert failures[3]["traceback"] == "pending traceback"


def test_d_disk_failure_is_authoritative_when_ipc_is_missing(tmp_path: Path) -> None:
    (tmp_path / "checkpoints" / "startup").mkdir(parents=True)
    persist_startup_milestone(tmp_path, 3, "MODEL_LOAD_STARTED")
    failure = build_worker_failure(
        worker_id=3,
        phase="STARTUP",
        last_startup_milestone="MODEL_LOAD_STARTED",
        exc=ValueError("disk-only startup failure"),
        traceback_text="disk-only traceback",
    )
    persist_startup_failure(tmp_path, failure)
    processes = [FakeProcess(index) for index in range(3)] + [FakeProcess(3, alive=False, exitcode=1)]
    diagnostics = collect_startup_diagnostics(
        output=tmp_path,
        processes=processes,
        ready_by_worker={},
        failures_by_worker={},
    )
    assert diagnostics[3]["failure_artifact_present"] is True
    assert diagnostics[3]["exception_type"] == "ValueError"
    assert diagnostics[3]["traceback"] == "disk-only traceback"


def test_e_duplicate_ready_is_deduplicated_by_worker_id() -> None:
    ready: dict[int, dict] = {}
    failures: dict[int, dict] = {}
    duplicates: dict[int, int] = {}
    payload = ready_for(0)
    for _ in range(2):
        record_startup_message(
            payload,
            ready_by_worker=ready,
            failures_by_worker=failures,
            duplicate_ready_counts=duplicates,
        )
    assert list(ready) == [0]
    assert duplicates == {0: 1}


def test_worker_startup_failure_protocol_orders_disk_ipc_and_nonzero_exit() -> None:
    source = (ROOT / "scripts" / "run_l4_dual_ttt_dfs1024_benchmark.py").read_text(encoding="utf-8")
    outer = source.index('if phase == "STARTUP":')
    persist = source.index("persist_startup_failure(output, failure)", outer)
    pickle_guard = source.index("pickle.dumps(failure)", persist)
    queue_send = source.index("ready.put(failure)", pickle_guard)
    flush = source.index("flush_child_queue(ready)", queue_send)
    nonzero_exit = source.index("raise SystemExit(1)", flush)
    assert outer < persist < pickle_guard < queue_send < flush < nonzero_exit


def test_all_requested_startup_milestones_are_frozen() -> None:
    assert STARTUP_MILESTONES == (
        "PROCESS_STARTED",
        "CUDA_BOUND",
        "MODEL_LOAD_STARTED",
        "MODEL_LOAD_COMPLETE",
        "TOKENIZER_VERIFIED",
        "LORA_ATTACHED",
        "DEFAULT_STATE_CAPTURED",
        "DATASET_LOADED",
        "DECODER_READY",
        "MODEL_READY_PAYLOAD_BUILT",
        "MODEL_READY_PAYLOAD_PICKLE_PASS",
        "MODEL_READY_SENT",
        "START_BARRIER_ENTERED",
        "START_BARRIER_RELEASED",
    )


def test_scientific_config_and_cohort_content_is_frozen() -> None:
    expected = {
        "benchmark_config.json": "9e2a978302d13ce427c9e180603e642e3353fb8dbb9c31e11cb110d396fb1493",
        "BENCHMARK_TASK_IDS.json": "f2dba422faaaaf4e77bd2efb06603300f8be3b27ff656da28cfe58e8cb3a55fa",
    }
    for name, wanted in expected.items():
        value = json.loads((EXPERIMENT / name).read_text(encoding="utf-8"))
        canonical = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
        assert hashlib.sha256(canonical).hexdigest() == wanted
    assert (EXPERIMENT / "BENCHMARK_COHORT_SHA256.txt").read_text(encoding="utf-8").strip() == (
        "d6d4c7ac9017e2d3005a17c9b9ffdcf7f0e6c59ecd273abd87ba43e618c6f8c7"
    )
    config = load("benchmark_config.json")
    assert config["benchmark_id"] == "L4_DUAL_TTT_DFS1024_NOTEBOOK_BENCH_V1"
    assert config["ttt24_recipe"]["ttt_steps"] == 24
    assert config["ttt48_recipe"]["ttt_steps"] == 48
    assert config["search"] == {
        "engine": "authoritative_scalar_d1_regret",
        "policy": "CUMULATIVE_REGRET_r=4.00",
        "max_expanded_nodes": 1024,
        "max_completed_candidates": 32,
        "max_new_tokens": 931,
        "max_score": 1.6094379124341003,
        "frontier_floor": 1,
        "local_time_limit_seconds": 540.0,
        "pad_token_id": 13,
        "arc_tokens": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15],
        "lane_count": 1,
        "diagnostic_trace": False,
        "batch2_cross_cell": False,
        "batch4_regret": False,
    }


def test_v1_v2_failures_are_infrastructure_only_with_no_scientific_data() -> None:
    v1 = load("FAILED_V1_INFRA_PROVENANCE.json")
    v2 = load("FAILED_V2_STARTUP_PROVENANCE.json")
    assert v1["benchmark_id"] == v2["benchmark_id"] == "L4_DUAL_TTT_DFS1024_NOTEBOOK_BENCH_V1"
    assert v1["previous_gpu_run_scientific_data"] == "NONE"
    assert v2["previous_gpu_run_scientific_data"] == "NONE"
    assert v2["model_ready_uuid_fix"] == "PASS"
    assert v2["root_worker_exception"] == "UNKNOWN_NOT_PERSISTED"
    for record in (v1, v2):
        assert record["scientific_data_produced"] is False
        assert record["tasks_started"] == 0
        assert record["ttt_runs"] == 0
        assert record["dfs_cells"] == 0
        assert record["candidates"] == 0


def test_successful_runner_lifecycle_comparison_is_science_neutral() -> None:
    text = (EXPERIMENT / "SUCCESSFUL_RUNNER_LIFECYCLE_COMPARISON.md").read_text(encoding="utf-8")
    assert 'multiprocessing.get_context("spawn")' in text
    assert "CUDA_VISIBLE_DEVICES=i" in text
    assert "exact unique workers `{0,1,2,3}`" in text
    assert "No TTT, augmentation, DFS, candidate, cohort, batching, or" in text


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
