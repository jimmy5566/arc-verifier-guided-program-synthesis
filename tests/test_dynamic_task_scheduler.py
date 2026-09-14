from __future__ import annotations

from pathlib import Path

from inference.dynamic_task_scheduler import detect_cpu_dead_worker, run_cpu_retry_scheduler, run_cpu_scheduler, simulate_dynamic_queue, simulate_static_round_robin, task_seed


ROOT = Path(__file__).resolve().parents[1]


def test_dynamic_simulation_reduces_the_static_long_tail() -> None:
    tasks = ["a", "b", "c", "d", "e", "f", "g", "h"]
    durations = {"a": 20.0, "b": 1.0, "c": 2.0, "d": 1.0, "e": 18.0, "f": 1.0, "g": 2.0, "h": 1.0}
    static = simulate_static_round_robin(tasks, durations, 4)
    dynamic = simulate_dynamic_queue(tasks, durations, 4)
    assert static.assignments[0] == ["a", "e"]
    assert dynamic.makespan_seconds < static.makespan_seconds
    assert sorted(task for values in dynamic.assignments.values() for task in values) == sorted(tasks)


def test_cpu_spawn_queue_runs_once_balances_and_resumes() -> None:
    durations = {"a": 0.45, "b": 0.01, "c": 0.01, "d": 0.01, "e": 0.40, "f": 0.01, "g": 0.01, "h": 0.01}
    static, static_wall = run_cpu_scheduler(durations, 4, dynamic=False)
    dynamic, dynamic_wall = run_cpu_scheduler(durations, 4, dynamic=True)
    assert set(static) == set(durations) == set(dynamic)
    assert len(dynamic) == len(set(dynamic))
    assert dynamic_wall <= static_wall
    resumed, _wall = run_cpu_scheduler(durations, 4, dynamic=True, completed={"a", "e"})
    assert set(resumed) == set(durations) - {"a", "e"}
    # Fast workers should steal work after their first short job.
    assert any(list(dynamic.values()).count(worker) > 1 for worker in set(dynamic.values()))


def test_task_seed_is_task_scoped_and_not_worker_scoped() -> None:
    assert task_seed("abc", 7) == task_seed("abc", 7)
    assert task_seed("abc", 7) != task_seed("def", 7)
    assert task_seed("abc", 7, "augmentation:1") != task_seed("abc", 7, "augmentation:2")


def test_cpu_scheduler_retries_failed_task_once_without_duplicate_completion() -> None:
    completed, retries = run_cpu_retry_scheduler({"a": 0.01, "b": 0.01, "c": 0.01}, 2, fail_once={"b"})
    assert set(completed) == {"a", "b", "c"}
    assert retries == {"b": 1}


def test_cpu_scheduler_detects_a_dead_worker() -> None:
    assert detect_cpu_dead_worker() == 23


def test_native_runner_uses_dynamic_queue_and_compact_task_events() -> None:
    source = (ROOT / "scripts/run_qwen4b_native_augmentation_search.py").read_text(encoding="utf-8")
    assert "task_queue.get()" in source and '"dynamic_fifo_shared_queue"' in source
    assert "TASK_START" in source and "TASK_COMPLETE" in source and "TASK_RETRY" in source
    assert "get_context(\"spawn\")" in source and "torch.cuda.current_device()" in source
    assert "worker exited unexpectedly" in source and "TASK_DEADLINE_SKIPPED" in source
    assert "CANDIDATE_HEARTBEAT" not in source
    assert "buckets = [task_ids[index::worker_count]" not in source
