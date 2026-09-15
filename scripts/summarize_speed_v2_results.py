"""Produce the final Speed V2 benchmark report from frozen run artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _workload(artifact: dict[str, Any]) -> dict[str, Any]:
    records = artifact.get("records", {})
    by_worker: dict[str, dict[str, float | int]] = {}
    for record in records.values():
        worker = str(record.get("worker_id", "unknown"))
        value = by_worker.setdefault(worker, {"tasks": 0, "elapsed_seconds": 0.0})
        value["tasks"] = int(value["tasks"]) + 1
        value["elapsed_seconds"] = float(value["elapsed_seconds"]) + float(record.get("elapsed_seconds", 0.0))
    elapsed = float(artifact.get("runtime_seconds", 0.0))
    return {
        "wall_seconds": elapsed,
        "generation_wall_seconds_sum": float(artifact.get("generation_wall_seconds_sum", artifact.get("generation_seconds_sum", 0.0))),
        "original_likelihood_seconds_sum": float(artifact.get("original_likelihood_seconds_sum", 0.0)),
        "b_support_scoring_seconds_sum": float(artifact.get("b_support_scoring_seconds_sum", 0.0)),
        "unique_candidates": sum(int(record.get("unique_candidate_count", 0)) for record in records.values()),
        "generated_candidates": sum(int(record.get("generated_candidate_count", 0)) for record in records.values()),
        "tasks": len(records),
        "tasks_per_hour": len(records) * 3600 / elapsed if elapsed else None,
        "unique_candidates_per_second": sum(int(record.get("unique_candidate_count", 0)) for record in records.values()) / elapsed if elapsed else None,
        "peak_allocated_vram_mb": int(artifact.get("peak_allocated_vram_mb", max((int(record.get("peak_allocated_vram_mb", 0)) for record in records.values()), default=0))),
        "oom_or_retries": int(artifact.get("retry_count", sum(int(record.get("retry_count", 0)) for record in records.values()))),
        "failed_task_ids": list(artifact.get("deadline", {}).get("failed_task_ids", ())),
        "model_loads": len(artifact.get("worker_ready", ())),
        "worker_load": by_worker,
        "gpu_utilization": artifact.get("gpu_utilization"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("phase1_selection", "old_candidates", "old_selection", "old_report", "new_candidates", "new_selection", "comparison", "score", "output"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite final Speed V2 report")
    phase1, old_a, old_b, old_report, new_a, new_b, comparison, score = (_read(getattr(args, name)) for name in ("phase1_selection", "old_candidates", "old_selection", "old_report", "new_candidates", "new_selection", "comparison", "score"))
    if phase1.get("status") != "SPEED_V2_BATCH_CONFIGURATION_FROZEN_TARGET_BLIND":
        raise ValueError("Phase 1 batch choice is not frozen target-blind")
    if comparison.get("candidate_pool_support_exact_tasks") != 30 or comparison.get("b_attempt_exact_tasks") != 30:
        raise ValueError("Speed V2 failed target-blind candidate/B-attempt equivalence")
    if score.get("status") != "OLD_NEW_SCORED_AFTER_BOTH_PREDICTIONS_FROZEN":
        raise ValueError("final accuracy score is not post-freeze")
    old, new = _workload(old_a), _workload(new_a)
    # OLD control wall time includes the historical serial B likelihood tail;
    # NEW's B selector is CPU-only because evidence was already checkpointed.
    old_total = float(old_report.get("runtime_seconds", old["wall_seconds"] + float(old_b.get("public_reference_selection_runtime_seconds", 0.0))))
    new_total = new["wall_seconds"] + float(new_b.get("public_reference_selection_runtime_seconds", 0.0))
    result = {
        "experiment_id": "ARC2_SPEED_V2",
        "status": "COMPLETE",
        "selected_micro_batch_size": int(phase1["selected_batch_size"]),
        "phase1": phase1,
        "old": {**old, "total_wall_seconds_including_b_selection": old_total, "b_selection_seconds": float(old_b.get("public_reference_selection_runtime_seconds", 0.0))},
        "new": {**new, "total_wall_seconds_including_cpu_b_selection": new_total, "b_selection_seconds": float(new_b.get("public_reference_selection_runtime_seconds", 0.0))},
        "speedup": old_total / new_total if new_total else None,
        "runtime_reduction_pct": (1 - new_total / old_total) * 100 if old_total else None,
        "candidate_pool_exact_match": comparison["candidate_pool_support_exact_tasks"],
        "candidate_order_exact_match": comparison["candidate_order_exact_tasks"],
        "b_prediction_exact_match": comparison["b_attempt_exact_tasks"],
        "max_abs_likelihood_delta": comparison.get("max_abs_likelihood_delta"),
        "old_two_attempt_accuracy": score["old"]["two_attempt_exact"],
        "new_two_attempt_accuracy": score["new"]["two_attempt_exact"],
        "integrity": {
            "new_predictions_frozen_before_solutions": bool(score.get("integrity", {}).get("solutions_loaded_only_after_old_and_new_b_freezes")),
            "target_blind_batch_choice": True,
            "old_control_source": "pre-Speed-V2 Dynamic-B with serial B scoring",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("selected_micro_batch_size", "speedup", "runtime_reduction_pct", "candidate_pool_exact_match", "b_prediction_exact_match", "old_two_attempt_accuracy", "new_two_attempt_accuracy")}, sort_keys=True))


if __name__ == "__main__":
    main()
