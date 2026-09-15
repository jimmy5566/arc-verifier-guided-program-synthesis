"""Compare frozen Dynamic-B artifacts without opening ARC target solutions.

This is used twice: the four 8-task micro-batch conditions are all compared
to batch=1, then the selected Speed V2 30-task run is compared with the
reused OLD Dynamic-B artifact.  Runtime fields are informative; prediction
equivalence is decided exclusively from target-blind candidate and B attempt
artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _prediction_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _candidate_support(record: dict[str, Any]) -> dict[str, int]:
    candidates = record.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("candidate record lacks candidates")
    result: dict[str, int] = {}
    for candidate in candidates:
        key = _prediction_key(candidate.get("prediction"))
        if key in result:
            raise ValueError("deduplicated candidate pool contains duplicate prediction")
        result[key] = int(candidate.get("support_count", 1))
    return result


def _candidate_order(record: dict[str, Any]) -> list[str]:
    return [_prediction_key(candidate.get("prediction")) for candidate in record.get("candidates", ())]


def _b_attempts(record: dict[str, Any]) -> list[str]:
    candidates = list(record.get("candidates", ()))
    selected = record.get("public_reference_selection", {})
    indices = [int(index) for index in selected.get("attempt_candidate_indices", ())]
    if not indices or len(indices) > 2 or any(index < 0 or index >= len(candidates) for index in indices):
        raise ValueError("invalid frozen B attempt indices")
    values = [_prediction_key(candidates[index].get("prediction")) for index in indices]
    return values + [values[0]] * (2 - len(values))


def _evidence(record: dict[str, Any]) -> dict[str, tuple[float, tuple[float, ...]]]:
    candidates = list(record.get("candidates", ()))
    raw = record.get("b_support_evidence")
    if not isinstance(raw, list) or len(raw) != len(candidates):
        return {}
    by_index = {int(value.get("candidate_index", -1)): value for value in raw if isinstance(value, dict)}
    if set(by_index) != set(range(len(candidates))):
        return {}
    return {
        _prediction_key(candidate.get("prediction")): (
            float(by_index[index].get("original_log_likelihood")),
            tuple(float(value) for value in by_index[index].get("view_negative_log_likelihoods", ())),
        )
        for index, candidate in enumerate(candidates)
    }


def _runtime(artifact: dict[str, Any]) -> dict[str, Any]:
    records = artifact.get("records", {})
    by_worker: dict[str, dict[str, float | int]] = {}
    for record in records.values():
        worker = str(record.get("worker_id", "unknown"))
        item = by_worker.setdefault(worker, {"tasks": 0, "elapsed_seconds": 0.0})
        item["tasks"] = int(item["tasks"]) + 1
        item["elapsed_seconds"] = float(item["elapsed_seconds"]) + float(record.get("elapsed_seconds", 0.0))
    return {
        "wall_seconds": float(artifact.get("runtime_seconds", 0.0)),
        "generation_wall_seconds_sum": float(artifact.get("generation_wall_seconds_sum", 0.0)),
        "original_likelihood_seconds_sum": float(artifact.get("original_likelihood_seconds_sum", 0.0)),
        "b_support_scoring_seconds_sum": float(artifact.get("b_support_scoring_seconds_sum", 0.0)),
        "peak_allocated_vram_mb": int(artifact.get("peak_allocated_vram_mb", 0)),
        "retries": int(artifact.get("retry_count", 0)),
        "failed_task_ids": list(artifact.get("deadline", {}).get("failed_task_ids", ())),
        "worker_load": by_worker,
    }


def compare(baseline: dict[str, Any], candidate: dict[str, Any], baseline_selection: dict[str, Any], candidate_selection: dict[str, Any]) -> dict[str, Any]:
    base_records, new_records = baseline.get("records"), candidate.get("records")
    base_selected, new_selected = baseline_selection.get("records"), candidate_selection.get("records")
    if not all(isinstance(value, dict) for value in (base_records, new_records, base_selected, new_selected)):
        raise ValueError("all input artifacts need record dictionaries")
    task_ids = sorted(base_records)
    if set(new_records) != set(task_ids) or set(base_selected) != set(task_ids) or set(new_selected) != set(task_ids):
        raise ValueError("artifact task sets differ")
    per_task: dict[str, Any] = {}
    for task_id in task_ids:
        base_support, new_support = _candidate_support(base_records[task_id]), _candidate_support(new_records[task_id])
        base_order, new_order = _candidate_order(base_records[task_id]), _candidate_order(new_records[task_id])
        base_attempts, new_attempts = _b_attempts(base_selected[task_id]), _b_attempts(new_selected[task_id])
        old_evidence, new_evidence = _evidence(base_records[task_id]), _evidence(new_records[task_id])
        deltas: list[float] = []
        if old_evidence and new_evidence and set(old_evidence) == set(new_evidence):
            for key in old_evidence:
                old_original, old_views = old_evidence[key]; new_original, new_views = new_evidence[key]
                if len(old_views) == len(new_views):
                    deltas.extend([abs(old_original - new_original), *(abs(left - right) for left, right in zip(old_views, new_views, strict=True))])
        per_task[task_id] = {
            "candidate_pool_support_exact": base_support == new_support,
            "candidate_order_exact": base_order == new_order,
            "candidate_count_baseline": len(base_support),
            "candidate_count_candidate": len(new_support),
            "b_attempts_exact": base_attempts == new_attempts,
            "baseline_attempts": base_attempts,
            "candidate_attempts": new_attempts,
            "max_abs_likelihood_delta": max(deltas) if deltas else None,
        }
    return {
        "status": "TARGET_BLIND_ARTIFACT_COMPARISON_COMPLETE",
        "targets_or_solutions_opened": False,
        "task_count": len(task_ids),
        "task_ids": task_ids,
        "task_ids_hash": hashlib.sha256(json.dumps(task_ids, separators=(",", ":")).encode()).hexdigest(),
        "candidate_pool_support_exact_tasks": sum(item["candidate_pool_support_exact"] for item in per_task.values()),
        "candidate_order_exact_tasks": sum(item["candidate_order_exact"] for item in per_task.values()),
        "b_attempt_exact_tasks": sum(item["b_attempts_exact"] for item in per_task.values()),
        "max_abs_likelihood_delta": max((item["max_abs_likelihood_delta"] or 0.0 for item in per_task.values()), default=0.0),
        "baseline_runtime": _runtime(baseline),
        "candidate_runtime": _runtime(candidate),
        "per_task": per_task,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-candidates", type=Path, required=True)
    parser.add_argument("--candidate-candidates", type=Path, required=True)
    parser.add_argument("--baseline-selection", type=Path, required=True)
    parser.add_argument("--candidate-selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite frozen comparison report")
    result = compare(_read(args.baseline_candidates), _read(args.candidate_candidates), _read(args.baseline_selection), _read(args.candidate_selection))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("status", "task_count", "candidate_pool_support_exact_tasks", "candidate_order_exact_tasks", "b_attempt_exact_tasks", "max_abs_likelihood_delta")}, sort_keys=True))


if __name__ == "__main__":
    main()
