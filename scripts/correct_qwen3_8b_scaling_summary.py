"""Correct derived latency quantiles from frozen scaling task checkpoints.

No model, challenge, or solution input is opened.  This only recomputes the
nearest-rank p90 statistic from already-frozen per-task timing records.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import median


def p90(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[math.ceil(len(ordered) * 0.9) - 1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scaling", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    aggregate = json.loads(args.scaling.read_text(encoding="utf-8"))
    corrected: dict[str, dict[str, dict[str, float | int]]] = {}
    for worker_count, condition in aggregate["conditions"].items():
        condition_result: dict[str, dict[str, float | int]] = {}
        for worker_id, worker in condition["worker_metrics"].items():
            root = args.checkpoints / f"{worker_count}w_symbolic"
            records = [json.loads(path.read_text(encoding="utf-8")) for path in root.glob("task_*.json")]
            assigned = set(condition["worker_reports"][int(worker_id)]["task_ids"])
            values = [float(record["total_task_seconds"]) for record in records if record["task_id"] in assigned and isinstance(record.get("total_task_seconds"), (int, float))]
            if not values:
                raise RuntimeError(f"missing timing records for {worker_count}w worker {worker_id}")
            condition_result[worker_id] = {"tasks_completed": len(values), "median_task_seconds": median(values), "p90_task_seconds": p90(values)}
        corrected[worker_count] = condition_result
    output = {
        "experiment_id": aggregate["experiment_id"],
        "stage": "QWEN3_8B_1_2_4_GPU_SCALING_SMOKE_CORRECTED_LATENCY_SUMMARY",
        "method": "nearest-rank p90 recomputed solely from frozen task checkpoint total_task_seconds; no inference or data file access",
        "source_scaling_file": str(args.scaling),
        "corrected_worker_latency": corrected,
        "speedup_and_throughput_unchanged": {key: {name: value for name, value in condition.items() if name in {"wall_seconds", "tasks_per_hour", "speedup_vs_1", "parallel_efficiency_4"}} for key, condition in aggregate["conditions"].items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "SUCCESS", "conditions": sorted(corrected)}))


if __name__ == "__main__":
    main()
