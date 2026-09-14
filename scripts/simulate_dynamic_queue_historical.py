"""Simulate static and work-stealing schedules from frozen task runtimes."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from inference.dynamic_task_scheduler import simulate_dynamic_queue, simulate_static_round_robin


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True); parser.add_argument("--artifact", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError("refusing to overwrite historical simulation")
    cohort = json.loads(args.cohort.read_text(encoding="utf-8")); artifact = json.loads(args.artifact.read_text(encoding="utf-8"))
    task_ids = list(cohort.get("task_ids", ())); records = artifact.get("records", {})
    if not task_ids or not isinstance(records, dict) or set(task_ids) != set(records): raise ValueError("cohort and frozen runtime records must match exactly")
    runtimes = {task_id: float(records[task_id]["elapsed_seconds"]) for task_id in task_ids}
    if any(value < 0 for value in runtimes.values()): raise ValueError("negative historical runtime")
    static = simulate_static_round_robin(task_ids, runtimes, args.workers); dynamic = simulate_dynamic_queue(task_ids, runtimes, args.workers)
    saved = static.makespan_seconds - dynamic.makespan_seconds
    payload = {"experiment_id": "ARC2_DYNAMIC_QUEUE_HISTORICAL_SIMULATION_V1", "status": "CPU_ONLY_HISTORICAL_SIMULATION", "task_count": len(task_ids), "worker_count": args.workers, "runtime_source": "frozen task elapsed_seconds; no GPU inference", "static": static.to_dict(), "dynamic": dynamic.to_dict(), "absolute_time_saved_seconds": saved, "wall_time_reduction_pct": 0.0 if static.makespan_seconds == 0 else 100 * saved / static.makespan_seconds, "estimated_gpu_utilization": {"static_pct": 0.0 if static.makespan_seconds == 0 else 100 * sum(runtimes.values()) / (args.workers * static.makespan_seconds), "dynamic_pct": 0.0 if dynamic.makespan_seconds == 0 else 100 * sum(runtimes.values()) / (args.workers * dynamic.makespan_seconds)}}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"static_makespan_seconds": static.makespan_seconds, "dynamic_makespan_seconds": dynamic.makespan_seconds, "saved_seconds": saved, "reduction_pct": payload["wall_time_reduction_pct"]}, sort_keys=True))


if __name__ == "__main__":
    main()
