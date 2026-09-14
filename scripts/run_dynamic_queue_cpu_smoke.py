"""CPU-only validation of the static and dynamic task schedulers."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from inference.dynamic_task_scheduler import run_cpu_scheduler


def main() -> None:
    # Unequal durations deliberately create the static round-robin long tail.
    durations = {"task_00": 0.45, "task_01": 0.01, "task_02": 0.02, "task_03": 0.01, "task_04": 0.40, "task_05": 0.01, "task_06": 0.02, "task_07": 0.01}
    static_assignments, static_wall = run_cpu_scheduler(durations, 4, dynamic=False)
    dynamic_assignments, dynamic_wall = run_cpu_scheduler(durations, 4, dynamic=True)
    resumed, _resumed_wall = run_cpu_scheduler(durations, 4, dynamic=True, completed={"task_00", "task_04"})
    payload = {"status": "CPU_DYNAMIC_QUEUE_SMOKE_PASS", "task_count": len(durations), "static": {"wall_seconds": static_wall, "task_workers": static_assignments}, "dynamic": {"wall_seconds": dynamic_wall, "task_workers": dynamic_assignments}, "every_task_once": set(dynamic_assignments) == set(durations) and len(dynamic_assignments) == len(durations), "fast_worker_received_additional_tasks": max(list(dynamic_assignments.values()).count(worker) for worker in set(dynamic_assignments.values())) > 1, "dynamic_not_slower": dynamic_wall <= static_wall, "resume_skipped_completed": set(resumed) == set(durations) - {"task_00", "task_04"}}
    if not all(payload[key] for key in ("every_task_once", "fast_worker_received_additional_tasks", "dynamic_not_slower", "resume_skipped_completed")):
        raise RuntimeError(payload)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
