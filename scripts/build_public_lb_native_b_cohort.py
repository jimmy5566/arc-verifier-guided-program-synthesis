"""Freeze the public-LB task IDs from challenges without accessing targets."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--challenge-path", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite immutable Public-LB cohort")
    challenges = json.loads(args.challenge_path.read_text(encoding="utf-8"))
    task_ids = sorted(challenges)
    if not task_ids:
        raise ValueError("challenge set is empty")
    payload = {"experiment_id": "ARC2_NATIVE_PUBLIC_LB_B_V1", "status": "PUBLIC_LB_TASKS_FROZEN_BEFORE_INFERENCE", "source": "competition test challenges", "task_count": len(task_ids), "task_ids": task_ids, "task_ids_hash": hashlib.sha256(json.dumps(task_ids, separators=(",", ":")).encode()).hexdigest(), "integrity": {"solutions_opened": False, "test_targets_available": False, "method": "frozen A candidate generation followed by frozen B selection"}}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "task_count": len(task_ids), "task_ids_hash": payload["task_ids_hash"]}, sort_keys=True))


if __name__ == "__main__":
    main()
