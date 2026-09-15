"""Recover a valid partial A artifact solely from atomic task checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _runner() -> Any:
    spec = importlib.util.spec_from_file_location("production_native_runner", ROOT / "scripts" / "run_qwen4b_native_augmentation_search.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load native runner checkpoint validator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("cohort", "config", "checkpoint_dir", "output"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--augmentation-count", type=int, required=True)
    parser.add_argument("--worker-count", type=int, required=True)
    parser.add_argument("--search-beams", type=int, default=1)
    parser.add_argument("--generation-micro-batch-size", type=int, default=1)
    parser.add_argument("--likelihood-micro-batch-size", type=int, default=1)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite recovered candidate artifact")
    cohort = json.loads(args.cohort.read_text(encoding="utf-8"))
    task_ids = tuple(cohort.get("task_ids", ()))
    if cohort.get("status") != "PUBLIC_LB_TASKS_FROZEN_BEFORE_INFERENCE" or not task_ids or len(task_ids) != len(set(task_ids)):
        raise ValueError("requires a frozen complete competition cohort")
    runner = _runner()
    config_sha256 = hashlib.sha256(args.config.read_bytes()).hexdigest()
    checkpoint_identity = runner.checkpoint_identity_for(
        config_sha256=config_sha256, task_ids=task_ids,
        augmentation_count=args.augmentation_count, worker_count=args.worker_count,
        search_beams=args.search_beams,
        generation_micro_batch_size=args.generation_micro_batch_size,
        likelihood_micro_batch_size=args.likelihood_micro_batch_size,
    )
    records: dict[str, Any] = {}
    corrupt_or_mismatched: list[str] = []
    for task_id in task_ids:
        checkpoint = args.checkpoint_dir / "tasks" / f"{task_id}.json"
        record = runner._valid_checkpoint(checkpoint, task_id, checkpoint_identity, config_sha256)
        if record is not None:
            records[task_id] = record
        elif checkpoint.exists():
            corrupt_or_mismatched.append(task_id)
    payload = {
        "experiment_id": "ARC2_NATIVE_PUBLIC_LB_B_V1",
        "status": "CANDIDATES_AND_RANKED_PREDICTIONS_FROZEN_BEFORE_EXACT_SCORING" if len(records) == len(task_ids) else "DEADLINE_PARTIAL_CANDIDATES_FROZEN",
        "protocol": "Recovered only from validated atomic Dynamic-B checkpoints; no model load, targets, or new candidate generation.",
        "task_ids_hash": hashlib.sha256(json.dumps(sorted(task_ids), separators=(",", ":")).encode()).hexdigest(),
        "config_sha256": config_sha256,
        "checkpoint_identity": checkpoint_identity,
        "recovery": {"checkpoint_dir": str(args.checkpoint_dir), "completed_task_ids": sorted(records), "unfinished_task_ids": sorted(set(task_ids) - set(records)), "corrupt_or_mismatched_task_ids": corrupt_or_mismatched},
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(args.output.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(args.output)
    print(json.dumps({"status": payload["status"], "recovered_task_count": len(records), "unfinished_task_count": len(task_ids) - len(records), "corrupt_or_mismatched_task_count": len(corrupt_or_mismatched), "solutions_opened": False}, sort_keys=True))


if __name__ == "__main__":
    main()
