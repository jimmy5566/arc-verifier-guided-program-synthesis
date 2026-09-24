"""Post-freeze, task-level Any-of-K scoring for Blackwell Phase 1.

This scorer deliberately has no model imports.  It validates the immutable
candidate artifact before opening the development solutions, then records only
the requested candidate-pool diagnostic.  It is not a selector or a promotion
criterion on its own.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from inference.kaggle_l4_parallel_runner import atomic_write_json
from scripts.run_eval3_reference_ttt import _task_hash
from scripts.run_eval3_runtime_opt import FROZEN_STATUS


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for name in ("manifest", "candidates", "challenge_path", "solutions_path", "output"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite frozen Phase-1 report: {args.output}")

    manifest = _read(args.manifest)
    candidates = _read(args.candidates)
    task_ids = list(manifest.get("task_ids", ()))
    if (
        manifest.get("status") != "EVAL3_REFERENCE_TTT_COHORT_FROZEN"
        or len(task_ids) != 3
        or manifest.get("task_ids_hash") != _task_hash(task_ids)
    ):
        raise ValueError("invalid frozen Eval3 manifest")
    if hashlib.sha256(args.challenge_path.read_bytes()).hexdigest() != manifest.get("source_challenge_sha256"):
        raise ValueError("challenge source differs from frozen Eval3 manifest")
    if candidates.get("status") != FROZEN_STATUS:
        raise ValueError("candidate artifact was not frozen by the Blackwell runtime")
    if list(candidates.get("task_ids", ())) != task_ids or set(candidates.get("records", ())) != set(task_ids):
        raise ValueError("candidate artifact does not cover the immutable Eval3 task order")
    for task_id in task_ids:
        record = candidates["records"][task_id]
        if record.get("task_id") != task_id or not isinstance(record.get("candidates"), list):
            raise ValueError(f"invalid frozen candidate record for {task_id}")

    # Target boundary: all candidate pools above are validated and immutable.
    solutions = _read(args.solutions_path)
    if set(task_ids) - set(solutions):
        raise ValueError("solutions omit a frozen Eval3 task")
    rows: list[dict[str, Any]] = []
    recovered: list[str] = []
    for task_id in task_ids:
        record = candidates["records"][task_id]
        hits = [
            index
            for index, candidate in enumerate(record["candidates"])
            if candidate.get("prediction") == solutions[task_id]
        ]
        if hits:
            recovered.append(task_id)
        rows.append(
            {
                "task_id": task_id,
                "any_of_k": bool(hits),
                "correct_candidate_indices": hits,
                "unique_candidate_count": int(record.get("unique_candidate_count", len(record["candidates"]))),
                "invalid_candidate_count": int(record.get("invalid_candidate_count", 0)),
            }
        )
    report = {
        "experiment_id": candidates.get("experiment_id"),
        "status": "COMPLETE_SCORED_AFTER_CANDIDATE_FREEZE",
        "scoring_unit": "task_level_any_of_k",
        "candidate_artifact_sha256": hashlib.sha256(args.candidates.read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "challenge_sha256": hashlib.sha256(args.challenge_path.read_bytes()).hexdigest(),
        "task_ids": task_ids,
        "task_ids_hash": manifest["task_ids_hash"],
        "ANY_OF_K": f"{len(recovered)}/{len(task_ids)}",
        "any_of_k_count": len(recovered),
        "recovered_task_ids": recovered,
        "per_task": rows,
        "integrity": {
            "candidate_freeze_validated_before_solutions_opened": True,
            "selector_not_run": True,
            "any_of_k_is_diagnostic_only": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output, report)
    print(json.dumps({"event": "BLACKWELL_PHASE1_ANYK_SCORED", "ANY_OF_K": report["ANY_OF_K"], "output": str(args.output)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
