"""CPU-only, post-freeze Any-of-K scorer for one Phase-2 Eval3 task."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from inference.kaggle_l4_parallel_runner import atomic_write_json
from scripts.phase2_backend_contract import valid_frozen_artifact


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for name in ("candidates", "solutions_path", "output"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    return parser.parse_args()


def main() -> None:
    args = _args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite a frozen Phase-2 score")
    candidates = _read(args.candidates)
    if not valid_frozen_artifact(candidates, args.task_id):
        raise ValueError("candidate freeze is not a valid single-task Phase-2 artifact")
    record = candidates["records"][args.task_id]
    if not isinstance(record.get("candidates"), list):
        raise ValueError("candidate record is malformed")
    # Target boundary: candidate schema and source artifact are immutable now.
    solutions = _read(args.solutions_path)
    if args.task_id not in solutions:
        raise ValueError("development solution is unavailable for frozen task")
    hits = [index for index, candidate in enumerate(record["candidates"]) if candidate.get("prediction") == solutions[args.task_id]]
    payload: dict[str, Any] = {
        "status": "COMPLETE_SCORED_AFTER_CANDIDATE_FREEZE",
        "task_id": args.task_id,
        "candidate_artifact_sha256": hashlib.sha256(args.candidates.read_bytes()).hexdigest(),
        "ANY_OF_K": f"{int(bool(hits))}/1",
        "any_of_k_count": int(bool(hits)),
        "correct_candidate_indices": hits,
        "unique_candidate_count": int(record.get("unique_candidate_count", len(record["candidates"]))),
        "invalid_candidate_count": int(record.get("invalid_candidate_count", 0)),
        "candidate_freeze_validated_before_solutions_opened": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output, payload)
    print(json.dumps({"event": "BLACKWELL_PHASE2_ANYK_SCORED", **payload}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
