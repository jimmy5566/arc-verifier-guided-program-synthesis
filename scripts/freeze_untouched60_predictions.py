"""Freeze A/B two-attempt predictions from one identical candidate pool.

This deliberately has no solution-file argument or import.  It is the hard
boundary immediately before retrospective exact scoring.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "src"))
from inference.nvarc_public_reference import prediction_key, two_attempt_indices


def _json_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _a_attempts(record: dict[str, Any]) -> list[int]:
    candidates = list(record.get("candidates", ()))
    return two_attempt_indices(record.get("ranked_candidate_indices", ()), candidates)


def _b_attempts(record: dict[str, Any]) -> list[int]:
    selection = record.get("public_reference_selection")
    if not isinstance(selection, dict):
        raise ValueError("B record lacks frozen public-reference selection")
    candidates = list(record.get("candidates", ()))
    attempts = [int(index) for index in selection.get("attempt_candidate_indices", ())]
    expected = two_attempt_indices(selection.get("ranked_candidate_indices", ()), candidates)
    if attempts != expected:
        raise ValueError("B attempt policy is not the predeclared two-distinct-output policy")
    return attempts


def _freeze(label: str, artifact: dict[str, Any], task_ids: list[str]) -> dict[str, Any]:
    records = artifact.get("records")
    if not isinstance(records, dict) or set(records) != set(task_ids):
        raise ValueError(f"{label}: candidate artifact task IDs do not match the immutable manifest")
    if label == "A" and artifact.get("status") != "CANDIDATES_AND_RANKED_PREDICTIONS_FROZEN_BEFORE_EXACT_SCORING":
        raise ValueError("A: candidates must be frozen before selection")
    if label == "B" and artifact.get("status") != "PUBLIC_REFERENCE_SELECTION_FROZEN_BEFORE_EXACT_SCORING":
        raise ValueError("B: public-reference selection must be frozen before prediction freeze")
    by_task: dict[str, Any] = {}
    for task_id in task_ids:
        record = records[task_id]
        candidates = list(record.get("candidates", ()))
        if not candidates or any("prediction" not in item for item in candidates):
            raise ValueError(f"{label}/{task_id}: no complete candidate pool")
        attempts = _a_attempts(record) if label == "A" else _b_attempts(record)
        if not attempts or any(index < 0 or index >= len(candidates) for index in attempts):
            raise ValueError(f"{label}/{task_id}: invalid predeclared attempt indices")
        predictions = [candidates[index]["prediction"] for index in attempts]
        by_task[task_id] = {
            "attempt_candidate_indices": attempts,
            "attempt_1": predictions[0],
            "attempt_2": predictions[1] if len(predictions) > 1 else None,
            "attempt_count": len(predictions),
            "candidate_count": len(candidates),
            "candidate_pool_task_hash": _json_hash(candidates),
            "duplicate_attempt": len(predictions) == 2 and prediction_key(predictions[0]) == prediction_key(predictions[1]),
        }
    return {
        "experiment_id": "ARC2_UNTOUCHED60_NATIVE_PUBLIC_REFERENCE_B_V1",
        "method": label,
        "status": "PREDICTIONS_FROZEN_BEFORE_EXACT_SCORING",
        "task_count": len(task_ids),
        "task_ids": task_ids,
        "task_ids_hash": _json_hash(sorted(task_ids)),
        "source_candidate_artifact_sha256": None,  # filled by caller from bytes
        "prediction_protocol": "attempt 1 plus predeclared next distinct ranked output; no target/solution data was available",
        "records": by_task,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--a-candidates", type=Path, required=True)
    parser.add_argument("--b-selection", type=Path, required=True)
    parser.add_argument("--a-output", type=Path, required=True)
    parser.add_argument("--b-output", type=Path, required=True)
    args = parser.parse_args()
    if args.a_output.exists() or args.b_output.exists():
        raise FileExistsError("refusing to overwrite an immutable prediction artifact")
    manifest, a, b = _read(args.manifest), _read(args.a_candidates), _read(args.b_selection)
    task_ids = list(manifest.get("task_ids", ()))
    if manifest.get("status") != "COHORT_FROZEN_BEFORE_INFERENCE_AND_TARGET_ACCESS" or len(task_ids) != 60:
        raise ValueError("requires the immutable untouched60 manifest")
    if b.get("public_reference_source_sha256") != hashlib.sha256(args.a_candidates.read_bytes()).hexdigest():
        raise ValueError("B is not demonstrably derived from this exact A candidate artifact")
    a_out, b_out = _freeze("A", a, task_ids), _freeze("B", b, task_ids)
    a_out["source_candidate_artifact_sha256"] = hashlib.sha256(args.a_candidates.read_bytes()).hexdigest()
    b_out["source_candidate_artifact_sha256"] = hashlib.sha256(args.b_selection.read_bytes()).hexdigest()
    # Candidate grids must be byte-for-byte identical across A and B; only the
    # ranking/attempt selection may differ.
    for task_id in task_ids:
        if _json_hash(a["records"][task_id]["candidates"]) != _json_hash(b["records"][task_id]["candidates"]):
            raise ValueError(f"{task_id}: A/B candidate pools differ")
    for target, value in ((args.a_output, a_out), (args.b_output, b_out)):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "A_B_PREDICTIONS_FROZEN_BEFORE_EXACT_SCORING", "task_count": len(task_ids), "a_sha256": hashlib.sha256(args.a_output.read_bytes()).hexdigest(), "b_sha256": hashlib.sha256(args.b_output.read_bytes()).hexdigest()}, sort_keys=True))


if __name__ == "__main__":
    main()
