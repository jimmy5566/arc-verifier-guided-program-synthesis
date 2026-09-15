"""Score OLD versus NEW Speed V2 B attempts after both artifacts are frozen."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _attempts(record: dict[str, Any]) -> list[list[list[list[int]]]]:
    candidates = list(record.get("candidates", ()))
    indices = [int(value) for value in record.get("public_reference_selection", {}).get("attempt_candidate_indices", ())]
    if not indices or len(indices) > 2 or any(index < 0 or index >= len(candidates) for index in indices):
        raise ValueError("invalid frozen B attempts")
    values = [candidates[index].get("prediction") for index in indices]
    return values + [values[0]] * (2 - len(values))


def _validate(candidate: dict[str, Any], selection: dict[str, Any], task_ids: list[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    records, selected = candidate.get("records"), selection.get("records")
    if candidate.get("status") != "CANDIDATES_AND_RANKED_PREDICTIONS_FROZEN_BEFORE_EXACT_SCORING" or selection.get("status") != "PUBLIC_REFERENCE_SELECTION_FROZEN_BEFORE_EXACT_SCORING":
        raise ValueError("candidate and B selection must be frozen before scoring")
    if not isinstance(records, dict) or not isinstance(selected, dict) or set(records) != set(task_ids) or set(selected) != set(task_ids):
        raise ValueError("artifact task IDs do not match immutable Speed V2 cohort")
    for task_id in task_ids:
        if records[task_id].get("candidates") != selected[task_id].get("candidates"):
            raise ValueError(f"{task_id}: B selection is not tied to its candidate pool")
        _attempts(selected[task_id])
    return records, selected


def _metrics(records: dict[str, Any], selection: dict[str, Any], solutions: dict[str, Any], task_ids: list[str]) -> dict[str, Any]:
    top1 = two = any_of_k = 0
    outcomes: dict[str, bool] = {}
    for task_id in task_ids:
        expected, attempts = solutions[task_id], _attempts(selection[task_id])
        top1 += int(attempts[0] == expected)
        outcomes[task_id] = any(value == expected for value in attempts)
        two += int(outcomes[task_id])
        any_of_k += int(any(candidate.get("prediction") == expected for candidate in records[task_id]["candidates"]))
    return {"top1_exact": top1, "two_attempt_exact": two, "any_of_k_diagnostic": any_of_k, "outcomes": outcomes}


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("manifest", "old_candidates", "old_selection", "new_candidates", "new_selection", "solutions_path", "output"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite post-freeze Speed V2 score")
    manifest = _read(args.manifest)
    task_ids = list(manifest.get("task_ids", ()))
    if manifest.get("status") != "COHORT_FROZEN_BEFORE_SPEED_V2_INFERENCE_AND_TARGET_ACCESS" or len(task_ids) != 30:
        raise ValueError("requires immutable frozen 30-task Speed V2 manifest")
    old_records, old_selected = _validate(_read(args.old_candidates), _read(args.old_selection), task_ids)
    new_records, new_selected = _validate(_read(args.new_candidates), _read(args.new_selection), task_ids)
    # The only target boundary.  Every OLD and NEW B output is frozen and
    # structurally validated above before solution data is imported.
    from arc.io import load_solutions
    solutions = load_solutions(args.solutions_path)
    if any(task_id not in solutions for task_id in task_ids):
        raise ValueError("Speed V2 cohort target missing from solutions")
    old, new = _metrics(old_records, old_selected, solutions, task_ids), _metrics(new_records, new_selected, solutions, task_ids)
    result = {
        "experiment_id": "ARC2_SPEED_V2_FROZEN30_CONFIRMATION_V1",
        "status": "OLD_NEW_SCORED_AFTER_BOTH_PREDICTIONS_FROZEN",
        "task_count": len(task_ids),
        "old": {key: value for key, value in old.items() if key != "outcomes"},
        "new": {key: value for key, value in new.items() if key != "outcomes"},
        "prediction_exact_match_tasks": sum(old["outcomes"][task_id] == new["outcomes"][task_id] for task_id in task_ids),
        "old_only_two_attempt": sorted(task_id for task_id in task_ids if old["outcomes"][task_id] and not new["outcomes"][task_id]),
        "new_only_two_attempt": sorted(task_id for task_id in task_ids if new["outcomes"][task_id] and not old["outcomes"][task_id]),
        "integrity": {"solutions_loaded_only_after_old_and_new_b_freezes": True, "no_target_aware_batch_selection": True},
        "artifact_sha256": {name: hashlib.sha256(getattr(args, name).read_bytes()).hexdigest() for name in ("manifest", "old_candidates", "old_selection", "new_candidates", "new_selection")},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("task_count", "old", "new", "prediction_exact_match_tasks")}, sort_keys=True))


if __name__ == "__main__":
    main()
