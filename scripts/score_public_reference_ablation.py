"""Score fully frozen Native A/B/C/D artifacts only after prediction freeze."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

FROZEN_STATUSES = {
    "CANDIDATES_AND_RANKED_PREDICTIONS_FROZEN_BEFORE_EXACT_SCORING",
    "PUBLIC_REFERENCE_SELECTION_FROZEN_BEFORE_EXACT_SCORING",
}


def _ranked(record: dict[str, Any], condition: str) -> list[int]:
    if condition in {"B", "D"}:
        selection = record.get("public_reference_selection", {})
        values = selection.get("ranked_candidate_indices", ())
    else:
        values = record.get("ranked_candidate_indices", ())
    return [int(value) for value in values]


def _attempts(record: dict[str, Any], condition: str) -> list[int]:
    if condition in {"B", "D"}:
        values = record.get("public_reference_selection", {}).get("attempt_candidate_indices", ())
        return [int(value) for value in values]
    # Existing A/C ranks are already deduplicated by full test-output tuple.
    return _ranked(record, condition)[:2]


def _validate(artifact: dict[str, Any], expected_ids: set[str]) -> dict[str, Any]:
    records = artifact.get("records")
    if artifact.get("status") not in FROZEN_STATUSES or not isinstance(records, dict) or set(records) != expected_ids:
        raise ValueError("requires a complete candidate/prediction artifact frozen before exact scoring")
    for task_id, record in records.items():
        candidates = record.get("candidates", ())
        if not candidates or any("prediction" not in candidate for candidate in candidates):
            raise ValueError(f"{task_id}: missing frozen candidates")
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("a", "b", "c", "d", "challenge_path", "solutions_path", "output"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite post-freeze score report")
    artifacts = {key: json.loads(getattr(args, key).read_text(encoding="utf-8")) for key in "abcd"}
    expected_ids = set(artifacts["a"].get("records", {}))
    if not expected_ids:
        raise ValueError("A must contain frozen task records")
    records = {key.upper(): _validate(value, expected_ids) for key, value in artifacts.items()}
    # The sole target boundary: all four artifacts have passed structural
    # completeness checks before loading local ARC solution grids.
    from arc.io import load_challenges, load_solutions
    challenges, solutions = load_challenges(args.challenge_path), load_solutions(args.solutions_path)
    if any(task_id not in challenges or task_id not in solutions for task_id in expected_ids):
        raise ValueError("frozen task is absent from challenge or solution data")
    conditions: dict[str, dict[str, Any]] = {key: {"top1": 0, "attempt12": 0, "any_of_k": 0, "by_task": {}} for key in "ABCD"}
    for task_id in sorted(expected_ids):
        expected = solutions[task_id]
        for condition in "ABCD":
            record = records[condition][task_id]; candidates = record["candidates"]
            rank = _ranked(record, condition); attempts = _attempts(record, condition)
            if not rank or any(index < 0 or index >= len(candidates) for index in rank + attempts):
                raise ValueError(f"{condition}/{task_id}: invalid frozen ranking")
            top1 = candidates[rank[0]]["prediction"] == expected
            attempt12 = any(candidates[index]["prediction"] == expected for index in attempts)
            any_of_k = any(candidate["prediction"] == expected for candidate in candidates)
            conditions[condition]["top1"] += int(top1); conditions[condition]["attempt12"] += int(attempt12); conditions[condition]["any_of_k"] += int(any_of_k)
            conditions[condition]["by_task"][task_id] = {"top1_exact": top1, "attempt12_exact": attempt12, "any_of_k_exact": any_of_k, "candidate_count": len(candidates)}
    baseline = conditions["A"]["by_task"]
    for condition in "BCD":
        by_task = conditions[condition]["by_task"]
        conditions[condition]["rescued_vs_A"] = sorted(task_id for task_id in expected_ids if by_task[task_id]["top1_exact"] and not baseline[task_id]["top1_exact"])
        conditions[condition]["harmed_vs_A"] = sorted(task_id for task_id in expected_ids if baseline[task_id]["top1_exact"] and not by_task[task_id]["top1_exact"])
    result = {
        "experiment_id": "ARC2_PUBLIC_REFERENCE_SELECTION_SEARCH_ABLATION_V1",
        "status": "A_B_C_D_SCORED_AFTER_ALL_PREDICTIONS_FROZEN",
        "task_count": len(expected_ids),
        "protocol": "All candidate pools/rankings were structurally complete and frozen before this process imported solutions. Attempt 2 is predeclared as the next distinct ranked output.",
        "artifacts_sha256": {key.upper(): hashlib.sha256(getattr(args, key).read_bytes()).hexdigest() for key in "abcd"},
        "conditions": conditions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: {metric: conditions[key][metric] for metric in ("top1", "attempt12", "any_of_k")} for key in "ABCD"}, sort_keys=True))


if __name__ == "__main__":
    main()
