"""Compare frozen30 construction audits without accessing tests or solutions."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def _operation(record: dict[str, object]) -> str | None:
    spec = record.get("matching_complete_rulespec")
    if not isinstance(spec, dict):
        return None
    skeleton = spec.get("skeleton", {})
    steps = skeleton.get("steps", []) if isinstance(skeleton, dict) else []
    if not isinstance(steps, list) or not steps:
        return None
    final = steps[-1]
    return final.get("operation") if isinstance(final, dict) and isinstance(final.get("operation"), str) else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--repeat-forensics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite frozen construction forensics")
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    current = json.loads(args.current.read_text(encoding="utf-8"))
    repeat = json.loads(args.repeat_forensics.read_text(encoding="utf-8"))
    if baseline["task_ids_hash"] != current["task_ids_hash"]:
        raise ValueError("baseline/current cohorts differ")
    primary_repeat = {task_id: value["primary"] for task_id, value in repeat["records"].items()}
    records: dict[str, dict[str, object]] = {}
    recovered_by_operation: dict[str, list[str]] = defaultdict(list)
    for task_id, before in baseline["records"].items():
        after = current["records"][task_id]
        was_exact, is_exact = before["status"] == "COVERED", after["status"] == "COVERED"
        operation = _operation(after)
        if not was_exact and is_exact and operation:
            recovered_by_operation[operation].append(task_id)
        taxonomy = primary_repeat.get(task_id, after["taxonomy"])
        records[task_id] = {
            "baseline_status": before["status"],
            "current_status": after["status"],
            "candidate_count": after.get("complete_rulespec_candidates", 0),
            "candidate_contains_exact_rulespec": is_exact,
            "recovered": not was_exact and is_exact,
            "semantic_category": taxonomy,
            "operation": operation,
        }
    remaining = Counter(record["semantic_category"] for record in records.values() if record["current_status"] != "COVERED")
    artifact = {
        "experiment_id": "ARC2_V3_COMPLETE_RULESPEC_CONSTRUCTION_PUSH",
        "status": "TRAIN_ONLY_CONSTRUCTION_FORENSICS_COMPLETE",
        "task_ids_hash": current["task_ids_hash"],
        "protocol": "Compares frozen train-only audit artifacts and train-pair repeat forensic labels only. No test grid/output, solution file, model, Kaggle or E2E run is accessed.",
        "before_exact_coverage": baseline["coverage"],
        "after_exact_coverage": current["coverage"],
        "correct_rulespec_candidate_recall": sum(record["candidate_contains_exact_rulespec"] for record in records.values()),
        "recovered_by_operation": {key: sorted(value) for key, value in sorted(recovered_by_operation.items())},
        "remaining_taxonomy": dict(sorted(remaining.items())),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"before": artifact["before_exact_coverage"], "after": artifact["after_exact_coverage"], "recovered": sum(len(value) for value in recovered_by_operation.values()), "recall": artifact["correct_rulespec_candidate_recall"], "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()}, sort_keys=True))


if __name__ == "__main__":
    main()
