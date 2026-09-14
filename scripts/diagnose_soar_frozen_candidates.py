"""Train-only health audit for persisted SOAR candidate programs.

It intentionally does not accept a solution file.  It can therefore be run
before any retrospective exact scoring and cannot use a test target.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    artifact = json.loads(args.artifact.read_text(encoding="utf-8"))
    records: dict[str, dict[str, Any]] = artifact["records"]
    candidates = [candidate for record in records.values() for candidate in record["candidate_programs"]]
    verification = [candidate.get("verification", {}) for candidate in candidates]
    executions = [entry for item in verification for entry in item.get("train_execution", ())]
    best_by_task: dict[str, dict[str, Any]] = {}
    for task_id, record in sorted(records.items()):
        programs = record["candidate_programs"]
        passes = [int(program.get("verification", {}).get("train_pass_count", 0)) for program in programs]
        pair_count = max((int(program.get("verification", {}).get("train_pair_count", 0)) for program in programs), default=0)
        best = max(passes, default=0)
        best_by_task[task_id] = {
            "train_pair_count": pair_count,
            "best_train_pass_count": best,
            "best_train_fraction": (best / pair_count) if pair_count else 0.0,
            "near_miss": bool(pair_count and best == pair_count - 1),
            "train_exact_candidates": sum(bool(program.get("all_train_exact")) for program in programs),
        }
    status_counts = Counter(str(item.get("status", "MISSING")) for item in executions)
    error_counts = Counter(
        str(item.get("error") or item.get("status") or "MISSING")
        for item in executions
        if not item.get("ok")
    )
    summary = {
        "status": "TRAIN_ONLY_DIAGNOSTIC_COMPLETE",
        "artifact": str(args.artifact),
        "task_count": len(records),
        "candidate_count": len(candidates),
        "funnel": {
            "extracted_program": sum(candidate.get("extracted_code") is not None for candidate in candidates),
            "parse_valid": sum(bool(item.get("parse_valid")) for item in verification),
            "static_safe": sum(bool(item.get("static_safe")) for item in verification),
            "process_started_train_executions": sum(bool(item.get("process_started")) for item in executions),
            "runtime_completed_train_executions": sum(item.get("status") not in {"TIMEOUT", "PROCESS_START_FAILED"} for item in executions),
            "output_valid_train_executions": sum(item.get("grid") is not None for item in executions),
            "train_exact_candidates": sum(bool(candidate.get("all_train_exact")) for candidate in candidates),
        },
        "raw_output": {
            "closed_python_fence": sum(str(candidate.get("raw_model_output", "")).rstrip().endswith("```") for candidate in candidates),
            "unclosed_or_nonfenced": sum(not str(candidate.get("raw_model_output", "")).rstrip().endswith("```") for candidate in candidates),
            "max_characters": max((len(str(candidate.get("raw_model_output", ""))) for candidate in candidates), default=0),
            "median_characters": sorted(len(str(candidate.get("raw_model_output", ""))) for candidate in candidates)[len(candidates) // 2] if candidates else 0,
        },
        "execution_status_counts": dict(sorted(status_counts.items())),
        "top_failure_reasons": error_counts.most_common(20),
        "tasks": best_by_task,
        "near_miss_tasks": sorted(task_id for task_id, item in best_by_task.items() if item["near_miss"]),
        "tasks_with_train_exact": sorted(task_id for task_id, item in best_by_task.items() if item["train_exact_candidates"] > 0),
        "legacy_interface_observations": {
            "legacy_runner_prompt": "custom transform(input_grid) prompt; not the public SOAR prompt formatter",
            "legacy_runner_sampling": "K=8, temperature=0.2, top_p=0.95, max_new_tokens=768",
            "official_reference": "public SOAR uses its numpy task formatter, includes test input, defaults to temperature=1/top_p=1/min_p=.05, tutorial initial max_tokens=4096 and a separate repair phase",
        },
        "leakage_audit": "No solution path was accepted or read. All values derive from persisted generation and train-execution records.",
    }
    write_json(args.output, summary)
    print(json.dumps({"event": "SOAR_TRAIN_ONLY_DIAGNOSTIC_COMPLETE", "tasks": len(records), "candidates": len(candidates), "train_exact_candidates": summary["funnel"]["train_exact_candidates"]}, sort_keys=True))


if __name__ == "__main__":
    main()
