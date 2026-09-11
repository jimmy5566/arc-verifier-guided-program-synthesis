"""Score one completed frozen LLM condition, never before its prediction freeze."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np

from arc.io import discover_dataset_files, load_dataset


def deterministic_baseline_ids(root: Path) -> set[str]:
    results = root / "experiments" / "results"
    v0 = json.loads((results / "SOLVER_LIB_V0.json").read_text(encoding="utf-8"))
    ids = set(v0["unique_solved_task_ids"])
    for filename in ("SOLVER_LIB_V1_OBJECT.json", "SOLVER_LIB_V2_RELATION_COMPOSITION.json", "SOLVER_LIB_V3_PATTERN.json"):
        ids.update(json.loads((results / filename).read_text(encoding="utf-8"))["newly_solved_task_ids"])
    ids.update(json.loads((results / "ARC2_CAPABILITY_SYSTEM_V1.json").read_text(encoding="utf-8"))["newly_solved_task_ids"])
    if len(ids) != 22:
        raise RuntimeError(f"frozen deterministic baseline regression: expected 22 IDs, got {len(ids)}")
    return ids


def score_predictions(records: dict, tasks: dict) -> tuple[list[str], list[str]]:
    exact: list[str] = []
    wrong: list[str] = []
    for task_id, record in records.items():
        raw = record.get("prediction")
        if raw is None:
            continue
        prediction = [np.asarray(grid, dtype=np.int16) for grid in raw]
        expected = tasks[task_id].test
        if len(prediction) == len(expected) and all(np.array_equal(item, example.output.values) for item, example in zip(prediction, expected)):
            exact.append(task_id)
        else:
            wrong.append(task_id)
    return sorted(exact), sorted(wrong)


def update_experiment_csv(root: Path, task_config: dict, result: dict) -> None:
    """Replace only this condition's row so scoring is safely re-runnable."""
    path = root / "experiments" / "experiments.csv"
    fields = ["experiment_id", "date", "git_commit", "solver", "representation", "search_method", "llm_model", "candidate_budget", "validation_split", "tasks_solved", "accuracy", "runtime_seconds", "gpu_hours", "notes"]
    with path.open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["experiment_id"] != result["experiment_id"]]
    frozen = json.loads((root / task_config["frozen_config"]).read_text(encoding="utf-8"))
    rows.append({
        "experiment_id": result["experiment_id"], "date": str(date.today()), "git_commit": "uncommitted",
        "solver": "LLM Hypothesis Generator V1", "representation": "frozen registry canonical structured programs",
        "search_method": "Qwen3 generation + exact symbolic train verification", "llm_model": frozen["model"]["model"],
        "candidate_budget": str(frozen["generation"]["candidate_budget"]),
        "validation_split": task_config.get("split", "development_confirmation"), "tasks_solved": str(result["exact_solved"]),
        "accuracy": str(result["exact_solved"] / result["task_count"]), "runtime_seconds": str(result["runtime_seconds"]),
        "gpu_hours": str(result["runtime_seconds"] / 3600),
        "notes": json.dumps({"status": result["status"], "candidate_count": result["candidate_count"], "schema_valid_rate": result["schema_valid_rate"], "train_consistent_rate": result["train_consistent_rate"], "llm_only": result["llm_only_newly_solved_task_ids"]}),
    })
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--task-config", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    checkpoint = json.loads(args.checkpoint.read_text(encoding="utf-8"))
    task_config = json.loads(args.task_config.read_text(encoding="utf-8"))
    declared_ids = task_config["task_ids"]
    # This gate is intentionally before discovery/opening of the solution file.
    if not checkpoint.get("complete") or set(checkpoint.get("records", ())) != set(declared_ids):
        raise RuntimeError("solution access is prohibited until every declared task has a frozen checkpoint record")
    files = discover_dataset_files("data/raw")
    scored = load_dataset(files["training_challenges"], files["training_solutions"])
    exact, wrong = score_predictions(checkpoint["records"], scored)
    baseline = deterministic_baseline_ids(Path.cwd())
    statuses = Counter(status for item in checkpoint["records"].values() for status in item["candidate_statuses"])
    candidates = sum(statuses.values())
    runtime = float(checkpoint["runtime_seconds"])
    count = len(declared_ids)
    result = {
        "experiment_id": task_config["config_id"],
        "status": "COMPLETE_SCORED_AFTER_PREDICTION_FREEZE",
        "protocol": checkpoint["protocol"],
        "task_ids": declared_ids,
        "task_count": count,
        "exact_solved": len(exact),
        "exact_solved_task_ids": exact,
        "train_consistent_but_test_wrong": len(wrong),
        "train_consistent_but_test_wrong_task_ids": wrong,
        "frozen_deterministic_exact": len(set(declared_ids) & baseline),
        "deterministic_plus_llm_union": len((set(declared_ids) & baseline) | set(exact)),
        "llm_only_newly_solved_task_ids": sorted(set(exact) - baseline),
        "composition_only_task_ids": sorted(set(exact) - baseline),
        "status_counts": dict(statuses),
        "candidate_count": candidates,
        "schema_valid_rate": None if not candidates else 1 - statuses["SCHEMA_INVALID"] / candidates,
        "type_valid_rate": None if not candidates else 1 - (statuses["SCHEMA_INVALID"] + statuses["TYPE_INVALID"]) / candidates,
        "executable_rate": None if not candidates else (statuses["TRAIN_INCONSISTENT"] + statuses["TRAIN_CONSISTENT"]) / candidates,
        "train_consistent_rate": None if not candidates else statuses["TRAIN_CONSISTENT"] / candidates,
        "runtime_seconds": runtime,
        "runtime_per_task_seconds": runtime / count,
        "prompt_tokens": checkpoint.get("total_prompt_tokens", 0),
        "completion_tokens": checkpoint.get("total_output_tokens", 0),
        "prompt_tokens_per_task": checkpoint.get("total_prompt_tokens", 0) / count,
        "completion_tokens_per_task": checkpoint.get("total_output_tokens", 0) / count,
        "leakage_audit": "The checkpoint completion and declared-ID gates were checked before loading training_solutions.",
    }
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    args.report.write_text("# " + task_config["config_id"] + "\n\n```json\n" + json.dumps(result, indent=2) + "\n```\n", encoding="utf-8")
    update_experiment_csv(Path.cwd(), task_config, result)
    print(json.dumps({"exact_solved": result["exact_solved"], "task_count": count, "result": str(args.result)}, indent=2))


if __name__ == "__main__":
    main()
