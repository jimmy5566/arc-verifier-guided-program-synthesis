"""Formal, frozen, leakage-safe evaluation for Solver Library v0."""
from __future__ import annotations

import csv
from datetime import date
import json
from pathlib import Path
from time import perf_counter

from arc.io import discover_dataset_files, load_dataset
from solvers.library_v0 import FixedCropSolver, ForegroundBBoxCropSolver, GlobalTransformSolver, RecolorSolver, TranslationSolver
from verification.verifier import verify_prediction


SOLVERS = {
    "GlobalTransformSolver": GlobalTransformSolver,
    "RecolorSolver": RecolorSolver,
    "FixedCropSolver": FixedCropSolver,
    "ForegroundBBoxCropSolver": ForegroundBBoxCropSolver,
    "TranslationSolver": TranslationSolver,
}


def main() -> None:
    files = discover_dataset_files("data/raw")
    # Inference phase: challenge data contains train outputs and unlabeled tests.
    tasks = load_dataset(files["training_challenges"])
    results: dict[str, dict[str, object]] = {}
    frozen: dict[str, dict[str, list[list[object]]]] = {}
    overall_start = perf_counter()
    for name, factory in SOLVERS.items():
        started = perf_counter()
        applicable: list[str] = []
        records: dict[str, list[list[object]]] = {}
        for task_id, task in tasks.items():
            solver = factory().fit(task)
            if solver.candidates:
                applicable.append(task_id)
            records[task_id] = [solver.predict(example.input.values, 1) for example in task.test]
        frozen[name] = records
        results[name] = {
            "applicable_tasks": len(applicable),
            "applicable_task_ids": applicable,
            "inference_runtime_seconds": perf_counter() - started,
        }

    # Scoring phase: this is the first read that attaches hidden test outputs.
    scored_tasks = load_dataset(files["training_challenges"], files["training_solutions"])
    for name, records in frozen.items():
        solved: list[str] = []
        outputs_solved = 0
        outputs_total = 0
        wrong = 0
        for task_id, task in scored_tasks.items():
            predictions_by_test = records[task_id]
            exact = [
                bool(predictions) and verify_prediction(example.output.values, predictions[0]).exact_match
                for example, predictions in zip(task.test, predictions_by_test)
            ]
            outputs_solved += sum(exact)
            outputs_total += len(exact)
            if exact and all(exact):
                solved.append(task_id)
            elif any(predictions_by_test):
                wrong += 1
        results[name].update(
            {
                "exact_solved_tasks": len(solved),
                "solved_task_ids": solved,
                "test_outputs_solved": outputs_solved,
                "test_outputs_total": outputs_total,
                "train_consistent_but_test_wrong_tasks": wrong,
            }
        )
    unique = sorted(set().union(*(set(result["solved_task_ids"]) for result in results.values())))
    payload = {
        "experiment_id": "SOLVER_LIB_V0",
        "protocol": "challenge-only inference and test-prediction freeze before training-solution scoring",
        "total_tasks": len(tasks),
        "solvers": results,
        "unique_solved_tasks": len(unique),
        "unique_solved_task_ids": unique,
        "runtime_seconds": perf_counter() - overall_start,
    }
    output = Path("experiments/results/SOLVER_LIB_V0.json")
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log = Path("experiments/experiments.csv")
    fields = [
        "experiment_id", "date", "git_commit", "solver", "representation", "search_method", "llm_model",
        "candidate_budget", "validation_split", "tasks_solved", "accuracy", "runtime_seconds", "gpu_hours", "notes",
    ]
    with log.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["experiment_id"] != "SOLVER_LIB_V0"]
    record = {
        "experiment_id": "SOLVER_LIB_V0",
        "date": str(date.today()),
        "git_commit": "uncommitted",
        "solver": "Solver Library v0",
        "representation": "whole-grid / foreground primitives",
        "search_method": "none",
        "llm_model": "",
        "candidate_budget": 1,
        "validation_split": "official_training_tasks_train_only_inference",
        "tasks_solved": len(unique),
        "accuracy": len(unique) / len(tasks),
        "runtime_seconds": payload["runtime_seconds"],
        "gpu_hours": 0,
        "notes": json.dumps({name: result["solved_task_ids"] for name, result in results.items()}),
    }
    with log.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow(record)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
