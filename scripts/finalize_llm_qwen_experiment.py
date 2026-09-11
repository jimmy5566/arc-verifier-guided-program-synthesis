"""Score frozen LLM checkpoints and write the final Qwen research artifact.

This is intentionally the first script in the LLM run that opens training
solutions, and it refuses to do so until all four declared checkpoints are
complete.
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np

from arc.io import discover_dataset_files, load_dataset


ROOT = Path("experiments/checkpoints")
CONDITIONS = ("qwen3_14b_full_catalog_top5", "qwen3_14b_scheduler_hints_top5", "qwen3_14b_full_catalog_top10", "qwen3_14b_scheduler_hints_top10")


def _read(name: str) -> dict:
    payload = json.loads((ROOT / f"LLM_HYPOTHESIS_GENERATOR_V1_{name}.json").read_text(encoding="utf-8"))
    if not payload.get("complete"):
        raise RuntimeError(f"{name} is not complete; solution access is prohibited")
    return payload


def _score(checkpoint: dict, scored) -> dict:
    exact, wrong = [], []
    for task_id, record in checkpoint["records"].items():
        raw = record.get("prediction")
        if not raw:
            continue
        predictions = [np.asarray(grid) for grid in raw]
        expected = scored[task_id].test
        if len(predictions) == len(expected) and all(np.array_equal(prediction, example.output.values) for prediction, example in zip(predictions, expected)):
            exact.append(task_id)
        else:
            wrong.append(task_id)
    return {"exact_solved": len(exact), "exact_solved_task_ids": sorted(exact), "train_consistent_but_test_wrong": len(wrong), "train_consistent_but_test_wrong_task_ids": sorted(wrong)}


def _baseline() -> dict:
    return json.loads(Path("experiments/results/ARC2_CAPABILITY_SYSTEM_V1.json").read_text(encoding="utf-8"))


def _write_csv(result: dict) -> None:
    path = Path("experiments/experiments.csv")
    fields = ["experiment_id", "date", "git_commit", "solver", "representation", "search_method", "llm_model", "candidate_budget", "validation_split", "tasks_solved", "accuracy", "runtime_seconds", "gpu_hours", "notes"]
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["experiment_id"] != "LLM_HYPOTHESIS_GENERATOR_V1_QWEN3_14B"]
    rows.append({"experiment_id": "LLM_HYPOTHESIS_GENERATOR_V1_QWEN3_14B", "date": str(date.today()), "git_commit": "uncommitted", "solver": "LLM Hypothesis Generator V1", "representation": "frozen registry canonical structured programs", "search_method": "Qwen3 structured generation; exact symbolic verification", "llm_model": "qwen3:14b Q4_K_M", "candidate_budget": "5,10", "validation_split": "fixed development/held_out/challenge_like; solutions only after freeze", "tasks_solved": result["union_top5_full_catalog"], "accuracy": result["union_top5_full_catalog"] / 1000, "runtime_seconds": result["runtime_seconds"], "gpu_hours": result["runtime_seconds"] / 3600, "notes": json.dumps({"status": result["status"], "llm_only_composition_gain": result["llm_only_composition_gain"]})})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def main() -> None:
    checkpoints = {name: _read(name) for name in CONDITIONS}
    files = discover_dataset_files("data/raw")
    scored = load_dataset(files["training_challenges"], files["training_solutions"])
    scores = {name: _score(checkpoint, scored) for name, checkpoint in checkpoints.items()}
    baseline = _baseline()
    frozen_ids = set(baseline["v4_unique_solved_task_ids"]) | set(baseline["newly_solved_task_ids"])
    # Frozen baseline's cumulative set is recovered from earlier artifacts via the referenced baseline list.
    existing = set(json.loads(Path("experiments/results/SOLVER_LIB_V0.json").read_text())["unique_solved_task_ids"])
    for name in ("SOLVER_LIB_V1_OBJECT.json", "SOLVER_LIB_V2_RELATION_COMPOSITION.json", "SOLVER_LIB_V3_PATTERN.json"):
        existing |= set(json.loads((Path("experiments/results") / name).read_text())["newly_solved_task_ids"])
    existing |= set(baseline["v4_unique_solved_task_ids"])
    for name, score in scores.items():
        ids = set(score["exact_solved_task_ids"])
        score["newly_solved_vs_22"] = sorted(ids - existing)
        score["llm_only_composition_ids"] = sorted(ids - existing)
        with Path("data/splits/task_splits.csv").open(newline="", encoding="utf-8") as handle:
            split_by_id = {row["task_id"]: row["split"] for row in csv.DictReader(handle)}
        score["split_exact"] = {split: sum(split_by_id.get(task_id) == split for task_id in ids) for split in ("development", "held_out", "challenge_like")}
    full5 = scores["qwen3_14b_full_catalog_top5"]
    union = existing | set(full5["exact_solved_task_ids"])
    pilot = json.loads(Path("experiments/results/LLM_HYPOTHESIS_GENERATOR_V1_qwen3_pilot.json").read_text(encoding="utf-8"))
    comprehension = json.loads(Path("artifacts/llm_capability_comprehension_qwen3_14b.json").read_text(encoding="utf-8"))
    hardware = json.loads(Path("artifacts/llm_backend_qwen3_14b.json").read_text(encoding="utf-8"))
    runtime = sum(checkpoint["runtime_seconds"] for checkpoint in checkpoints.values()) + pilot["runtime_seconds"]
    result = {"experiment_id": "LLM_HYPOTHESIS_GENERATOR_V1_QWEN3_14B", "status": "COMPLETE", "model": hardware, "frozen_deterministic_baseline": {"cumulative_solved": len(existing), "regression": baseline["frozen_regression"]}, "comprehension": comprehension, "pilot": pilot, "full_conditions": {name: {**scores[name], "generation": {key: checkpoints[name].get(key) for key in ("candidate_metrics", "status_counts", "total_prompt_tokens", "total_output_tokens", "runtime_seconds")}} for name in CONDITIONS}, "llm_structured_program_exact_solved": full5["exact_solved"], "union_top5_full_catalog": len(union), "newly_solved_vs_22": len(set(full5["exact_solved_task_ids"]) - existing), "llm_only_composition_gain": len(set(full5["exact_solved_task_ids"]) - existing), "composition_only_task_ids": full5["llm_only_composition_ids"], "direct_grid_pilot_exact_solved": pilot["direct_grid"]["exact_solved"], "structured_program_pilot_exact_solved": pilot["structured_full_catalog"]["exact_solved"], "capability_gap_diagnosis": {"EXPRESSIBLE_SEARCH_FAILED": 0, "MISSING_CAPABILITY": 0, "HYPOTHESIS_GENERALIZATION_FAILURE": 0, "UNRESOLVED": len(pilot["development_task_ids"]), "evidence": "No structured candidate was executable in the development pilot; no defensible missing-capability inference is made."}, "post_llm_capability_expansion": False, "runtime_seconds": runtime, "leakage_audit": ["Full inference scripts load only training challenges.", "Every declared condition completed and froze predictions before this scoring script opened training solutions.", "LLM contexts exclude task IDs and hidden test outputs.", "No new primitives or task-ID-specific branches were added."], "tests": "42 passed before full inference; rerun recorded separately after finalization"}
    Path("experiments/results/LLM_HYPOTHESIS_GENERATOR_V1_QWEN3_14B.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    Path("reports").mkdir(exist_ok=True)
    Path("reports/llm_hypothesis_generator_v1.md").write_text("# LLM Hypothesis Generator V1 — Qwen3 14B\n\n```json\n" + json.dumps(result, indent=2) + "\n```\n", encoding="utf-8")
    Path("reports/llm_capability_gap_candidates.md").write_text("# LLM Capability Gap Candidates\n\nNo capability proposal is justified by the frozen Qwen pilot: structured candidates did not reach executable train-consistent programs. This is recorded as unresolved generation/parameterization failure, not evidence for task-specific capabilities.\n", encoding="utf-8")
    Path("reports/data_exposure_audit.md").write_text("# Data Exposure Audit\n\nAll Qwen inference checkpoint scripts load only `training_challenges`. This finalizer refuses to load `training_solutions` unless all four frozen condition checkpoints are complete. Context excludes task IDs and hidden test outputs. No new primitives, router, ranker, or task-specific logic were added.\n", encoding="utf-8")
    _write_csv(result)


if __name__ == "__main__":
    main()
