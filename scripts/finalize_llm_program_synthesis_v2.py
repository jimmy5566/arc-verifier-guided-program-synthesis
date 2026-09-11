"""Gate and score V2 pilot conditions after all predictions are frozen."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

from arc.io import discover_dataset_files, load_dataset

try:
    from scripts.finalize_llm_condition import deterministic_baseline_ids, score_predictions
except ModuleNotFoundError:  # pragma: no cover - direct entrypoint
    from finalize_llm_condition import deterministic_baseline_ids, score_predictions


ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def condition_metrics(checkpoint: dict, scored: dict, baseline: set[str]) -> dict:
    candidates = [candidate for record in checkpoint["records"].values() for candidate in record["candidate_results"]]
    statuses = Counter(candidate["status"] for candidate in candidates)
    total = len(candidates)
    rate = lambda numerator: None if not total else numerator / total
    provider_failed = statuses["PROVIDER_FAILED"]
    model_candidates = total - provider_failed
    model_rate = lambda numerator: None if not model_candidates else numerator / model_candidates
    parameterized = [candidate for candidate in candidates if candidate["parameter_resolutions"]]
    parameter_success = sum(all(resolution["status"] == "SUCCESS" for resolution in candidate["parameter_resolutions"]) for candidate in parameterized)
    exact, wrong = score_predictions(checkpoint["records"], scored)
    failure = Counter()
    for candidate in candidates:
        status = candidate["status"]
        if status == "PROVIDER_FAILED": label = "UNRESOLVED"
        elif status == "MACRO_SCHEMA_INVALID" or status == "MACRO_TYPE_INVALID": label = "MACRO_API_MISUNDERSTANDING"
        elif status in {"PARAMETER_AMBIGUOUS", "PARAMETER_INVALID"}: label = "PARAMETER_SOLVER_FAILURE"
        elif status == "COMPILER_INVALID": label = "COMPILER_LIMITATION"
        elif status in {"INVALID", "EXECUTION_FAILED", "LOW_LEVEL_INVALID"}: label = "COMPILER_LIMITATION"
        elif status == "TRAIN_INCONSISTENT": label = "PROGRAM_COMPOSITION_FAILURE"
        elif status == "TRAIN_CONSISTENT": label = "GENERALIZATION_FAILURE" if wrong else "SUCCESS"
        else: label = "UNRESOLVED"
        failure[label] += 1
    return {
        "task_count": checkpoint["task_count"], "candidate_count": total, "status_counts": dict(statuses),
        # Schema/type rates are conditional on a model response; provider outages
        # are reported separately instead of being mislabeled as invalid JSON.
        "provider_failure_rate": rate(provider_failed),
        "macro_response_candidate_count": model_candidates,
        "macro_schema_valid_rate": model_rate(model_candidates - statuses["MACRO_SCHEMA_INVALID"]),
        "macro_type_valid_rate": model_rate(model_candidates - statuses["MACRO_SCHEMA_INVALID"] - statuses["MACRO_TYPE_INVALID"]),
        "parameter_resolution_success_rate": None if not parameterized else parameter_success / len(parameterized),
        "compile_success_rate": rate(sum(candidate["compiled_program"] is not None for candidate in candidates)),
        "low_level_executable_rate": rate(statuses["TRAIN_INCONSISTENT"] + statuses["TRAIN_CONSISTENT"]),
        "train_consistent_rate": rate(statuses["TRAIN_CONSISTENT"]),
        "exact_solved": len(exact), "exact_solved_task_ids": exact,
        "exact_rate": len(exact) / checkpoint["task_count"], "train_consistent_but_test_wrong": len(wrong),
        "baseline_overlap_exact_solves": len(set(exact) & baseline),
        "newly_solved_task_ids": sorted(set(exact) - baseline),
        "composition_only_exact_solves": len(set(exact) - baseline),
        "composition_only_task_ids": sorted(set(exact) - baseline),
        "runtime_seconds": checkpoint["runtime_seconds"],
        "prompt_tokens": checkpoint["total_prompt_tokens"], "completion_tokens": checkpoint["total_output_tokens"],
        "failure_counts": dict(failure),
    }


def v1_comparable_metrics(task_ids: set[str]) -> dict:
    checkpoint = read_json(ROOT / "experiments" / "checkpoints" / "LLM_CONFIRMATION_DEV_100.json")
    candidates = [status for task_id, record in checkpoint["records"].items() if task_id in task_ids for status in record["candidate_statuses"]]
    counts = Counter(candidates); total = len(candidates)
    return {"task_count": len(task_ids), "candidate_count": total, "schema_valid_rate": (total - counts["SCHEMA_INVALID"]) / total, "type_valid_rate": (total - counts["SCHEMA_INVALID"] - counts["TYPE_INVALID"]) / total, "executable_rate": (counts["TRAIN_INCONSISTENT"] + counts["TRAIN_CONSISTENT"]) / total, "train_consistent_rate": counts["TRAIN_CONSISTENT"] / total, "exact_rate": 0.0, "prompt_tokens": sum(record.get("prompt_tokens", 0) for task_id, record in checkpoint["records"].items() if task_id in task_ids), "completion_tokens": sum(record.get("output_tokens", 0) for task_id, record in checkpoint["records"].items() if task_id in task_ids), "runtime_seconds": sum(record.get("elapsed_seconds", 0) for task_id, record in checkpoint["records"].items() if task_id in task_ids)}


def update_experiments_csv(result: dict, config: dict) -> None:
    """Replace only the V2 row after the prediction-freeze scoring gate."""
    path = ROOT / "experiments" / "experiments.csv"
    fields = ["experiment_id", "date", "git_commit", "solver", "representation", "search_method", "llm_model", "candidate_budget", "validation_split", "tasks_solved", "accuracy", "runtime_seconds", "gpu_hours", "notes"]
    with path.open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["experiment_id"] != result["experiment_id"]]
    symbolic = result["conditions"]["symbolic_parameter_solver"]
    rows.append({
        "experiment_id": result["experiment_id"], "date": str(date.today()), "git_commit": "uncommitted",
        "solver": "LLM Program Synthesis V2", "representation": "Macro DSL -> deterministic compiler -> capability executor",
        "search_method": "Qwen3 macro planning + hard train verifier", "llm_model": result["model"],
        "candidate_budget": str(result["candidate_budget"]), "validation_split": config["split"],
        "tasks_solved": str(symbolic["exact_solved"]), "accuracy": str(symbolic["exact_rate"]),
        "runtime_seconds": str(symbolic["runtime_seconds"]), "gpu_hours": str(symbolic["runtime_seconds"] / 3600),
        "notes": json.dumps({"status": result["status"], "decision": result["pilot_go_no_go"]["decision"], "symbolic_executable": symbolic["low_level_executable_rate"], "symbolic_train_consistent": symbolic["train_consistent_rate"], "composition_only": symbolic["composition_only_task_ids"]}),
    })
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-config", type=Path, default=ROOT / "configs" / "llm_program_synthesis_v2_pilot_50_qwen3_8b.json")
    parser.add_argument("--frozen-config", type=Path, default=ROOT / "configs" / "frozen_llm_program_synthesis_v2_qwen3_8b_transformers.json")
    parser.add_argument("--symbolic-checkpoint", type=Path, required=True)
    parser.add_argument("--direct-checkpoint", type=Path, required=True)
    parser.add_argument("--result", type=Path, default=ROOT / "experiments" / "results" / "LLM_PROGRAM_SYNTHESIS_V2_PILOT_50.json")
    parser.add_argument("--report", type=Path, default=ROOT / "reports" / "llm_program_synthesis_v2.md")
    args = parser.parse_args()
    config = read_json(args.task_config); frozen = read_json(args.frozen_config); declared = set(config["task_ids"])
    frozen_sha256 = hashlib.sha256(args.frozen_config.read_bytes()).hexdigest()
    checkpoints = {"symbolic_parameter_solver": read_json(args.symbolic_checkpoint), "direct_parameter_ablation": read_json(args.direct_checkpoint)}
    for name, checkpoint in checkpoints.items():
        if checkpoint.get("stage") != "PILOT_PREDICTIONS_FROZEN" or not checkpoint.get("complete") or set(checkpoint.get("records", ())) != declared:
            raise RuntimeError(f"solution access prohibited: {name} checkpoint is not a complete frozen prediction set with the declared task IDs")
        if checkpoint.get("frozen_config_sha256") != frozen_sha256:
            raise RuntimeError(f"solution access prohibited: {name} checkpoint does not match the declared frozen Qwen3-8B configuration")
    files = discover_dataset_files("data/raw")
    all_scored = load_dataset(files["training_challenges"], files["training_solutions"])
    scored = {task_id: all_scored[task_id] for task_id in declared}
    baseline = deterministic_baseline_ids(ROOT)
    conditions = {name: condition_metrics(checkpoint, scored, baseline) for name, checkpoint in checkpoints.items()}
    symbolic = conditions["symbolic_parameter_solver"]
    go = symbolic["low_level_executable_rate"] >= 0.25 or symbolic["train_consistent_rate"] > 0 or symbolic["composition_only_exact_solves"] > 0
    result = {"experiment_id": "LLM_PROGRAM_SYNTHESIS_V2", "status": "PILOT_COMPLETE_SCORED_AFTER_PREDICTION_FREEZE", "pilot_config": config, "model": frozen["model"]["model_source"], "candidate_budget": frozen["generation"]["candidate_budget"], "frozen_config_sha256": frozen_sha256, "frozen_deterministic_baseline_solved": len(baseline), "v1_comparable_development": v1_comparable_metrics(declared), "conditions": conditions, "pilot_go_no_go": {"decision": "GO" if go else "NO_GO", "rule": "GO only when symbolic V2 has executable rate >= 25%, train-consistent programs, or composition-only exact solves.", "reason": "evidence-based from frozen pilot metrics"}, "leakage_audit": "Both symbolic and direct Qwen3-8B checkpoints were complete, marked PILOT_PREDICTIONS_FROZEN, contained the same declared IDs, and matched the frozen-config SHA before this scorer opened training_solutions."}
    args.result.parent.mkdir(parents=True, exist_ok=True); args.report.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    primary_result = ROOT / "experiments" / "results" / "LLM_PROGRAM_SYNTHESIS_V2.json"
    primary_result.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    report = ["# LLM Program Synthesis V2", "", "Pilot is scored only after symbolic and direct predictions freeze.", "", "| Condition | Macro schema | Macro type | Parameter resolution | Compile | Executable | Train consistent | Exact |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, metrics in conditions.items(): report.append(f"| {name} | {metrics['macro_schema_valid_rate']:.3f} | {metrics['macro_type_valid_rate']:.3f} | {metrics['parameter_resolution_success_rate'] or 0:.3f} | {metrics['compile_success_rate']:.3f} | {metrics['low_level_executable_rate']:.3f} | {metrics['train_consistent_rate']:.3f} | {metrics['exact_rate']:.3f} |")
    report.extend(["", f"Frozen deterministic baseline: {len(baseline)} / 1000.", f"Pilot decision: **{result['pilot_go_no_go']['decision']}**."])
    args.report.write_text("\n".join(report) + "\n", encoding="utf-8")
    failures = ["# LLM Program Synthesis V2 failures", "", "This report is generated only after the prediction-freeze scoring gate.", ""]
    for name, metrics in conditions.items():
        failures.extend([f"## {name}", "", "| Failure class | Candidate count |", "| --- | ---: |"])
        failures.extend(f"| {label} | {count} |" for label, count in sorted(metrics["failure_counts"].items()))
        failures.append("")
    (ROOT / "reports" / "llm_program_synthesis_v2_failures.md").write_text("\n".join(failures), encoding="utf-8")
    update_experiments_csv(result, config)
    print(json.dumps({"decision": result["pilot_go_no_go"]["decision"], "conditions": {name: {"executable": metrics["low_level_executable_rate"], "exact": metrics["exact_solved"]} for name, metrics in conditions.items()}}, indent=2))


if __name__ == "__main__":
    main()
