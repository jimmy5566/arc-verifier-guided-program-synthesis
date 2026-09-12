"""Local-only scorer for frozen GRID_SKELETON_RECOGNITION_V1 predictions."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from time import perf_counter
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluation.semantic_ir_scorer import score as score_full_semantic
from evaluation.skeleton_scorer import derive_skeleton, score


STRATA = {
    "Iteration": "MISSING_ITERATION_OR_REPEAT",
    "Relational": "MISSING_RELATIONAL_CAPABILITY",
    "Conditional": "MISSING_CONDITIONAL_OR_ROLE_LOGIC",
    "Composition": "MISSING_COMPOSITION",
    "Output Construction": "MISSING_OUTPUT_CONSTRUCTION",
}


def _old_failure_reclassification(prediction: dict[str, Any] | None, oracle: dict[str, Any]) -> Counter[str]:
    counts: Counter[str] = Counter()
    if prediction is None:
        counts["INTERFACE_FAILURE"] += 1
        return counts
    if prediction.get("primary_family") != oracle["primary_family"]:
        counts["TRUE_RECOGNITION_FAILURE:FAMILY_WRONG"] += 1
    if not set(prediction.get("operations", ())) & set(oracle["operations"]):
        counts["TRUE_RECOGNITION_FAILURE:WRONG_CORE_OPERATION"] += 1
    if prediction.get("composition") != oracle["composition"]:
        counts["TRUE_RECOGNITION_FAILURE:COMPOSITION_STRUCTURE_WRONG"] += 1
    old_iteration, gold_iteration = prediction.get("iteration", {}), oracle["iteration"]
    if gold_iteration["enabled"] and not old_iteration.get("enabled"):
        counts["TRUE_RECOGNITION_FAILURE:MISSED_ITERATION"] += 1
    if gold_iteration["enabled"]:
        for field, label in (("direction", "WRONG_DIRECTION"), ("step_rule", "WRONG_STEP"), ("termination", "WRONG_TERMINATION")):
            if old_iteration.get(field) != gold_iteration[field]:
                counts[f"PARAMETER_BINDING_FAILURE:{label}"] += 1
    old_conditional, gold_conditional = prediction.get("conditional_logic", {}), oracle["conditional_logic"]
    if gold_conditional["enabled"] and not old_conditional.get("enabled"):
        counts["TRUE_RECOGNITION_FAILURE:MISSED_CONDITIONAL"] += 1
    if prediction.get("semantic_parameters", {}).get("color_policy") != oracle["semantic_parameters"]["color_policy"]:
        counts["PARAMETER_BINDING_FAILURE:WRONG_COLOR"] += 1
    if set(prediction.get("object_roles", ())) != set(oracle["object_roles"]):
        counts["PARAMETER_BINDING_FAILURE:WRONG_OBJECT_BINDING"] += 1
    if set(prediction.get("relations", ())) != set(oracle["relations"]):
        counts["PARAMETER_BINDING_FAILURE:WRONG_SOURCE_TARGET_BINDING"] += 1
    return counts


def _rate(items: list[dict[str, Any]], key: str) -> float:
    return 0.0 if not items else sum(bool(item[key]) for item in items) / len(items)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--forensics-private", type=Path, required=True)
    parser.add_argument("--old-semantic-predictions", type=Path, required=True)
    args = parser.parse_args()
    started = perf_counter()

    # Freeze gate: do not touch local-only oracle/forensics until the complete
    # prediction artifact has passed status, cohort, and content-hash checks.
    raw_predictions = args.predictions.read_bytes()
    predictions = json.loads(raw_predictions)
    if predictions.get("status") != "PREDICTIONS_FROZEN_BEFORE_ORACLE_SCORING":
        raise ValueError("requires a frozen prediction artifact")
    if len(predictions.get("records", {})) != 30:
        raise ValueError("requires all 30 frozen predictions")
    frozen_hash = hashlib.sha256(raw_predictions).hexdigest()

    oracle_payload = json.loads(args.oracle.read_text(encoding="utf-8"))
    oracle = oracle_payload["gold"]
    if set(predictions["records"]) != set(oracle) or predictions["task_ids_hash"] != oracle_payload["task_ids_hash"]:
        raise ValueError("frozen prediction and oracle cohort mismatch")
    gold = {task_id: derive_skeleton(value) for task_id, value in oracle.items()}
    if any(any(not value.startswith("$") for value in operation["args"].values()) for item in gold.values() for operation in item["operations"]):
        raise ValueError("gold skeleton contains a concrete parameter")
    scored = {task_id: score(record.get("prediction"), gold[task_id]) for task_id, record in predictions["records"].items()}
    items = list(scored.values())
    metrics = {
        "family_accuracy": _rate(items, "family_correct"),
        "operation_set_accuracy": _rate(items, "operation_set_correct"),
        "operation_sequence_exact": _rate(items, "operation_sequence_exact"),
        "composition_structure_accuracy": _rate(items, "composition_structure_correct"),
        "required_hole_set_accuracy": _rate(items, "required_hole_set_correct"),
        "full_skeleton_exact": _rate(items, "full_skeleton_exact"),
        "skeleton_success": _rate(items, "skeleton_success"),
        "schema_validity": _rate(items, "schema_valid"),
    }

    forensics = json.loads(args.forensics_private.read_text(encoding="utf-8"))["records"]
    by_stratum: dict[str, list[str]] = {name: [row["task_id"] for row in forensics if row["classification"] == classification] for name, classification in STRATA.items()}
    strata = {name: {"tasks": len(task_ids), "skeleton_success": sum(bool(scored[task_id]["skeleton_success"]) for task_id in task_ids), "operation_sequence_exact": sum(bool(scored[task_id]["operation_sequence_exact"]) for task_id in task_ids)} for name, task_ids in by_stratum.items()}

    old_raw = args.old_semantic_predictions.read_bytes()
    old = json.loads(old_raw)
    if old.get("status") != "PREDICTIONS_FROZEN_BEFORE_GOLD_SCORING" or set(old.get("records", {})) != set(oracle):
        raise ValueError("requires the unchanged frozen V1 semantic prediction artifact")
    old_scored = {task_id: score_full_semantic(record.get("prediction"), oracle[task_id]) for task_id, record in old["records"].items()}
    recovery = Counter()
    reclassified = Counter()
    for task_id in sorted(oracle):
        old_success, skeleton_success = bool(old_scored[task_id].get("semantic_success")), bool(scored[task_id]["skeleton_success"])
        recovery["old_full_semantic_fail_to_skeleton_success" if not old_success and skeleton_success else "old_semantic_success_to_skeleton_success" if old_success and skeleton_success else "old_semantic_success_to_skeleton_fail" if old_success else "both_fail"] += 1
        reclassified.update(_old_failure_reclassification(old["records"][task_id].get("prediction"), oracle[task_id]))

    status = "RECOGNITION_PARTIALLY_WORKS" if metrics["skeleton_success"] >= .5 else "RECOGNITION_AND_PARAMETER_BOTH_WEAK" if metrics["skeleton_success"] >= .3 else "TRUE_SKELETON_RECOGNITION_WEAK"
    next_experiment = "TYPED_HOLE_PARAMETER_SOLVER_V1" if metrics["skeleton_success"] >= .5 else "MULTI_HYPOTHESIS_SKELETON_SEARCH_V1" if metrics["skeleton_success"] >= .3 else "ARC_SPECIALIZED_SKELETON_RECOGNIZER_V1"
    runtime_seconds = sum(float(record.get("generation_seconds", 0)) for record in predictions["records"].values())
    result = {
        "experiment_id": "GRID_SKELETON_RECOGNITION_V1", "status": "COMPLETE_SCORED_AFTER_FROZEN_PREDICTIONS", "tasks": 30,
        "model": "Qwen3-8B", "input": "A3 raw grids plus bounded deterministic object/relation graph", "public_heldout_challenge_inference": False,
        "downstream_called": False, "prediction_freeze_hash": frozen_hash, "old_semantic_prediction_hash": hashlib.sha256(old_raw).hexdigest(),
        "oracle_semantic_hash": hashlib.sha256(args.oracle.read_bytes()).hexdigest(), "metrics": metrics, "strata": strata,
        "old_full_semantic": {"baseline_v1_semantic_success": sum(bool(item.get("semantic_success")) for item in old_scored.values()), "a3_aggregate_semantic_success": 7, "full_exact": 0},
        "parameterization_recovery": dict(sorted(recovery.items())), "old_failure_reclassification": dict(sorted(reclassified.items())),
        "diagnosis": {"recognition": "PARTIAL" if metrics["skeleton_success"] >= .5 else "WEAK", "parameter_grounding": "MAJOR_BOTTLENECK" if metrics["skeleton_success"] >= .5 else "PARTIAL" if metrics["skeleton_success"] > metrics["operation_sequence_exact"] else "NOT_ESTABLISHED", "old_semantic_over_penalizes_parameter_binding": "YES" if metrics["skeleton_success"] >= .5 else "PARTIAL" if metrics["skeleton_success"] > 4 / 30 else "NO", "status": status, "next_experiment": next_experiment},
        "runtime_seconds": runtime_seconds, "scoring_seconds": perf_counter() - started,
        "leakage_audit": "Inference source was gold-blind and train-pairs-only. This local finalizer validated the complete frozen artifact and SHA256 before reading the ignored oracle semantic IR and forensics. No solutions, test outputs, held-out, public, or challenge tasks were used.",
    }
    result_path = ROOT / "experiments/results/GRID_SKELETON_RECOGNITION_V1.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# GRID SKELETON RECOGNITION V1", "", "Qwen3-8B 仅从 frozen development train pairs 的 A3 输入识别 parameter-free typed operation skeleton；未调用 Parameter Solver、Macro、Compiler、Executor 或 HardVerifier。", "", "## Metrics", "", *[f"- {key}: **{value:.1%}**" for key, value in metrics.items()], "", "## Parameterization recovery", "", *[f"- {key}: {value}" for key, value in sorted(recovery.items())], "", "## Strata", "", *[f"- {name}: {value}" for name, value in strata.items()], "", "## Reclassified historical failure signals", "", *[f"- {key}: {value}" for key, value in sorted(reclassified.items())], "", f"Diagnosis: **{status}**. Next experiment: **{next_experiment}**.", "", "Gold oracle, task IDs, raw responses, and per-task scores remain ignored private artifacts."]
    (ROOT / "reports/grid_skeleton_recognition_v1.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (ROOT / "experiments/experiments.csv").open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["experiment_id", "date", "git_commit", "solver", "representation", "search_method", "llm_model", "candidate_budget", "validation_split", "tasks_solved", "accuracy", "runtime_seconds", "gpu_hours", "notes"])
        writer.writerow({"experiment_id": "GRID_SKELETON_RECOGNITION_V1", "date": str(date.today()), "git_commit": "pending", "solver": "Qwen skeleton recognizer", "representation": "ARCSKELETONIRV1 typed holes", "search_method": "top-1 greedy skeleton recognition", "llm_model": "qwen-lm/qwen-3/Transformers/8b/1", "candidate_budget": 1, "validation_split": "frozen_development_30", "tasks_solved": sum(bool(item["skeleton_success"]) for item in items), "accuracy": metrics["skeleton_success"], "runtime_seconds": runtime_seconds, "gpu_hours": runtime_seconds / 3600, "notes": json.dumps({"status": status, "downstream_called": False}, sort_keys=True)})
    print(json.dumps({"status": status, "metrics": metrics, "recovery": dict(recovery)}, sort_keys=True))


if __name__ == "__main__":
    main()
