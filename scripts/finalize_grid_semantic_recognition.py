"""Score a frozen gold-blind semantic prediction artifact locally."""
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

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from evaluation.semantic_ir_scorer import score


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--predictions", type=Path, required=True); parser.add_argument("--gold", type=Path, required=True); parser.add_argument("--forensics-private", type=Path, required=True); args = parser.parse_args(); started = perf_counter()
    frozen = json.loads(args.predictions.read_text(encoding="utf-8"))
    if frozen.get("status") != "PREDICTIONS_FROZEN_BEFORE_GOLD_SCORING": raise ValueError("scoring requires a complete frozen prediction artifact")
    gold = json.loads(args.gold.read_text(encoding="utf-8"))["gold"]
    if set(frozen["records"]) != set(gold): raise ValueError("prediction/gold task sets differ")
    scored = {task_id: score(record.get("prediction"), gold[task_id]) for task_id, record in frozen["records"].items()}
    total = len(scored); families = defaultdict(list)
    for task_id, item in scored.items(): families[gold[task_id]["primary_family"]].append(item)
    def rate(key: str) -> float: return sum(bool(item.get(key)) for item in scored.values()) / total
    metrics = {"primary_family_accuracy": rate("primary_family_correct"), "core_operation_accuracy": rate("core_operation_correct"), "object_role_accuracy": sum(item.get("object_role_f1", 0) for item in scored.values()) / total, "relation_f1": sum(item.get("relation_f1", 0) for item in scored.values()) / total, "selection_criteria_accuracy": rate("selection_criteria_correct"), "iteration_detection_accuracy": rate("iteration_detection_correct"), "iteration_semantic_accuracy": rate("iteration_semantic_correct"), "conditional_logic_accuracy": rate("conditional_logic_correct"), "composition_accuracy": rate("composition_correct"), "output_construction_accuracy": rate("output_construction_correct"), "semantic_parameter_accuracy": rate("semantic_parameter_correct"), "full_semantic_ir_exact": rate("full_exact"), "semantic_ir_success": rate("semantic_success"), "slot_f1": sum(item.get("slot_f1", 0) for item in scored.values()) / total}
    family_metrics = {family: {"tasks": len(items), "semantic_success": sum(bool(item.get("semantic_success")) for item in items), "primary_family_correct": sum(bool(item.get("primary_family_correct")) for item in items)} for family, items in sorted(families.items())}
    def focused(family: str) -> dict[str, float | int]:
        items = families.get(family, [])
        return {"tasks": len(items), "iteration_detection": sum(bool(item.get("iteration_detection_correct")) for item in items), "iteration_semantics": sum(bool(item.get("iteration_semantic_correct")) for item in items), "conditional_detection": sum(bool(item.get("conditional_logic_correct")) for item in items), "semantic_success": sum(bool(item.get("semantic_success")) for item in items), "relation_f1": 0.0 if not items else sum(float(item.get("relation_f1", 0)) for item in items) / len(items), "object_role_f1": 0.0 if not items else sum(float(item.get("object_role_f1", 0)) for item in items) / len(items)}
    focused_metrics = {"iteration_repeat": focused("ITERATION_REPEAT"), "relational": focused("RELATIONAL_TRANSFORMATION"), "conditional_role": focused("CONDITIONAL_ROLE_LOGIC"), "output_construction": focused("OUTPUT_CONSTRUCTION")}
    errors = Counter(error for item in scored.values() for error in item.get("failure_taxonomy", ()))
    errors["IR_SCHEMA_FAILURE"] += sum(not bool(item.get("schema_valid")) for item in scored.values())
    forensic_records = json.loads(args.forensics_private.read_text(encoding="utf-8"))["records"]
    def stratum(classification: str) -> dict[str, int]:
        task_ids = [item["task_id"] for item in forensic_records if item["classification"] == classification]
        return {"tasks": len(task_ids), "semantic_success": sum(bool(scored[task_id].get("semantic_success")) for task_id in task_ids)}
    forensic_strata = {"iteration_repeat": stratum("MISSING_ITERATION_OR_REPEAT"), "relational": stratum("MISSING_RELATIONAL_CAPABILITY"), "conditional_role": stratum("MISSING_CONDITIONAL_OR_ROLE_LOGIC"), "composition_execution_gap": stratum("MISSING_COMPOSITION"), "output_construction": stratum("MISSING_OUTPUT_CONSTRUCTION")}
    status = "GRID_RECOGNITION_STRONG" if metrics["semantic_ir_success"] >= .7 else ("GRID_RECOGNITION_PARTIAL" if metrics["semantic_ir_success"] >= .4 else "GRID_RECOGNITION_WEAK")
    generation_seconds = sum(float(item.get("generation_seconds", 0)) for item in frozen["records"].values())
    result = {"experiment_id": "GRID_SEMANTIC_RECOGNITION_V1", "status": "COMPLETE", "task_count": total, "model": "Qwen3-8B", "downstream_capability_gate_used": False, "macro_compiler_executor_called": False, "new_heldout_public_challenge_inference": False, "prediction_freeze_hash": hashlib.sha256(args.predictions.read_bytes()).hexdigest(), "oracle_gold_hash": hashlib.sha256(args.gold.read_bytes()).hexdigest(), "metrics": metrics, "family_metrics": family_metrics, "focused_metrics": focused_metrics, "forensic_strata_metrics": forensic_strata, "failure_taxonomy": dict(sorted(errors.items())), "recognition_status": status, "execution_bottleneck": "CAPABILITY_COVERAGE", "generation_seconds_sum": generation_seconds, "runtime_seconds": generation_seconds, "scoring_seconds": perf_counter() - started, "leakage_audit": ["Kaggle inference artifact declares train-pairs-only input and gold-blind protocol.", "Gold was opened only by this local scorer after prediction artifact freeze.", "No raw grids, oracle IR, test outputs, task IDs, or raw responses are included in this public aggregate."]}
    out = ROOT / "experiments/results/GRID_SEMANTIC_RECOGNITION_V1.json"; out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# GRID SEMANTIC RECOGNITION V1", "", "Qwen3-8B 只根据 development train input/output pairs 进行 ARCSEMANTICIRV1 识别；未调用 downstream capability gate、Macro、Compiler、Executor、HardVerifier 或 program search。", "", "## Overall", "", *[f"- {key}: **{value:.1%}**" for key, value in metrics.items()], "", f"Recognition status: **{status}**.", "", "## Family-level", "", "| Family | Tasks | Semantic success | Family correct |", "| --- | ---: | ---: | ---: |", *[f"| {family} | {item['tasks']} | {item['semantic_success']} | {item['primary_family_correct']} |" for family, item in family_metrics.items()], "", "## Forensic strata", "", *[f"- {name}: {values}" for name, values in forensic_strata.items()], "", "`composition_execution_gap` 是上一轮 forensics 的执行表达缺口，而非独立 semantic COMPOSITION family；本独立 semantic gold 不会把 recolor/global transform 错标为 composition。", "", "## Focused diagnostics", "", *[f"- {name}: {values}" for name, values in focused_metrics.items()], "", "## Failure taxonomy", "", *[f"- {key}: {value}" for key, value in sorted(errors.items())], "", "Gold、task IDs、raw grids、test solutions、raw responses 与逐题 IR 均保留在 ignored private artifacts。"]
    (ROOT / "reports/grid_semantic_recognition_v1.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (ROOT / "experiments/experiments.csv").open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["experiment_id", "date", "git_commit", "solver", "representation", "search_method", "llm_model", "candidate_budget", "validation_split", "tasks_solved", "accuracy", "runtime_seconds", "gpu_hours", "notes"])
        writer.writerow({"experiment_id": "GRID_SEMANTIC_RECOGNITION_V1", "date": str(date.today()), "git_commit": "pending", "solver": "Qwen semantic recognizer", "representation": "ARCSEMANTICIRV1", "search_method": "top-1 greedy recognition only", "llm_model": "qwen-lm/qwen-3/Transformers/8b/1", "candidate_budget": 1, "validation_split": "development_only_30", "tasks_solved": sum(bool(item["semantic_success"]) for item in scored.values()), "accuracy": metrics["semantic_ir_success"], "runtime_seconds": generation_seconds, "gpu_hours": generation_seconds / 3600, "notes": json.dumps({"recognition_status": status, "downstream_used": False}, sort_keys=True)})
    print(json.dumps({"status": status, "metrics": metrics}, sort_keys=True))


if __name__ == "__main__": main()
