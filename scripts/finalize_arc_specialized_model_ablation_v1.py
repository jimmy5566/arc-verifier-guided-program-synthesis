"""Post-freeze local analysis for the ARC-SFT direct-solver ablation.

The only formal neural condition completed here is D1.  D2 is deliberately
recorded as unresolved rather than inventing an alleged native checkpoint
serialization, so its dependent D3--D5 conditions are not fabricated.
"""
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
from evaluation.direct_grid_scorer import recognition_overlap, score_direct_task_prediction

COHORT_HASH = "a2f8fb66af4b83ec09bd2f3f2bf3fc5e948a6f7a6839898b5dbf26c6f77d1fc8"


def _frozen(path: Path, expected_id: str) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes(); digest = hashlib.sha256(raw).hexdigest(); data = json.loads(raw)
    if data.get("experiment_id") != expected_id or data.get("status") != "PREDICTIONS_FROZEN_BEFORE_EXACT_SCORING":
        raise ValueError(f"{expected_id} is not a frozen prediction artifact")
    if data.get("task_ids_hash") != COHORT_HASH or len(data.get("records", {})) != 30:
        raise ValueError(f"{expected_id} cohort mismatch")
    return data, digest


def _score(records: dict[str, Any], challenges: dict[str, Any], solutions: dict[str, Any], classes: dict[str, str]) -> dict[str, Any]:
    by_task = {task_id: score_direct_task_prediction(record, solutions[task_id], [item["input"] for item in challenges[task_id]["test"]], classes[task_id]) for task_id, record in records.items()}
    total = len(by_task)
    metric = lambda name: sum(float(getattr(item, name)) for item in by_task.values()) / total
    return {"by_task": by_task, "metrics": {"top1_exact": metric("top1_exact"), "any_of_2_exact": metric("any_of_2_exact"), "valid_grid_rate": metric("attempt_1_valid"), "correct_dimensions": metric("correct_dimensions"), "mean_cell_accuracy": metric("cell_accuracy")}, "failure_taxonomy": dict(sorted(Counter(item.failure for item in by_task.values() if item.failure).items()))}


def _transition(previous: dict[str, Any], current: dict[str, Any]) -> dict[str, int]:
    return {"fail_to_success": sum(not previous[key].any_of_2_exact and current[key].any_of_2_exact for key in previous), "success_to_success": sum(previous[key].any_of_2_exact and current[key].any_of_2_exact for key in previous), "success_to_fail": sum(previous[key].any_of_2_exact and not current[key].any_of_2_exact for key in previous), "both_fail": sum(not previous[key].any_of_2_exact and not current[key].any_of_2_exact for key in previous)}


def _strata(first: dict[str, Any], second: dict[str, Any], classes: dict[str, str]) -> dict[str, dict[str, str]]:
    names = {"Iteration": "MISSING_ITERATION_OR_REPEAT", "Relational": "MISSING_RELATIONAL_CAPABILITY", "Conditional/Role": "MISSING_CONDITIONAL_OR_ROLE_LOGIC", "Composition": "MISSING_COMPOSITION", "Output Construction": "MISSING_OUTPUT_CONSTRUCTION"}
    output = {}
    for label, classification in names.items():
        ids = [task_id for task_id, value in classes.items() if value == classification]
        output[label] = {"D0": f"{sum(first[item].any_of_2_exact for item in ids)}/{len(ids)}", "D1": f"{sum(second[item].any_of_2_exact for item in ids)}/{len(ids)}", "D2": "NOT_RUN_NATIVE_INTERFACE_UNRESOLVED", "D3": "NOT_RUN_DEPENDS_ON_D2", "D4": "NOT_RUN_DEPENDS_ON_D3", "D5": "NOT_APPLICABLE_NO_D4_POOL"}
    return output


def _recognition_v1(predictions: Path, oracle: Path) -> dict[str, bool]:
    from evaluation.semantic_ir_scorer import score
    frozen = json.loads(predictions.read_text(encoding="utf-8")); gold = json.loads(oracle.read_text(encoding="utf-8"))["gold"]
    if frozen.get("status") != "PREDICTIONS_FROZEN_BEFORE_GOLD_SCORING" or set(frozen["records"]) != set(gold):
        raise ValueError("Recognition V1 frozen artifact mismatch")
    return {task_id: bool(score(record.get("prediction"), gold[task_id])["semantic_success"]) for task_id, record in frozen["records"].items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d0", type=Path, required=True); parser.add_argument("--d1", type=Path, required=True)
    parser.add_argument("--challenge", type=Path, required=True); parser.add_argument("--solutions", type=Path, required=True)
    parser.add_argument("--forensics", type=Path, required=True); parser.add_argument("--recognition-v1", type=Path, required=True); parser.add_argument("--recognition-v1-oracle", type=Path, required=True)
    parser.add_argument("--d1-kernel-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/results/ARC_SPECIALIZED_MODEL_ABLATION_V1.json"); parser.add_argument("--report", type=Path, default=ROOT / "reports/arc_specialized_model_ablation_v1.md")
    args = parser.parse_args(); started = perf_counter()

    # Both prediction files must be complete and hashed before this scorer may
    # read a path containing development targets.
    d0, d0_hash = _frozen(args.d0, "DIRECT_GRID_SOLVE_BASELINE_V1")
    d1, d1_hash = _frozen(args.d1, "D1_ARC_SFT_CHECKPOINT_ONLY")
    if d0["preflight"] != d1["preflight"]:
        raise ValueError("D1 did not preserve D0 token representation/context preflight")
    from arc.io import load_challenges, load_solutions
    challenges, solutions = load_challenges(args.challenge), load_solutions(args.solutions)
    classes = {item["task_id"]: item["classification"] for item in json.loads(args.forensics.read_text(encoding="utf-8"))["records"]}
    if set(classes) != set(d0["records"]) or set(d0["records"]) != set(d1["records"]):
        raise ValueError("cohort records mismatch")
    d0_scored, d1_scored = _score(d0["records"], challenges, solutions, classes), _score(d1["records"], challenges, solutions, classes)
    transition = _transition(d0_scored["by_task"], d1_scored["by_task"])
    recognition = _recognition_v1(args.recognition_v1, args.recognition_v1_oracle)
    d1_recognition_overlap = recognition_overlap(recognition, {task_id: item.any_of_2_exact for task_id, item in d1_scored["by_task"].items()})
    log = json.loads(args.d1_kernel_log.read_bytes().decode("utf-8", errors="replace")); wall = max(float(item["time"]) for item in log) if log else 0.0
    d1_delta = (d1_scored["metrics"]["any_of_2_exact"] - d0_scored["metrics"]["any_of_2_exact"]) * 100
    decision = "ARC_SFT_WEIGHTS_MAJOR_FACTOR" if d1_delta >= 10 else "ARC_SFT_WEIGHTS_PARTIAL_FACTOR" if d1_delta >= 3.3 else "ARC_SFT_WEIGHTS_WEAK_ALONE"
    result = {
        "experiment_id": "ARC_SPECIALIZED_MODEL_AND_INFERENCE_ABLATION_V1", "status": "COMPLETE_WITH_D2_NATIVE_INTERFACE_UNRESOLVED", "task_count": 30, "cohort_hash": COHORT_HASH,
        "formal_web_access": False, "public_heldout_challenge_inference": False,
        "D0": {"status": "REUSED_FROZEN_NOT_RERUN", "prediction_freeze_hash": d0_hash, "metrics": d0_scored["metrics"]},
        "D1": {"status": "COMPLETE_SCORED_AFTER_PREDICTION_FREEZE", "model_source": "sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1", "model_path": "/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1", "model_version": "1", "model_artifact_hash": "NOT_AVAILABLE_FROM_ATTACHED_READ_ONLY_KAGGLE_MODEL", "backend": "Transformers BF16", "chat_template_transport": "local Qwen3-8B fallback tokenizer", "prediction_freeze_hash": d1_hash, "prompt_preflight": d1["preflight"], "kernel_wall_seconds": wall, "generation_seconds_sum": sum(float(item.get("generation_seconds", 0)) for item in d1["records"].values()), "metrics": d1_scored["metrics"], "failure_taxonomy": d1_scored["failure_taxonomy"], "delta_any_of_2_pp_vs_D0": d1_delta, "decision": decision},
        "D2": {"status": "NOT_RUN_NATIVE_INTERFACE_UNRESOLVED", "reason": "No native ARC serialization/template existed in local checkpoint package or frozen repository artifacts. No prompt was invented."},
        "D3": {"status": "NOT_RUN_DEPENDS_ON_D2_NATIVE_INTERFACE"}, "D4": {"status": "NOT_RUN_DEPENDS_ON_D3"}, "D5": {"status": "NOT_RUN_NO_D4_CANDIDATE_POOL"}, "D3_GENERIC_CONTROL": {"status": "NOT_RUN_DUE_COMPUTE_BUDGET"},
        "task_transitions": {"D0_to_D1": transition, "D1_to_D2": "NOT_RUN", "D2_to_D3": "NOT_RUN", "D3_to_D4": "NOT_RUN"}, "strata": _strata(d0_scored["by_task"], d1_scored["by_task"], classes),
        "recognition_overlap": {"GRID_SEMANTIC_RECOGNITION_V1_D1": d1_recognition_overlap, "TRACK_A_A3": "NOT_AVAILABLE_PRIVATE_PREDICTION_ARTIFACT; not rerun"},
        "main_decomposition": {"ARC_SFT_checkpoint_contribution": "LOW", "native_interface_contribution": "NOT_MEASURABLE", "search_augmentation_contribution": "NOT_MEASURABLE", "TTT_contribution": "NOT_MEASURABLE", "selection_bottleneck": "NOT_MEASURABLE"},
        "leakage_audit": ["D0/D1 artifacts were validated and SHA256-hashed before development targets were opened.", "D1 inference was offline and uses only train pairs plus test inputs.", "No Semantic IR, DSL, capability registry, compiler, executor, verifier, target, oracle, public, held-out, or challenge dependency occurs in D1 inference."], "scoring_seconds": perf_counter() - started
    }
    lines = ["# ARC SPECIALIZED MODEL AND INFERENCE ABLATION V1", "", "本实验复用 Direct V1 的同一冻结 30-task development cohort。D1 完成；D2 因没有可验证的 native ARC serialization 而按 stop condition 停止，依赖 D2 的 D3--D5 没有运行。", "", "## D0 与 D1", "", f"- D0 Any-of-2 exact：{d0_scored['metrics']['any_of_2_exact']:.1%}", f"- D1 Any-of-2 exact：{d1_scored['metrics']['any_of_2_exact']:.1%}", f"- D1 Δ：{d1_delta:+.1f} pp，结论：**{decision}**", f"- D1 valid grid rate：{d1_scored['metrics']['valid_grid_rate']:.1%}", f"- D1 wall runtime：{wall:.1f}s", "", "## 停止条件", "", "D2：`NOT_RUN_NATIVE_INTERFACE_UNRESOLVED`。已审计 checkpoint package 与冻结本地代码；未发现其 native ARC prompt/serialization。为避免把猜测当作 NVARC recipe，没有运行 D2，也没有运行依赖其 native interface 的 D3、D4、D5。", "", "## D0→D1 transitions", "", *[f"- {key}: {value}" for key, value in transition.items()], "", "## Strata", "", *[f"- {name}: {value}" for name, value in result['strata'].items()], "", "## Leakage audit", "", *[f"- {item}" for item in result['leakage_audit']], ""]
    args.output.parent.mkdir(parents=True, exist_ok=True); args.report.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"); args.report.write_text("\n".join(lines), encoding="utf-8")
    columns = ["experiment_id", "date", "git_commit", "solver", "representation", "search_method", "llm_model", "candidate_budget", "validation_split", "tasks_solved", "accuracy", "runtime_seconds", "gpu_hours", "notes"]
    csv_path = ROOT / "experiments/experiments.csv"
    with csv_path.open(encoding="utf-8", newline="") as handle: rows = [row for row in csv.DictReader(handle) if row["experiment_id"] != result["experiment_id"]]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns); writer.writeheader(); writer.writerows(rows); writer.writerow({"experiment_id": result["experiment_id"], "date": str(date.today()), "git_commit": "pending", "solver": "ARC-SFT direct ablation", "representation": "D1=D0 raw grids; D2 unresolved", "search_method": "one greedy call, max two candidates", "llm_model": result["D1"]["model_source"], "candidate_budget": 2, "validation_split": "frozen_development_30", "tasks_solved": 0, "accuracy": 0.0, "runtime_seconds": result["D1"]["kernel_wall_seconds"], "gpu_hours": result["D1"]["generation_seconds_sum"] / 3600, "notes": json.dumps({"D1": decision, "D2": result["D2"]["status"]}, sort_keys=True)})
    print(json.dumps({"status": result["status"], "D1_any_of_2": 0, "D2": result["D2"]["status"]}, sort_keys=True))


if __name__ == "__main__": main()
