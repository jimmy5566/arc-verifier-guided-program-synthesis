"""Local-only scorer for the frozen DIRECT_GRID_SOLVE_BASELINE_V1 artifact.

This file is intentionally excluded from the Kaggle source attachment.  It
opens the development targets only after verifying that complete prediction
bytes have been frozen and hashed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from time import perf_counter
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluation.direct_grid_scorer import recognition_overlap, score_direct_task_prediction


FROZEN_HASH = "a2f8fb66af4b83ec09bd2f3f2bf3fc5e948a6f7a6839898b5dbf26c6f77d1fc8"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _recognition_success(predictions: Path, gold: Path) -> dict[str, bool]:
    """Score the already-frozen V1 semantic artifact after direct freeze."""
    from evaluation.semantic_ir_scorer import score

    semantic = json.loads(predictions.read_text(encoding="utf-8"))
    oracle = json.loads(gold.read_text(encoding="utf-8"))["gold"]
    if semantic.get("status") != "PREDICTIONS_FROZEN_BEFORE_GOLD_SCORING" or set(semantic["records"]) != set(oracle):
        raise ValueError("frozen Recognition V1 artifact/gold mismatch")
    return {task_id: bool(score(record.get("prediction"), oracle[task_id])["semantic_success"]) for task_id, record in semantic["records"].items()}


def _render_report(result: dict[str, Any]) -> str:
    metrics, overlap = result["metrics"], result["recognition_overlap"]
    return "\n".join([
        "# DIRECT GRID SOLVE BASELINE V1", "",
        "D0 使用 Qwen3-8B，只读取完整 train input/output grids 与 test input；每题一次贪心调用，输出至多两个网格候选。Kaggle 推理未导入 target、oracle、Semantic IR、capability library、compiler、executor、verifier 或 search；本地仅在预测文件冻结后进行 exact scoring。", "",
        "## Direct 指标", "",
        *[f"- {key}: **{value:.1%}**" if isinstance(value, float) else f"- {key}: **{value}**" for key, value in metrics.items()], "",
        "## 与 Recognition V1 的逐题交集", "",
        *[f"- {key}: {value}" for key, value in overlap.items()], "",
        "## 解释", "",
        f"- 状态：**{result['direct_solver_status']}**", f"- Semantic IR 可能形成接口限制的信号：**{result['semantic_interface_signal']}**", f"- 相对冻结 executable capability coverage（1/30）的 direct-only exact 增益：**+{result['direct_advantage_cases']} task(s)**", "",
        f"- Kaggle kernel wall runtime：**{result['kernel_wall_seconds']:.1f}s**；所有 worker generation 时间之和：**{result['generation_seconds_sum']:.1f}s**。", "",
        "这是本地 frozen 30-task development exact baseline，不能等同于 Kaggle Public LB 百分比，也不构成模型能力上限。", "",
        "## Failure taxonomy", "",
        *[f"- {key}: {value}" for key, value in result["failure_taxonomy"].items()], "",
        "公开聚合结果不包含 task IDs、grids、targets、raw responses 或 oracle records。", "",
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--challenge-path", type=Path, required=True)
    parser.add_argument("--solutions-path", type=Path, required=True)
    parser.add_argument("--forensics-private", type=Path, required=True)
    parser.add_argument("--capability-private", type=Path, required=True)
    parser.add_argument("--recognition-predictions", type=Path, required=True)
    parser.add_argument("--recognition-oracle", type=Path, required=True)
    parser.add_argument("--kernel-log", type=Path, required=True)
    parser.add_argument("--result", type=Path, default=ROOT / "experiments/results/DIRECT_GRID_SOLVE_BASELINE_V1.json")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/direct_grid_solve_baseline_v1.md")
    args = parser.parse_args()
    started = perf_counter()

    # The direct artifact is fully parsed and cryptographically identified
    # before any path capable of supplying an ARC target is opened.
    frozen_bytes = args.predictions.read_bytes()
    prediction_hash = hashlib.sha256(frozen_bytes).hexdigest()
    frozen = json.loads(frozen_bytes)
    if frozen.get("status") != "PREDICTIONS_FROZEN_BEFORE_EXACT_SCORING":
        raise ValueError("exact scoring requires a frozen direct prediction artifact")
    if frozen.get("task_ids_hash") != FROZEN_HASH or len(frozen.get("records", {})) != 30:
        raise ValueError("direct artifact does not contain the exact frozen cohort")
    task_ids = set(frozen["records"])
    if any(not isinstance(item, dict) for item in frozen["records"].values()):
        raise ValueError("invalid frozen record")
    # Kaggle's progress bars occasionally contain byte fragments outside a
    # strict UTF-8 sequence. They are only human stderr payloads; replacing
    # them preserves the JSON event timings used for wall-runtime reporting.
    log_records = json.loads(args.kernel_log.read_bytes().decode("utf-8", errors="replace"))
    kernel_wall_seconds = max(float(item["time"]) for item in log_records) if log_records else 0.0

    # Gold access begins only here.
    from arc.io import load_challenges, load_solutions

    challenges, solutions = load_challenges(args.challenge_path), load_solutions(args.solutions_path)
    if not task_ids <= set(challenges) or not task_ids <= set(solutions):
        raise ValueError("development challenge/target task sets do not cover frozen predictions")
    forensics = {item["task_id"]: item["classification"] for item in json.loads(args.forensics_private.read_text(encoding="utf-8"))["records"]}
    if set(forensics) != task_ids:
        raise ValueError("private forensic task set mismatch")
    capability_records = json.loads(args.capability_private.read_text(encoding="utf-8"))["records"]
    structured_exact = {item["task_id"] for item in capability_records if item.get("test_exact_capabilities")}
    if not structured_exact <= task_ids:
        raise ValueError("private capability task set mismatch")
    per_task = {}
    for task_id in sorted(task_ids):
        record, challenge = frozen["records"][task_id], challenges[task_id]
        if len(challenge["test"]) != len(solutions[task_id]):
            raise ValueError("test input/target count mismatch")
        item = score_direct_task_prediction(record, solutions[task_id], [example["input"] for example in challenge["test"]], forensics[task_id])
        per_task[task_id] = item
    recognition = _recognition_success(args.recognition_predictions, args.recognition_oracle)
    if set(recognition) != task_ids:
        raise ValueError("recognition/direct task set mismatch")
    overlap = recognition_overlap(recognition, {task_id: item.any_of_2_exact for task_id, item in per_task.items()})
    total = len(per_task)
    mean = lambda field: sum(float(getattr(item, field)) for item in per_task.values()) / total
    exact = sum(item.any_of_2_exact for item in per_task.values())
    direct_exact_ids = {task_id for task_id, item in per_task.items() if item.any_of_2_exact}
    direct_only_ids = direct_exact_ids - structured_exact
    metrics = {
        "valid_grid_rate": mean("attempt_1_valid"), "top1_test_exact": mean("top1_exact"), "any_of_2_test_exact": mean("any_of_2_exact"),
        "output_shape_correct": mean("correct_dimensions"), "color_set_valid": mean("color_set_valid"),
        "mean_cell_accuracy": mean("cell_accuracy"), "mean_changed_cell_precision": mean("changed_precision"), "mean_changed_cell_recall": mean("changed_recall"),
    }
    status = "DIRECT_SOLVER_STRONG" if metrics["any_of_2_test_exact"] >= .25 else "DIRECT_SOLVER_PARTIAL" if metrics["any_of_2_test_exact"] >= .15 else "DIRECT_SOLVER_WEAK" if metrics["any_of_2_test_exact"] >= .05 else "DIRECT_SOLVER_VERY_WEAK"
    signal = "YES" if overlap["recognition_fail_direct_exact"] >= 3 else "PARTIAL" if overlap["recognition_fail_direct_exact"] else "NO"
    result = {
        "experiment_id": "DIRECT_GRID_SOLVE_BASELINE_V1", "status": "COMPLETE_SCORED_AFTER_PREDICTION_FREEZE", "task_count": total,
        "model": "Qwen3-8B", "backend": "Transformers BF16", "condition": "D0_RAW_GRIDS_ONLY", "public_heldout_inference": False, "web_access": False,
        "cohort_task_ids_hash": FROZEN_HASH, "prediction_freeze_hash": prediction_hash,
        "recognition_artifact": "GRID_SEMANTIC_RECOGNITION_V1", "recognition_success_count": sum(recognition.values()),
        "metrics": metrics, "recognition_overlap": overlap, "direct_exact_tasks": exact, "structured_capability_representable": len(structured_exact),
        "direct_advantage_cases": len(direct_only_ids), "direct_solver_status": status, "semantic_interface_signal": signal,
        "failure_taxonomy": dict(sorted(Counter(item.failure for item in per_task.values() if item.failure).items())),
        "prompt_preflight": frozen["preflight"], "context_window": 32768,
        "generation_seconds_sum": sum(float(record.get("generation_seconds", 0.0)) for record in frozen["records"].values()), "kernel_wall_seconds": kernel_wall_seconds,
        "scoring_seconds": perf_counter() - started,
        "leakage_audit": ["Direct prediction bytes and SHA256 were established before target files were opened.", "Kaggle inference source only uses local model files plus challenge train pairs and test inputs.", "No public/held-out/challenge inference, web access, target, oracle, task label, semantic IR, capability library, compiler, executor, verifier, or search dependency is present in the inference runner."],
    }
    args.result.parent.mkdir(parents=True, exist_ok=True); args.report.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.report.write_text(_render_report(result), encoding="utf-8")
    columns = ["experiment_id", "date", "git_commit", "solver", "representation", "search_method", "llm_model", "candidate_budget", "validation_split", "tasks_solved", "accuracy", "runtime_seconds", "gpu_hours", "notes"]
    csv_path = ROOT / "experiments/experiments.csv"
    with csv_path.open(encoding="utf-8", newline="") as handle:
        existing_rows = [row for row in csv.DictReader(handle) if row["experiment_id"] != "DIRECT_GRID_SOLVE_BASELINE_V1"]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns); writer.writeheader(); writer.writerows(existing_rows)
        writer.writerow({"experiment_id": "DIRECT_GRID_SOLVE_BASELINE_V1", "date": str(date.today()), "git_commit": "pending", "solver": "Qwen3-8B direct grid", "representation": "raw grids only", "search_method": "one greedy call, at most two returned candidates", "llm_model": "qwen-lm/qwen-3/Transformers/8b/1", "candidate_budget": 2, "validation_split": "frozen_development_30", "tasks_solved": exact, "accuracy": metrics["any_of_2_test_exact"], "runtime_seconds": result["generation_seconds_sum"], "gpu_hours": result["generation_seconds_sum"] / 3600, "notes": json.dumps({"status": status, "prediction_freeze_hash": prediction_hash}, sort_keys=True)})
    print(json.dumps({"status": result["status"], "any_of_2_exact": exact, "prediction_freeze_hash": prediction_hash}, sort_keys=True))


if __name__ == "__main__":
    main()
