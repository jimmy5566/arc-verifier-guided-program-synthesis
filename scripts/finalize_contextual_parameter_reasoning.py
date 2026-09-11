"""Validate/publish the downloaded one-shot contextual aggregate and report."""
from __future__ import annotations

import argparse
import csv
import json
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "experiments/results/CONTEXTUAL_PARAMETER_REASONING_V1.json"
REPORT = ROOT / "reports/contextual_parameter_reasoning_v1.md"
LEDGER = ROOT / "experiments/experiments.csv"


def pct(value): return "N/A" if value is None else f"{100 * value:.1f}%"


def report(result: dict) -> str:
    lines = ["# Contextual Parameter Reasoning V1", "", "- 60-case frozen Macro API benchmark；无 ARC data、ARC solution 或新的 ARC inference。", "- R1/R2/R3 的全部程序冻结后，父进程才使用 canonical semantic scorer 做评分。", "", "| Condition | Success | Remaining repaired / 12 | Q1 regression / 36 | Patches | Precision |", "| --- | ---: | ---: | ---: | ---: | --- |"]
    for key, value in result["conditions"].items(): lines.append(f"| {key} | {value['semantic_success']}/60 ({pct(value['semantic_success_rate'])}) | {value['remaining_parameter_repaired']}/12 | {value['q1_success_regressed']}/36 | {value['patch_count']} | {pct(value['patch_precision'])} |")
    best = result["conditions"][result["best"]]
    lines.extend(("", "## 最佳条件", "", f"- best: `{result['best']}`；status: `{result['contextual_parameter_layer_status']}`。", f"- 60.0% → {pct(best['semantic_success_rate'])}；新增解决 {best['semantic_success'] - 36}；距离 80% parameter-only ceiling 尚 {48 - best['semantic_success']} cases。", f"- remaining repair: {best['remaining_parameter_repaired']}/12；regression: {best['q1_success_regressed']}/36；abstain: {best['abstain_count']}；compile-valid is retained by frozen skeleton/local patch contract。", "", "## 字段与转移", "", f"- transitions: `{best['transition_matrix']}`", f"- per field: `{best['per_field']}`", "", "## 运行", "", f"- worker→GPU: `{result['worker_to_gpu_mapping']}`；wall: {result['run_wall_seconds']:.1f}s。", f"- warmup: `{result['model_warmup']}`", "", "## 泄漏审计", "", "- model input 仅含 instruction、family、frozen skeleton、target slot、legal candidates、relation features；不含 case ID、canonical、semantic label 或 expected candidate。", "- R1 映射由全局 Macro contract 和 relation ontology 驱动；R3 默认 KEEP Q1。", ""))
    return "\n".join(lines)


def update_ledger(result: dict, commit: str) -> None:
    with LEDGER.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle); fields = list(reader.fieldnames or []); rows = [item for item in reader if item["experiment_id"] != result["experiment_id"]]
    best = result["conditions"][result["best"]]
    rows.append({"experiment_id": result["experiment_id"], "date": str(date.today()), "git_commit": commit, "solver": "Qwen3-8B contextual parameter reasoning", "representation": "typed relation context IR + legal candidate likelihood + conservative local patch", "search_method": "R1 deterministic relation normalizer; R2 likelihood classifier; R3 priority gate", "llm_model": result.get("model", {}).get("model_source", "qwen-lm/qwen-3/Transformers/8b/1"), "candidate_budget": 1, "validation_split": "frozen_macro_api_60_no_arc", "tasks_solved": 0, "accuracy": 0.0, "runtime_seconds": result["run_wall_seconds"], "gpu_hours": result["run_wall_seconds"] * 2 / 3600, "notes": json.dumps({"best": result["best"], "status": result["contextual_parameter_layer_status"], "remaining_repaired": best["remaining_parameter_repaired"], "regression": best["q1_success_regressed"], "arc_data_used": False}, sort_keys=True)})
    with LEDGER.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--input", type=Path, required=True); parser.add_argument("--git-commit", required=True); args = parser.parse_args()
    if RESULT.exists() or REPORT.exists(): raise FileExistsError("refusing to overwrite published result/report")
    result = json.loads(args.input.read_text())
    if result.get("status") != "COMPLETE_FROZEN_ONE_RUN_PER_R1_R2_R3" or result.get("case_count") != 60 or set(result.get("conditions", ())) != {"R1_RELATIONAL_NORMALIZER", "R2_CONTEXTUAL_CANDIDATE_CLASSIFIER", "R3_CONSERVATIVE_CONTEXTUAL_REPAIR"}: raise ValueError("invalid aggregate")
    RESULT.write_text(json.dumps(result, indent=2) + "\n"); REPORT.write_text(report(result)); update_ledger(result, args.git_commit); print(json.dumps({"result": str(RESULT), "report": str(REPORT), "best": result["best"]}))


if __name__ == "__main__": main()
