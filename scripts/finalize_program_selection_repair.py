"""Validate and publish the safe aggregate from the frozen S1--S4 run."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = ROOT / "experiments" / "results" / "PROGRAM_SELECTION_REPAIR_V1.json"
REPORT_PATH = ROOT / "reports" / "program_selection_repair_v1.md"
LEDGER_PATH = ROOT / "experiments" / "experiments.csv"


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def stable_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def contains_forbidden_key(value: object, forbidden: set[str]) -> bool:
    if isinstance(value, dict):
        return any(key in forbidden or contains_forbidden_key(item, forbidden) for key, item in value.items())
    if isinstance(value, list):
        return any(contains_forbidden_key(item, forbidden) for item in value)
    return False


def validate(result: dict[str, Any], frozen: dict[str, Any], benchmark: dict[str, Any]) -> None:
    if result.get("experiment_id") != "PROGRAM_SELECTION_REPAIR_V1":
        raise ValueError("unexpected experiment identifier")
    if result.get("status") != "COMPLETE_FROZEN_ONE_RUN_PER_CONDITION":
        raise ValueError("formal one-pass run did not complete")
    if result.get("arc_data_used") or result.get("arc_solutions_used") or result.get("new_arc_inference"):
        raise ValueError("this semantic-interface experiment must not use ARC")
    if result.get("case_count") != 60 or benchmark.get("case_count") != 60:
        raise ValueError("requires exactly the frozen 60-case benchmark")
    if any(result.get(key) != frozen.get(key) for key in ("benchmark_hash", "registry_hash", "schema_hash")):
        raise ValueError("result/config benchmark identity mismatch")
    if result.get("frozen_config_sha256") != stable_hash(frozen):
        raise ValueError("result was not generated from the frozen config")
    required = ("S1_MINIMALITY_AWARE_SELECTION", "S2_FAMILY_FIRST_SELECTION", "S3_CONTRASTIVE_SKELETON_RANKING", "S4_FAMILY_FILTERED_CONTRASTIVE_MINIMALITY")
    if tuple(result.get("conditions", {}).keys()) != required:
        raise ValueError("all and only frozen S1--S4 conditions are required")
    forbidden_public_keys = {"raw_response", "materialized_program", "canonical_program", "expected_macro_ids", "literal_values"}
    if contains_forbidden_key(result, forbidden_public_keys):
        raise ValueError("aggregate result contains raw/canonical program material")
    for name in required:
        item = result["conditions"][name]
        if item.get("case_count") != 60:
            raise ValueError(f"{name} has incomplete case coverage")
        if item["funnel"].get("compile_valid", 0) > 60:
            raise ValueError(f"{name} has invalid compile count")
        if item.get("semantic_success", 0) > 60:
            raise ValueError(f"{name} has invalid semantic count")


def row(name: str, item: dict[str, Any]) -> str:
    return "| " + " | ".join((
        name,
        f"{item['semantic_success']}/60 ({pct(item['semantic_success_rate'])})",
        str(item["canonical_exact"]),
        str(item["equivalent_alternate"]),
        str(item["over_transformed"]),
        str(item["wrong_operation"]),
        str(item["wrong_parameter"]),
        f"{item['funnel']['compile_valid']}/60 ({pct(item['compile_valid_rate'])})",
    )) + " |"


def report(result: dict[str, Any]) -> str:
    baseline = result["s0_frozen_baseline"]
    conditions = result["conditions"]
    s0 = {
        "semantic_success": baseline["semantic_success"], "semantic_success_rate": baseline["semantic_success_rate"],
        "canonical_exact": 28, "equivalent_alternate": 3, "over_transformed": baseline["over_transformed"],
        "wrong_operation": baseline["wrong_operation"], "wrong_parameter": baseline["wrong_parameter"],
        "funnel": {"compile_valid": baseline["compile_valid"]}, "compile_valid_rate": 1.0,
    }
    lines = [
        "# Program Selection Repair V1",
        "",
        "- 冻结的 60-case Macro API semantic-interface benchmark；没有 ARC 数据、ARC solution 或新 ARC inference。",
        "- S0 仅引用冻结 C3 审计；S1–S4 在同一份预冻结配置下各执行一次。",
        "- 评分继续使用 deterministic C3 semantic scorer；没有 LLM judge。公开结果不含 raw model response 或 program material。",
        "",
        "## 总表",
        "",
        "| Condition | Semantic Success | Exact Canonical | Equivalent Alternate | Over-Transform | Wrong Operation | Wrong Parameter | Compile |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
        row("S0 C3 frozen", s0),
        *[row(name, conditions[name]) for name in conditions],
        "",
        "## 选择与回归",
        "",
    ]
    for name, item in conditions.items():
        transitions = item["failure_transitions"]
        lines.extend((
            f"### {name}",
            "",
            f"- S0 OVER_TRANSFORMED → success: {transitions.get('S0_OVER_TRANSFORMED_TO_SUCCESS', 0)}",
            f"- S0 WRONG_OPERATION → success: {transitions.get('S0_WRONG_OPERATION_TO_SUCCESS', 0)}",
            f"- S0 WRONG_PARAMETER → success: {transitions.get('S0_WRONG_PARAMETER_TO_SUCCESS', 0)}",
            f"- S0 success retained / regressed: {item['s0_success_retained']} / {item['s0_success_regressed']} ({pct(item['semantic_regression_rate'])})",
            f"- generation calls: {item['generation_calls']}; selected extra semantic operations: {item['selection_diagnostics']['extra_semantic_operations']}",
            "",
        ))
    best = conditions[result["best_selection_pipeline"]]
    lines.extend((
        "## 决策",
        "",
        f"- best: {result['best_selection_pipeline']} — {best['semantic_success']}/60 ({pct(best['semantic_success_rate'])}); compile {pct(best['compile_valid_rate'])}。",
        f"- semantic status: {result['semantic_status']}；ARC_ORACLE_LADDER_READY: {'YES' if result['arc_oracle_ladder_ready'] else 'NO'}。",
        f"- worker→GPU: {result['worker_to_gpu_mapping']}；runtime: {result['run_wall_seconds']:.1f}s。",
    ))
    return "\n".join(lines) + "\n"


def update_ledger(result: dict[str, Any], git_commit: str) -> None:
    fields: list[str]
    with LEDGER_PATH.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = [entry for entry in reader if entry["experiment_id"] != result["experiment_id"]]
    best = result["conditions"][result["best_selection_pipeline"]]
    rows.append({
        "experiment_id": result["experiment_id"], "date": str(date.today()), "git_commit": git_commit,
        "solver": "Qwen3-8B semantic program selection repair", "representation": "C3 structured compiler-constrained Macro skeleton selection",
        "search_method": "frozen S1-S4 deterministic candidate selection", "llm_model": result["model"]["model_source"],
        "candidate_budget": 1, "validation_split": "frozen_macro_api_60_no_arc", "tasks_solved": 0, "accuracy": 0.0,
        "runtime_seconds": result["run_wall_seconds"], "gpu_hours": result["run_wall_seconds"] * 4 / 3600,
        "notes": json.dumps({"semantic_status": result["semantic_status"], "best": result["best_selection_pipeline"], "semantic_success": best["semantic_success"], "compile_valid": best["funnel"]["compile_valid"], "arc_oracle_ladder_ready": result["arc_oracle_ladder_ready"], "arc_data_used": False}, sort_keys=True),
    })
    with LEDGER_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="Downloaded Kaggle aggregate JSON")
    parser.add_argument("--git-commit", required=True)
    args = parser.parse_args()
    frozen = read(ROOT / "configs" / "PROGRAM_SELECTION_REPAIR_V1_FROZEN_CONFIG.json")
    benchmark = read(ROOT / "configs" / "macro_api_comprehension_benchmark_v1.json")
    result = read(args.input)
    validate(result, frozen, benchmark)
    if RESULT_PATH.exists() or REPORT_PATH.exists():
        raise FileExistsError("refusing to overwrite published experiment result/report")
    RESULT_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    REPORT_PATH.write_text(report(result), encoding="utf-8")
    update_ledger(result, args.git_commit)
    print(json.dumps({"result": str(RESULT_PATH), "report": str(REPORT_PATH), "best": result["best_selection_pipeline"], "status": result["semantic_status"]}))


if __name__ == "__main__":
    main()
