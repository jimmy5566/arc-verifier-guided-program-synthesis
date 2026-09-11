"""Validate and publish the safe aggregate from frozen P1--P4 inference."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = ROOT / "experiments" / "results" / "PARAMETER_GROUNDING_REPAIR_V1.json"
REPORT_PATH = ROOT / "reports" / "parameter_grounding_repair_v1.md"
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
    if result.get("experiment_id") != "PARAMETER_GROUNDING_REPAIR_V1" or result.get("status") != "COMPLETE_FROZEN_ONE_RUN_PER_CONDITION":
        raise ValueError("unexpected or incomplete formal result")
    if result.get("arc_data_used") or result.get("arc_solutions_used") or result.get("new_arc_inference"):
        raise ValueError("parameter-grounding benchmark must not use ARC")
    if result.get("case_count") != 60 or benchmark.get("case_count") != 60:
        raise ValueError("requires frozen 60-case benchmark")
    if any(result.get(key) != frozen.get(key) for key in ("benchmark_hash", "registry_hash", "schema_hash")):
        raise ValueError("result/config benchmark identity mismatch")
    if result.get("frozen_config_sha256") != stable_hash(frozen):
        raise ValueError("result does not match frozen configuration")
    required = ("P1_NORMALIZED_SLOT_EXTRACTION", "P2_CONSTRAINED_PARAMETER_CHOICE", "P3_SLOTWISE_PARAMETER_SELECTION", "P4_SELECTIVE_PARAMETER_REPAIR")
    if tuple(result.get("conditions", {}).keys()) != required:
        raise ValueError("all and only frozen P1--P4 conditions are required")
    if contains_forbidden_key(result, {"raw_response", "raw_responses", "materialized_program", "baseline_program", "canonical_program", "expected_macro_ids", "literal_values", "program"}):
        raise ValueError("aggregate contains raw model response or program material")
    for name in required:
        item = result["conditions"][name]
        if item.get("case_count") != 60 or item["funnel"].get("compile_valid", 0) > 60:
            raise ValueError(f"invalid coverage/counts for {name}")


def report(result: dict[str, Any]) -> str:
    p0 = result["p0_s2_baseline"]; conditions = result["conditions"]; p4 = conditions["P4_SELECTIVE_PARAMETER_REPAIR"]
    lines = [
        "# Parameter Grounding Repair V1", "",
        "- 冻结的 60-case Macro API benchmark；无 ARC 数据、ARC solution 或新 ARC inference。",
        "- P0 仅引用冻结 S2；P1–P4 在同一份预冻结配置下各运行一次。family、skeleton、Macro ID 与 step order 对每个 case 均冻结为 S2 选择。",
        "- worker 只收到 instruction、冻结 S2 family/skeleton/program 和 registry-derived parameter contract；canonical program 与 semantic label 仅在四个 prediction 全部冻结后由父进程评分。",
        "", "## Phase A forensics", "",
        f"- S2 wrong-parameter: {result['forensics']['s2_wrong_parameter_cases']}；pure parameter failures: {result['forensics']['pure_parameter_failure_count']}；non-parameter structural failures: {result['forensics']['non_parameter_structural_failure_count']}。",
        f"- C3 static-slot interface confounds: {result['forensics']['c3_static_slot_interface_confounds']}。",
        "", "## Conditions", "",
        "| Condition | Semantic Success | Wrong Parameter | Repaired / 17 | Regression / 31 | Compile | Calls |",
        "| --- | --- | ---: | ---: | ---: | --- | ---: |",
        f"| P0 S2 frozen | {p0['semantic_success']}/60 ({pct(p0['semantic_success_rate'])}) | {p0['wrong_parameter']} | 0 | 0 | {p0['compile_valid']}/60 (100.0%) | 0 |",
    ]
    for name, item in conditions.items():
        lines.append(f"| {name} | {item['semantic_success']}/60 ({pct(item['semantic_success_rate'])}) | {item['wrong_parameter']} | {item['parameter_failures_repaired']} | {item['s2_success_regressed']} | {item['funnel']['compile_valid']}/60 ({pct(item['compile_valid_rate'])}) | {item['generation_calls']} |")
    lines.extend(("", "## P4 field repair", "", "| Parameter | S2 errors | P4 repaired | Repair rate |", "| --- | ---: | ---: | --- |"))
    for field, item in p4["per_parameter_field"].items():
        lines.append(f"| {field} | {item['s2_errors']} | {item['repaired']} | {pct(item['repair_rate'])} |")
    lines.extend(("", "## P4 transition focus", ""))
    for key in ("SEMANTIC_WRONG_PARAMETER_TO_SEMANTIC_EXACT_CANONICAL", "SEMANTIC_WRONG_PARAMETER_TO_SEMANTIC_EQUIVALENT_ALTERNATE", "SEMANTIC_WRONG_PARAMETER_TO_SEMANTIC_WRONG_PARAMETER", "SEMANTIC_WRONG_PARAMETER_TO_SEMANTIC_WRONG_OPERATION", "SEMANTIC_EXACT_CANONICAL_TO_SEMANTIC_EXACT_CANONICAL", "SEMANTIC_EXACT_CANONICAL_TO_SEMANTIC_WRONG_PARAMETER"):
        lines.append(f"- {key}: {p4['transition_matrix'].get(key, 0)}")
    best = conditions[result["best_parameter_pipeline"]]
    lines.extend(("", "## Decision", "", f"- best: {result['best_parameter_pipeline']} — repair {best['parameter_failures_repaired']}/17 ({pct(best['parameter_failure_repair_rate'])}); regression {best['s2_success_regressed']}/31; compile {pct(best['compile_valid_rate'])}。", f"- status: {result['parameter_layer_status']}；SelectiveParameterRepairV1 enabled: {result['selective_parameter_repair_v1_enabled']}。", f"- worker→GPU: {result['worker_to_gpu_mapping']}；runtime: {result['run_wall_seconds']:.1f}s。", ""))
    return "\n".join(lines)


def update_ledger(result: dict[str, Any], git_commit: str) -> None:
    with LEDGER_PATH.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle); fields = list(reader.fieldnames or [])
        rows = [row for row in reader if row["experiment_id"] != result["experiment_id"]]
    best = result["conditions"][result["best_parameter_pipeline"]]
    rows.append({
        "experiment_id": result["experiment_id"], "date": str(date.today()), "git_commit": git_commit,
        "solver": "Qwen3-8B parameter semantic grounding repair", "representation": "frozen S2 family/skeleton + typed parameter choices", "search_method": "P1-P4 one-pass parameter ablation", "llm_model": result["model"]["model_source"], "candidate_budget": 1, "validation_split": "frozen_macro_api_60_no_arc", "tasks_solved": 0, "accuracy": 0.0, "runtime_seconds": result["run_wall_seconds"], "gpu_hours": result["run_wall_seconds"] * 4 / 3600,
        "notes": json.dumps({"best": result["best_parameter_pipeline"], "parameter_layer_status": result["parameter_layer_status"], "repaired": best["parameter_failures_repaired"], "regression": best["s2_success_regressed"], "arc_data_used": False}, sort_keys=True),
    })
    with LEDGER_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--git-commit", required=True)
    args = parser.parse_args()
    frozen = read(ROOT / "configs" / "PARAMETER_GROUNDING_REPAIR_V1_FROZEN_CONFIG.json")
    benchmark = read(ROOT / "configs" / "macro_api_comprehension_benchmark_v1.json")
    result = read(args.input); validate(result, frozen, benchmark)
    if RESULT_PATH.exists() or REPORT_PATH.exists():
        raise FileExistsError("refusing to overwrite published repair result/report")
    RESULT_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    REPORT_PATH.write_text(report(result), encoding="utf-8")
    update_ledger(result, args.git_commit)
    print(json.dumps({"result": str(RESULT_PATH), "report": str(REPORT_PATH), "best": result["best_parameter_pipeline"], "status": result["parameter_layer_status"]}))


if __name__ == "__main__":
    main()
