"""Validate and publish the safe aggregate from frozen Q1--Q4 retrieval."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "experiments" / "results" / "PARAMETER_SEMANTIC_RETRIEVAL_V1.json"
REPORT = ROOT / "reports" / "parameter_semantic_retrieval_v1.md"
LEDGER = ROOT / "experiments" / "experiments.csv"
CONDITIONS = ("Q1_DETERMINISTIC_ONTOLOGY_REPAIR", "Q2_LIKELIHOOD_PARAMETER_REPAIR", "Q3_CONSENSUS_PARAMETER_REPAIR", "Q4_CONSERVATIVE_HYBRID_REPAIR")


def read(path: Path) -> dict[str, Any]: return json.loads(path.read_text(encoding="utf-8"))
def stable_hash(value: object) -> str: return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
def pct(value: float | None) -> str: return "N/A" if value is None else f"{value * 100:.1f}%"


def forbidden(value: object) -> bool:
    keys = {"raw_response", "raw_responses", "materialized_program", "baseline_program", "canonical_program", "expected_macro_ids", "literal_values", "program", "candidate_scoring"}
    if isinstance(value, dict): return any(key in keys or forbidden(item) for key, item in value.items())
    if isinstance(value, list): return any(forbidden(item) for item in value)
    return False


def validate(result: Mapping[str, Any], frozen: Mapping[str, Any], benchmark: Mapping[str, Any]) -> None:
    if result.get("experiment_id") != "PARAMETER_SEMANTIC_RETRIEVAL_V1" or result.get("status") != "COMPLETE_FROZEN_ONE_RUN_PER_CONDITION": raise ValueError("incomplete or wrong experiment")
    if result.get("arc_data_used") or result.get("arc_solutions_used") or result.get("new_arc_inference"): raise ValueError("ARC use is forbidden")
    if result.get("case_count") != 60 or frozen.get("case_count") != 60: raise ValueError("requires 60 cases")
    if any(result.get(key) != frozen.get(key) for key in ("benchmark_hash", "registry_hash", "schema_hash")): raise ValueError("benchmark identity mismatch")
    if result.get("frozen_config_sha256") != stable_hash(frozen): raise ValueError("frozen config mismatch")
    if tuple(result.get("conditions", {})) != CONDITIONS: raise ValueError("all and only Q1-Q4 required")
    if result.get("worker_to_gpu_mapping") != {"0": 0, "1": 1, "2": 2, "3": 3}: raise ValueError("four-worker GPU mapping mismatch")
    if forbidden(result): raise ValueError("aggregate contains a forbidden raw/program field")
    for name in CONDITIONS:
        item = result["conditions"][name]
        if item.get("case_count") != 60 or item.get("compile_valid", 0) > 60 or item.get("parameter_failures_repaired", 0) > 17: raise ValueError(f"invalid condition counts: {name}")


def q4_slot_audit(*, checkpoints_root: Path, frozen: Mapping[str, Any], benchmark: Mapping[str, Any]) -> dict[str, dict[str, int | float]]:
    """Compute field-exact metrics only after Version 22 predictions froze.

    This corrects a presentation-level aggregate where a remaining *case*
    failure had been attributed to every field in that case.  It does not
    re-run inference or make any repair choice.
    """
    sys_path = str(ROOT / "src")
    import sys
    if sys_path not in sys.path: sys.path.insert(0, sys_path)
    from llm.compiler_aware_interface import skeleton_by_id
    from llm.parameter_grounding import choices_from_program
    cases = {str(item["case_id"]): item for item in benchmark["cases"]}
    fields = frozen["forensics"]["wrong_parameter_fields_by_case"]
    totals: dict[str, dict[str, int | float]] = {}
    root = checkpoints_root / "Q4_CONSERVATIVE_HYBRID_REPAIR"
    files = sorted(root.glob("case_*.json"))
    if len(files) != 60: raise ValueError("requires all 60 frozen Q4 checkpoints for slot audit")
    for path in files:
        record = read(path); case_id = str(record["case_id"])
        if case_id not in fields: continue
        baseline = frozen["s2_baseline_programs"][case_id]; skeleton = skeleton_by_id(str(baseline["skeleton_id"]))
        if skeleton is None: raise ValueError(f"unknown frozen skeleton for {case_id}")
        before = choices_from_program(skeleton, baseline["program"])
        after = choices_from_program(skeleton, record["program"])
        expected = choices_from_program(skeleton, cases[case_id]["canonical_program"])
        for field in fields[case_id]:
            item = totals.setdefault(field, {"s2_errors": 0, "patches": 0, "correct_patches": 0, "remaining_errors": 0})
            field_keys = [key for key in before if key.split(".", 1)[1] == field]
            item["s2_errors"] = int(item["s2_errors"]) + 1
            item["patches"] = int(item["patches"]) + sum(before[key] != after[key] for key in field_keys)
            item["correct_patches"] = int(item["correct_patches"]) + sum(before[key] != expected[key] and after[key] == expected[key] for key in field_keys)
            item["remaining_errors"] = int(item["remaining_errors"]) + int(any(after[key] != expected[key] for key in field_keys))
    for item in totals.values(): item["repair_rate"] = int(item["correct_patches"]) / int(item["s2_errors"]) if item["s2_errors"] else 0.0
    return {name: item for name, item in sorted(totals.items())}


def _condition_row(name: str, item: Mapping[str, Any]) -> str:
    return f"| {name} | {item['semantic_success']}/60 ({pct(item['semantic_success_rate'])}) | {item['parameter_failures_repaired']}/17 | {item['patch_count']} | {pct(item['patch_precision'])} | {item['s2_success_regressed']}/31 |"


def report(result: Mapping[str, Any]) -> str:
    q0, q4 = result["q0_s2_baseline"], result["conditions"]["Q4_CONSERVATIVE_HYBRID_REPAIR"]
    lines = [
        "# Parameter Semantic Retrieval V1", "",
        "- 冻结 60-case Macro API benchmark；没有 ARC 数据、ARC solution 或新 ARC inference。",
        "- 模型仅执行 local Qwen3-8B teacher-forced forward likelihood；无 completion generation、JSON schema 解析或自由参数生成。",
        "- 所有 Q1–Q4 预测冻结后，父进程才读取 semantic scorer/canonical program 进行离线评分；repair gate 从不接收它们。", "",
        "## Theoretical limit", "", "- parameter-only ceiling: 48/60 = 80.0%。", "",
        "## Conditions", "", "| Condition | Semantic Success | Repaired / 17 | Patches | Patch precision | Regression / 31 |", "| --- | --- | ---: | ---: | --- | ---: |",
        f"| Q0 S2 frozen | {q0['semantic_success']}/60 ({pct(q0['semantic_success_rate'])}) | 0/17 | 0 | N/A | N/A |",
    ]
    lines.extend(_condition_row(name, result["conditions"][name]) for name in CONDITIONS)
    lines.extend(("", "## Q4 Conservative Hybrid", "", f"- wrong-parameter remaining: {q4['wrong_parameter']}；patch count: {q4['patch_count']}；correct patches: {q4['correct_patches']}；incorrect patches: {q4['incorrect_patches']}；abstain failure cases: {q4['abstain_count']}。", f"- repair coverage: {pct(q4['repair_coverage'])}；S2 success preserved: {q4['s2_success_retained']}/31；compile-valid: {q4['compile_valid']}/60 ({pct(q4['compile_valid_rate'])})。", "", "## Per-slot Q4", "", "| Field | S2 errors | Patches | Correct patches | Remaining errors | Repair rate |", "| --- | ---: | ---: | ---: | ---: | --- |"))
    slot_metrics = result.get("q4_post_freeze_slot_audit", q4["per_parameter_field"])
    for field, item in slot_metrics.items(): lines.append(f"| {field} | {item['s2_errors']} | {item['patches']} | {item['correct_patches']} | {item['remaining_errors']} | {pct(item['repair_rate'])} |")
    orientation = slot_metrics.get("orientation", {"s2_errors": 0, "patches": 0, "correct_patches": 0})
    recall = orientation["correct_patches"] / orientation["s2_errors"] if orientation["s2_errors"] else 0.0
    precision = orientation["correct_patches"] / orientation["patches"] if orientation["patches"] else None
    best = result["conditions"][result["best_parameter_source"]]
    lines.extend(("", "## Orientation", "", f"- candidate accuracy proxy (correct patch recall): {pct(recall)}；repair precision: {pct(precision)}；recall: {pct(recall)}。", "", "## Decision", "", f"- best source: {result['best_parameter_source']}；strict-schema problem solved: 是（本实验无自由 JSON schema）。", f"- SelectiveParameterRepairV2: {'是' if result['selective_parameter_repair_v2_enabled'] else ('部分' if best['parameter_failures_repaired'] else '否')}；status: {result['parameter_layer_status']}。", f"- overall semantic success: 51.7% → {pct(best['semantic_success_rate'])}；reached 80% ceiling: {'YES' if best['semantic_success'] == 48 and best['s2_success_regressed'] == 0 else 'NO'}。", f"- worker→GPU: {result['worker_to_gpu_mapping']}；runtime: {result['run_wall_seconds']:.1f}s。", ""))
    return "\n".join(lines)


def update_ledger(result: Mapping[str, Any], commit: str) -> None:
    with LEDGER.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle); fields = list(reader.fieldnames or []); rows = [item for item in reader if item["experiment_id"] != result["experiment_id"]]
    best = result["conditions"][result["best_parameter_source"]]
    rows.append({"experiment_id": result["experiment_id"], "date": str(date.today()), "git_commit": commit, "solver": "Qwen3-8B finite parameter semantic retrieval", "representation": "registry ontology + canonical candidate IDs + local-only repair", "search_method": "Q1-Q4 frozen deterministic/forward-likelihood/contrastive retrieval", "llm_model": result["model"]["model_source"], "candidate_budget": 0, "validation_split": "frozen_macro_api_60_no_arc", "tasks_solved": 0, "accuracy": 0.0, "runtime_seconds": result["run_wall_seconds"], "gpu_hours": result["run_wall_seconds"] * 4 / 3600, "notes": json.dumps({"best": result["best_parameter_source"], "status": result["parameter_layer_status"], "repaired": best["parameter_failures_repaired"], "regression": best["s2_success_regressed"], "arc_data_used": False}, sort_keys=True)})
    with LEDGER.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--input", type=Path, required=True); parser.add_argument("--git-commit", required=True); parser.add_argument("--checkpoints-root", type=Path); parser.add_argument("--replace-after-freeze-audit", action="store_true"); args = parser.parse_args()
    result, frozen, benchmark = read(args.input), read(ROOT / "configs" / "PARAMETER_SEMANTIC_RETRIEVAL_V1_FROZEN_CONFIG.json"), read(ROOT / "configs" / "macro_api_comprehension_benchmark_v1.json")
    validate(result, frozen, benchmark)
    if RESULT.exists() or REPORT.exists():
        if not args.replace_after_freeze_audit: raise FileExistsError("refusing to overwrite published retrieval result/report")
        if args.checkpoints_root is None: raise ValueError("replacement requires frozen checkpoint root")
    if args.checkpoints_root is not None:
        result["q4_post_freeze_slot_audit"] = q4_slot_audit(checkpoints_root=args.checkpoints_root, frozen=frozen, benchmark=benchmark)
    RESULT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8"); REPORT.write_text(report(result), encoding="utf-8"); update_ledger(result, args.git_commit)
    print(json.dumps({"result": str(RESULT), "report": str(REPORT), "best": result["best_parameter_source"], "status": result["parameter_layer_status"]}))


if __name__ == "__main__": main()
