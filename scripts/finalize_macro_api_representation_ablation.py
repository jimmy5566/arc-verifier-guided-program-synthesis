"""Publish safe aggregate/report/ledger artifacts for the frozen R1--R4 run."""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = {"raw_response", "repair_raw_response", "skeleton_raw_response", "final_raw_response", "records", "checkpoints", "provider_error", "contract_snippet"}
LABELS = {"R1_TYPED_COMPACT": "R1 Typed Compact", "R2_TYPED_EXAMPLES": "R2 Typed + Examples", "R3_TWO_STAGE_TYPED": "R3 Two-Stage", "R4_VERIFIER_GUIDED_API_REPAIR": "R4 Verifier-Guided Repair"}
CATEGORIES = {"SINGLE_MACRO": "Single Macro", "PARAMETER_CONTRACT": "Parameter Contract", "TWO_STEP_TYPED_CHAIN": "Two-Step Typed Chain", "THREE_STEP_COMPOSITION": "Three-Step Composition"}


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def walk(value: object):
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield key
            yield from walk(item)
    elif isinstance(value, list):
        for item in value:
            yield from walk(item)
    else:
        yield value


def validate(result: Mapping[str, Any], frozen: Mapping[str, Any]) -> None:
    if result.get("experiment_id") != "MACRO_API_REPRESENTATION_ABLATION_V1" or result.get("status") != "COMPLETE_FROZEN_ONE_RUN_PER_CONDITION":
        raise ValueError("not a completed R1--R4 aggregate")
    if result.get("case_count") != 60 or result.get("arc_data_used") is not False or result.get("arc_solutions_used") is not False or result.get("new_arc_inference") is not False:
        raise ValueError("API-only protocol violation")
    for field in ("benchmark_hash", "registry_hash", "schema_hash"):
        if result.get(field) != frozen.get(field):
            raise ValueError(f"frozen mismatch: {field}")
    if result.get("prompt_hashes") != frozen.get("prompt_hashes") or result.get("generation", {}).get("candidate_budget") != 1:
        raise ValueError("representation freeze mismatch")
    if set(result.get("conditions", ())) != set(LABELS):
        raise ValueError("incomplete conditions")
    if any(str(item) in FORBIDDEN for item in walk(result)):
        raise ValueError("private per-case field in public aggregate")
    if any("/kaggle/" in str(item) or "C:\\Users\\" in str(item) for item in walk(result)):
        raise ValueError("runtime path in public aggregate")


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def best_category_rates(conditions: Mapping[str, Any]) -> dict[str, float]:
    return {category: max(condition["category_metrics"][category]["rates"]["compile_valid"] for condition in conditions.values()) for category in CATEGORIES}


def report(result: Mapping[str, Any], tests: int) -> str:
    conditions = result["conditions"]
    rows = []
    for key in LABELS:
        condition = conditions[key]
        rows.append(f"| {LABELS[key]} | {pct(condition['rates']['schema_valid'])} | {pct(condition['rates']['argument_contract_valid'])} | {pct(condition['rates']['type_valid'])} | {pct(condition['rates']['parameter_valid'])} | {pct(condition['rates']['compile_valid'])} | {condition['generation_calls']} | {condition['prompt_tokens'] + condition['completion_tokens']} |")
    r3, r4 = conditions["R3_TWO_STAGE_TYPED"], conditions["R4_VERIFIER_GUIDED_API_REPAIR"]
    failures = conditions[result["best_api_pipeline"]]["failure_types"][:5]
    failure_lines = "\n".join(f"{i}. {name}: {count}" for i, (name, count) in enumerate(failures, 1)) or "1. 无"
    category_lines = "\n".join(f"- {CATEGORIES[key]}: {pct(value)}" for key, value in best_category_rates(conditions).items())
    return f"""# Macro API Representation Ablation V1

## 冻结协议

- Qwen3-8B、local Transformers、BF16、4×NVIDIA L4、Internet Off。
- R1/R2/R3 各一次正式 pass；R4 复用 R2 原始输出，对每个无效输出最多一次 deterministic-feedback repair。
- 未加载 ARC grids、ARC solutions 或 task predictions。benchmark 仍为冻结的 60-case API-only 集合。
- raw / repaired model responses 仅位于忽略的 Kaggle/local checkpoint；本报告和公开 JSON 仅含 aggregate。

## 条件比较

| Condition | Schema | API | Type | Parameter | Compile | Calls | Total tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

## R3 / R4 专项指标

- R3 skeleton type-valid: {pct(r3['skeleton']['rates']['type_valid'])}; final compile-valid: {pct(r3['rates']['compile_valid'])}。
- R4 raw type / compile: {pct(r4['raw']['rates']['type_valid'])} / {pct(r4['raw']['rates']['compile_valid'])}。
- R4 repair attempted / successful: {r4['repair_attempted']} / {r4['repair_successful']}；repair success rate: {pct(r4['repair_success_rate'])}。
- R4 final type / compile: {pct(r4['rates']['type_valid'])} / {pct(r4['rates']['compile_valid'])}。

## 各类别最佳 compile-valid rate

{category_lines}

## 主要失败模式（最佳条件）

{failure_lines}

## 结论

- BEST_API_PIPELINE: **{result['best_api_pipeline']}**。
- 状态: **{result['api_status']}**；推荐下一实验: **{result['recommended_next_experiment']}**。
- R0 baseline type / compile: {pct(result['r0_frozen_baseline']['type_valid_rate'])} / {pct(result['r0_frozen_baseline']['compile_valid_rate'])}。
- Local tests: **{tests} passed**。
"""


def append_ledger(path: Path, result: Mapping[str, Any], code_commit: str) -> None:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields, rows = list(reader.fieldnames or ()), list(reader)
    if not fields or any(row.get("experiment_id") == result["experiment_id"] for row in rows):
        raise ValueError("invalid or duplicate experiment ledger entry")
    best = result["conditions"][result["best_api_pipeline"]]
    completed = str(result.get("completed_at_utc", ""))
    row = {"experiment_id": result["experiment_id"], "date": completed[:10] if len(completed) >= 10 else datetime.now().date().isoformat(), "git_commit": code_commit, "solver": "Macro API representation ablation", "representation": "R1 typed compact; R2 examples; R3 skeleton/parameters; R4 one repair", "search_method": "one frozen pass per condition", "llm_model": result["model"]["model_source"], "candidate_budget": "1", "validation_split": "frozen API-only 60-case development benchmark", "tasks_solved": str(best["funnel"]["compile_valid"]), "accuracy": str(best["rates"]["compile_valid"]), "runtime_seconds": str(result["run_wall_seconds"]), "gpu_hours": "", "notes": json.dumps({"api_status": result["api_status"], "best": result["best_api_pipeline"], "r4_repair_success_rate": result["conditions"]["R4_VERIFIER_GUIDED_API_REPAIR"]["repair_success_rate"], "arc_data_used": False}, sort_keys=True)}
    with path.open("a", encoding="utf-8", newline="") as handle:
        csv.DictWriter(handle, fieldnames=fields).writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--frozen-config", type=Path, default=ROOT / "configs" / "MACRO_API_REPRESENTATION_ABLATION_V1_FROZEN_CONFIG.json")
    parser.add_argument("--public-result", type=Path, default=ROOT / "experiments" / "results" / "MACRO_API_REPRESENTATION_ABLATION_V1.json")
    parser.add_argument("--report", type=Path, default=ROOT / "reports" / "macro_api_representation_ablation_v1.md")
    parser.add_argument("--ledger", type=Path, default=ROOT / "experiments" / "experiments.csv")
    parser.add_argument("--tests-passed", type=int, required=True)
    parser.add_argument("--code-commit", required=True)
    args = parser.parse_args()
    result, frozen = read(args.aggregate), read(args.frozen_config)
    validate(result, frozen)
    if args.public_result.exists() or args.report.exists():
        raise FileExistsError("refusing to overwrite one-shot public result/report")
    args.public_result.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.report.write_text(report(result, args.tests_passed), encoding="utf-8")
    append_ledger(args.ledger, result, args.code_commit)


if __name__ == "__main__":
    main()
