"""Publish safe aggregate/report/ledger artifacts for frozen C1--C3 results."""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ("C1_COMPILER_AWARE_CATALOGUE", "C2_COMPILER_VALID_SKELETON", "C3_STRUCTURED_COMPILER_CONSTRAINED")
LABELS = {
    "C0_R2_TYPED_EXAMPLES": "C0 R2 Typed + Examples（冻结历史）",
    "C1_COMPILER_AWARE_CATALOGUE": "C1 Compiler-Aware Catalogue",
    "C2_COMPILER_VALID_SKELETON": "C2 Compiler-Valid Skeleton",
    "C3_STRUCTURED_COMPILER_CONSTRAINED": "C3 Structured Compiler-Constrained",
}
FORBIDDEN_KEYS = {"raw_response", "skeleton_raw_response", "final_raw_response", "materialized_program", "records", "checkpoints", "provider_error"}


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
    if result.get("experiment_id") != "MACRO_API_COMPILER_AWARE_ABLATION_V1" or result.get("status") != "COMPLETE_FROZEN_ONE_RUN_PER_CONDITION":
        raise ValueError("not a completed one-shot C1--C3 aggregate")
    if result.get("case_count") != 60 or any(result.get(field) is not False for field in ("arc_data_used", "arc_solutions_used", "new_arc_inference")):
        raise ValueError("API-only protocol violation")
    for key in ("benchmark_hash", "registry_hash", "schema_hash", "prompt_hashes", "model", "sampling", "generation"):
        if result.get(key) != frozen.get(key):
            raise ValueError(f"frozen mismatch: {key}")
    if set(result.get("conditions", ())) != set(CONDITIONS):
        raise ValueError("incomplete C1--C3 conditions")
    if result.get("c0_frozen_baseline", {}).get("funnel", {}).get("compile_valid") != 14:
        raise ValueError("C0 did not preserve frozen R2 compile count")
    if result.get("c0_frozen_baseline", {}).get("funnel", {}).get("type_valid") != 50:
        raise ValueError("C0 did not preserve frozen R2 type count")
    for condition in CONDITIONS:
        summary = result["conditions"][condition]
        if summary.get("case_count") != 60 or summary.get("generation_calls", 0) < 60:
            raise ValueError(f"incomplete condition summary: {condition}")
    if any(str(item) in FORBIDDEN_KEYS or "/kaggle/" in str(item) or "C:\\Users\\" in str(item) for item in walk(result)):
        raise ValueError("unsafe public aggregate")


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def _rows(result: Mapping[str, Any]) -> str:
    all_conditions = {"C0_R2_TYPED_EXAMPLES": result["c0_frozen_baseline"]} | dict(result["conditions"])
    rows = []
    for name in ("C0_R2_TYPED_EXAMPLES",) + CONDITIONS:
        item = all_conditions[name]
        rows.append(f"| {LABELS[name]} | {pct(item['rates']['schema_valid'])} | {pct(item['rates']['type_valid'])} | {pct(item['rates']['parameter_valid'])} | {pct(item['rates']['compile_valid'])} | {item['generation_calls']} | {item['prompt_tokens'] + item['completion_tokens']} |")
    return "\n".join(rows)


def _best_categories(result: Mapping[str, Any]) -> list[str]:
    output = []
    for category in ("SINGLE_MACRO", "PARAMETER_CONTRACT", "TWO_STEP_TYPED_CHAIN", "THREE_STEP_COMPOSITION"):
        choices = [(name, item["category_metrics"][category]["rates"]["compile_valid"]) for name, item in result["conditions"].items()]
        name, rate = max(choices, key=lambda item: (item[1], item[0]))
        output.append(f"- {category}: {pct(rate)} ({name})")
    return output


def report(result: Mapping[str, Any], tests: int) -> str:
    conditions = result["conditions"]
    c2, c3 = conditions["C2_COMPILER_VALID_SKELETON"], conditions["C3_STRUCTURED_COMPILER_CONSTRAINED"]
    best = conditions[result["best_api_pipeline"]]
    failure_lines = "\n".join(f"- {name}: {count}" for name, count in best["failure_types"][:5]) or "- 无"
    return f"""# Macro API Compiler-Aware Ablation V1

## 冻结协议

- C0 仅引用冻结的 R2 aggregate，未重新生成；C1/C2/C3 各一轮正式 pass。
- Qwen3-8B、local Transformers、BF16、4×NVIDIA L4、Internet Off、temperature=0、thinking=false、candidate budget=1。
- 未加载 ARC grids、ARC solutions 或 task predictions。raw responses 仅在忽略的 checkpoint 中。

## 条件比较

| Condition | Schema | Type | Parameter | Compile | Calls | Total tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{_rows(result)}

## C2 / C3 结构指标

- C2 skeleton JSON / selection-valid / canonical-sequence-match: {c2['skeleton']['json_parseable']} / {c2['skeleton']['selection_valid']} / {c2['skeleton']['canonical_sequence_match']}（每项分母 60）。
- C3 structured JSON / skeleton-valid / parameter-constraint-valid / canonical-sequence-match: {c3['structured']['json_parseable']} / {c3['structured']['skeleton_selection_valid']} / {c3['structured']['parameter_constraint_valid']} / {c3['structured']['canonical_sequence_match']}（每项分母 60）。

## 各类别最佳 compile-valid rate

{chr(10).join(_best_categories(result))}

## 最佳条件 failure modes

{failure_lines}

## 结论

- BEST_API_PIPELINE: **{result['best_api_pipeline']}**；状态: **{result['api_status']}**。
- C0 R2 type / compile: {pct(result['c0_frozen_baseline']['rates']['type_valid'])} / {pct(result['c0_frozen_baseline']['rates']['compile_valid'])}；最佳 C1--C3 type / compile: {pct(best['rates']['type_valid'])} / {pct(best['rates']['compile_valid'])}。
- 推荐下一实验: **{result['recommended_next_experiment']}**。
- Local tests: **{tests} passed**。
"""


def append_ledger(path: Path, result: Mapping[str, Any], code_commit: str) -> None:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields, rows = list(reader.fieldnames or ()), list(reader)
    if not fields or any(row.get("experiment_id") == result["experiment_id"] for row in rows):
        raise ValueError("invalid or duplicate experiment ledger entry")
    best = result["conditions"][result["best_api_pipeline"]]
    row = {
        "experiment_id": result["experiment_id"], "date": str(result["completed_at_utc"])[:10] or datetime.now().date().isoformat(),
        "git_commit": code_commit, "solver": "Macro API compiler-aware ablation",
        "representation": "C1 compiler-aware catalogue; C2 compiler-valid skeleton; C3 structured compiler constraints",
        "search_method": "one frozen pass per new condition; C0 historical R2", "llm_model": result["model"]["model_source"],
        "candidate_budget": "1", "validation_split": "frozen API-only 60-case development benchmark",
        "tasks_solved": str(best["funnel"]["compile_valid"]), "accuracy": str(best["rates"]["compile_valid"]),
        "runtime_seconds": str(result["run_wall_seconds"]), "gpu_hours": "",
        "notes": json.dumps({"api_status": result["api_status"], "best": result["best_api_pipeline"], "arc_data_used": False, "c0_r2_rerun": False}, sort_keys=True),
    }
    with path.open("a", encoding="utf-8", newline="") as handle:
        csv.DictWriter(handle, fieldnames=fields).writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--frozen-config", type=Path, default=ROOT / "configs" / "MACRO_API_COMPILER_AWARE_ABLATION_V1_FROZEN_CONFIG.json")
    parser.add_argument("--public-result", type=Path, default=ROOT / "experiments" / "results" / "MACRO_API_COMPILER_AWARE_ABLATION_V1.json")
    parser.add_argument("--report", type=Path, default=ROOT / "reports" / "macro_api_compiler_aware_ablation_v1.md")
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
