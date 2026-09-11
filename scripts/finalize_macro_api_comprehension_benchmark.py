"""Publish only the safe aggregate from the one-shot Macro API benchmark.

The Kaggle runner stores per-case responses and parsed programs in its ignored
checkpoint directory.  This finalizer deliberately refuses those fields and
creates the public result, Markdown analysis, and experiment ledger entry
from the aggregate alone.
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_KEYS = {"raw_response", "parsed_program", "records", "checkpoints", "provider_error"}
CATEGORY_LABELS = {
    "SINGLE_MACRO": "Single Macro",
    "PARAMETER_CONTRACT": "Parameter Contract",
    "TWO_STEP_TYPED_CHAIN": "Two-Step Typed Chain",
    "THREE_STEP_COMPOSITION": "Three-Step Composition",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _contains_forbidden(value: object) -> bool:
    if isinstance(value, Mapping):
        if set(value) & FORBIDDEN_KEYS:
            return True
        return any(_contains_forbidden(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_forbidden(item) for item in value)
    return False


def validate_aggregate(result: Mapping[str, Any], benchmark: Mapping[str, Any]) -> None:
    if result.get("experiment_id") != "MACRO_API_COMPREHENSION_BENCHMARK_V1":
        raise ValueError("unexpected aggregate experiment id")
    if result.get("status") != "COMPLETE_FROZEN_ONE_RUN" or result.get("case_count") != 60:
        raise ValueError("aggregate is not a completed 60-case one-shot run")
    if result.get("benchmark_hash") != benchmark.get("benchmark_hash"):
        raise ValueError("aggregate/benchmark hash mismatch")
    if result.get("arc_data_used") is not False or result.get("arc_solutions_used") is not False:
        raise ValueError("API-only protocol violation")
    if result.get("candidate_budget") != 1:
        raise ValueError("candidate budget is not frozen at one")
    if _contains_forbidden(result):
        raise ValueError("aggregate contains a private per-case field")
    if any("/kaggle/" in str(value) or "C:\\Users\\" in str(value) for value in _walk(result)):
        raise ValueError("aggregate contains a private runtime path")
    if result.get("funnel", {}).get("response_received", 0) > 60:
        raise ValueError("invalid funnel count")


def _walk(value: object):
    if isinstance(value, Mapping):
        for item in value.values():
            yield from _walk(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk(item)
    else:
        yield value


def percentage(value: float | int) -> str:
    return f"{100 * float(value):.1f}%"


def v2_reproduction(result: Mapping[str, Any]) -> str:
    roots = {str(item[0]).split("/", 1)[0] for item in result.get("top_failure_types", [])}
    if {"PARAMETER_FAILURE", "TYPE_FAILURE"}.issubset(roots):
        return "YES"
    if roots & {"PARAMETER_FAILURE", "TYPE_FAILURE", "MACRO_API_FAILURE"}:
        return "PARTIAL"
    return "NO"


def report_text(result: Mapping[str, Any], benchmark: Mapping[str, Any], tests_passed: int) -> str:
    funnel = result["funnel"]
    rates = result["rates"]
    categories = result["category_metrics"]
    decision = str(result["go_decision"])
    reproduction = v2_reproduction(result)
    failures = list(result.get("top_failure_types", []))[:5]
    failure_lines = "\n".join(f"{index}. {name}: {count}" for index, (name, count) in enumerate(failures, 1)) or "1. None"
    category_rows = "\n".join(
        f"| {CATEGORY_LABELS[key]} | {categories[key]['case_count']} | {percentage(categories[key]['schema_valid'] / 15)} | {percentage(categories[key]['type_valid'] / 15)} | {percentage(categories[key]['compile_valid'] / 15)} |"
        for key in CATEGORY_LABELS
    )
    questions = {
        "Q1 Macro names": percentage(rates["macro_ids_valid"]),
        "Q2 Argument contracts": percentage(rates["argument_contract_valid"]),
        "Q3 Parameter wrappers": percentage(rates["parameter_valid"]),
        "Q4 Input/output types": percentage(rates["type_valid"]),
        "Q5 Legal two-step chains": percentage(categories["TWO_STEP_TYPED_CHAIN"]["type_valid"] / 15),
        "Q6 Legal three-step chains": percentage(categories["THREE_STEP_COMPOSITION"]["type_valid"] / 15),
        "Q7 V2 type-valid=0 primarily API teaching": reproduction,
        "Q8 ARC reasoning can now be meaningfully evaluated": "YES" if decision == "GO_API_LEARNED" else "NO",
    }
    question_lines = "\n".join(f"- **{label}:** {answer}" for label, answer in questions.items())
    interpretation = (
        "The model met the pre-registered API-comprehension gate; ARC reasoning can be separated from DSL use in the next controlled experiment."
        if decision == "GO_API_LEARNED"
        else "The frozen API-only condition did not meet the pre-registered reliability gate. ARC reasoning remains confounded with Macro DSL use, so no further ARC pilot should be inferred from this result."
    )
    return f"""# Macro API Comprehension Benchmark V1

## Protocol

- Model: Qwen3-8B Competition Model through local offline Transformers (BF16).
- Hardware: four independently loaded NVIDIA L4 workers with sequential artifact warm-up, staggered readiness, then a shared task queue.
- Formal runs: exactly one frozen 60-case run; temperature 0, thinking disabled, candidate budget 1, max new tokens {result['max_new_tokens']}.
- ARC data used: **No**. ARC solutions used: **No**. This benchmark does not execute grids or score task predictions.
- Canonical validation before the run: schema/type/compile **60/60**; benchmark hash `{benchmark['benchmark_hash']}`.
- Raw completions and parsed programs remain in ignored Kaggle/local checkpoint artifacts. This report and the public JSON contain aggregate metrics only.

## Overall funnel

`60 → {funnel['response_received']} → {funnel['json_parseable']} → {funnel['schema_valid']} → {funnel['argument_contract_valid']} → {funnel['type_valid']} → {funnel['parameter_valid']} → {funnel['composition_valid']} → {funnel['compile_valid']}`

| Metric | Rate |
| --- | ---: |
| JSON parseable | {percentage(rates['json_parseable'])} |
| Schema valid | {percentage(rates['schema_valid'])} |
| Macro API valid | {percentage(rates['argument_contract_valid'])} |
| Type valid | {percentage(rates['type_valid'])} |
| Parameter valid | {percentage(rates['parameter_valid'])} |
| Compile valid | {percentage(rates['compile_valid'])} |

## Category metrics

| Category | Cases | Schema valid | Type valid | Compile valid |
| --- | ---: | ---: | ---: | ---: |
{category_rows}

## Failure taxonomy

{failure_lines}

V2 API failure pattern reproduced: **{reproduction}**.

## Pre-registered decision

**{decision}** — recommended next experiment: **{result['recommended_next_experiment']}**.

{interpretation}

## Research questions

{question_lines}

## Verification

- Local test suite after implementation: **{tests_passed} passed**.
- Registry hash: `{result['registry_hash']}`
- Schema hash: `{result['schema_hash']}`
- Frozen config hash: `{result['frozen_config_sha256']}`
"""


def update_ledger(path: Path, result: Mapping[str, Any], git_commit: str) -> None:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or ())
        rows = list(reader)
    if not fieldnames:
        raise ValueError("experiment ledger has no header")
    if any(row.get("experiment_id") == result["experiment_id"] for row in rows):
        raise ValueError("experiment ledger already contains this one-shot result")
    completed = str(result.get("completed_at_utc", ""))
    date = completed[:10] if len(completed) >= 10 else datetime.now().date().isoformat()
    run_seconds = float(result.get("run_wall_seconds", 0.0))
    row = {
        "experiment_id": result["experiment_id"], "date": date, "git_commit": git_commit,
        "solver": "Qwen3 Macro API comprehension", "representation": "frozen V2 Macro DSL contract without ARC grids",
        "search_method": "one program per explicit natural-language transformation", "llm_model": result["model"]["model_source"],
        "candidate_budget": "1", "validation_split": "API-only fixed 60-case benchmark", "tasks_solved": str(result["funnel"]["compile_valid"]),
        "accuracy": str(result["rates"]["compile_valid"]), "runtime_seconds": str(run_seconds), "gpu_hours": "", 
        "notes": json.dumps({"status": result["status"], "decision": result["go_decision"], "type_valid_rate": result["rates"]["type_valid"], "compile_valid_rate": result["rates"]["compile_valid"], "arc_data_used": False, "formal_runs": 1}, sort_keys=True),
    }
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, default=ROOT / "configs" / "macro_api_comprehension_benchmark_v1.json")
    parser.add_argument("--public-result", type=Path, default=ROOT / "experiments" / "results" / "MACRO_API_COMPREHENSION_BENCHMARK_V1.json")
    parser.add_argument("--report", type=Path, default=ROOT / "reports" / "macro_api_comprehension_benchmark_v1.md")
    parser.add_argument("--experiments-csv", type=Path, default=ROOT / "experiments" / "experiments.csv")
    parser.add_argument("--tests-passed", type=int, required=True)
    parser.add_argument("--git-commit", required=True)
    args = parser.parse_args()
    result, benchmark = read_json(args.aggregate), read_json(args.benchmark)
    validate_aggregate(result, benchmark)
    for target in (args.public_result, args.report):
        if target.exists():
            raise FileExistsError(f"refusing to overwrite one-shot public artifact: {target}")
    args.public_result.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.public_result.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.report.write_text(report_text(result, benchmark, args.tests_passed), encoding="utf-8")
    update_ledger(args.experiments_csv, result, args.git_commit)


if __name__ == "__main__":
    main()
