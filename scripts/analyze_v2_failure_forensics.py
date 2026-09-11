"""Create sanitized V2 Pilot failure forensics from frozen local checkpoints only."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from llm.v2_failure_forensics import aggregate_rows, classify_checkpoint, nearest_rows


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "artifacts" / "qwen3_8b_v12_pilot_output"


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def integrity(checkpoint: dict[str, Any], expected_ids: set[str], expected_candidates: int, expected_statuses: dict[str, int]) -> dict[str, Any]:
    records = checkpoint.get("records", {})
    candidate_results = [candidate for record in records.values() for candidate in record.get("candidate_results", [])]
    actual_statuses = Counter(str(candidate.get("status")) for candidate in candidate_results)
    ids = set(records)
    parsed_hypothesis_count = sum(len(record.get("macro_hypotheses", [])) for record in records.values())
    model_response_candidates = [candidate for candidate in candidate_results if candidate.get("status") not in {"PROVIDER_FAILED", "TIMEOUT"}]
    candidate_hypothesis_ids_match = all(
        {str(candidate.get("hypothesis_id")) for candidate in record.get("candidate_results", []) if candidate.get("status") not in {"PROVIDER_FAILED", "TIMEOUT"}}
        == {str(hypothesis.get("hypothesis_id")) for hypothesis in record.get("macro_hypotheses", [])}
        for record in records.values()
    )
    return {
        "complete": checkpoint.get("complete") is True,
        "prediction_frozen": bool(checkpoint.get("prediction_frozen_at_utc")),
        "task_count": len(records),
        "declared_ids_match": ids == expected_ids,
        "duplicate_task_count": checkpoint.get("duplicate_task_count"),
        "candidate_count": len(candidate_results),
        "candidate_count_matches": len(candidate_results) == expected_candidates,
        "model_response_candidate_count": len(model_response_candidates),
        "parsed_hypothesis_count": parsed_hypothesis_count,
        "no_candidate_silently_missing": parsed_hypothesis_count == len(model_response_candidates) and candidate_hypothesis_ids_match,
        "status_counts": dict(actual_statuses),
        "status_counts_match": dict(actual_statuses) == expected_statuses,
        "config_sha256": checkpoint.get("frozen_config_sha256"),
        "all_record_task_ids_match_keys": all(record.get("task_id") == task_id for task_id, record in records.items()),
    }


def validator_audit(rows_by_condition: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    all_rows = [row for rows in rows_by_condition.values() for row in rows]
    schema = [row for row in all_rows if row["frozen_status"] == "MACRO_SCHEMA_INVALID"]
    types = [row for row in all_rows if row["frozen_status"] == "MACRO_TYPE_INVALID"]
    unusual = [row for row in all_rows if row["primary_failure_subtype"].startswith(("UNKNOWN", "JSON_PARSE_FAILURE", "PROVIDER_FAILURE"))]
    sample = sorted(schema, key=lambda row: (row["condition"], row["task_id"], row["candidate_index"]))[:10]
    sample += sorted(types, key=lambda row: (row["condition"], row["task_id"], row["candidate_index"]))[:10]
    sample += sorted(unusual, key=lambda row: (row["condition"], row["task_id"], row["candidate_index"]))
    unique = {(row["condition"], row["task_id"], row["candidate_index"]): row for row in sample}
    # All schema/type rows are replayed by the classifier against the actual
    # frozen validator; the explicit sample records the required audit subset.
    incorrect = [row for row in all_rows if row["frozen_status"] in {"MACRO_SCHEMA_INVALID", "MACRO_TYPE_INVALID"} and row["primary_failure_subtype"] == "UNKNOWN_VALIDATOR_FAILURE/UNCLASSIFIED"]
    return {"sample_count": len(unique), "schema_invalid_sample_count": 10, "type_invalid_sample_count": 10, "unusual_failure_count": len(unusual), "confirmed_correct_rejections": len(unique) - len(incorrect), "possible_validator_bugs": [] if not incorrect else [{key: row[key] for key in ("condition", "task_id", "candidate_index", "frozen_reason")} for row in incorrect], "method": "Replayed candidates against the frozen Macro registry and validate_macro_hypothesis; provider parse/context errors were checked against their retained error metadata."}


def report(result: dict[str, Any]) -> str:
    lines = ["# LLM Program Synthesis V2 failure forensics", "", "This is a deterministic post-hoc analysis of frozen Pilot 50 checkpoints. It performed no inference, used no GPU, and did not load ARC grids or solutions.", ""]
    for condition, payload in result["conditions"].items():
        funnel = payload["funnel"]
        lines += [f"## {condition}", "", "### Funnel", "", "| Response | JSON | Schema | Macro API | Type | Parameter | Composition | Compile |", "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |", f"| {funnel['responses']} | {funnel['json_parseable']} | {funnel['schema_valid']} | {funnel['macro_api_valid']} | {funnel['type_valid']} | {funnel['parameter_valid']} | {funnel['composition_valid']} | {funnel['compile_eligible']} |", "", "### Primary failure classes", ""]
        for name, count in sorted(payload["primary_failure_counts"].items(), key=lambda item: (-item[1], item[0])):
            lines.append(f"- `{name}`: {count}")
        lines += ["", "### Contract evidence", ""]
        lines.append(f"- Invalid fields: {', '.join(f'{name} ({count})' for name, count in payload['top_invalid_field_names']) or 'none'}.")
        lines.append(f"- Type mismatch pairs: {', '.join(f'{name} ({count})' for name, count in payload['top_type_mismatch_pairs']) or 'none'}.")
        lines.append(f"- Unknown Macro IDs: {', '.join(f'{name} ({count})' for name, count in payload['top_unknown_macro_ids']) or 'none'}.")
        tokens = payload["token_length_relationship"]
        lines.append(f"- Mean completion tokens: schema-valid {tokens['schema_valid_mean_completion_tokens']}, schema-invalid response {tokens['schema_invalid_response_mean_completion_tokens']}; confirmed max-token cutoffs {tokens['confirmed_max_token_cutoff_count']}.")
        lines += ["", "### Nearest to type-valid", ""]
        for item in payload["nearest_to_type_valid"]:
            lines.append(f"- `{item['task_id']}` candidate {item['candidate_index']}: `{item['primary_failure_subtype']}` at `{item['first_error_path']}` — {item['minimal_change_to_pass_stage']}")
        lines.append("")
    comparison = result["symbolic_vs_direct"]
    lines += ["## Symbolic versus direct", "", comparison["interpretation"], "", "## Validator audit", "", f"- Deterministic audit sample: {result['validator_audit']['sample_count']} candidates (10 schema-invalid, 10 type-invalid, plus every unusual/provider failure).", f"- Possible validator bugs: {len(result['validator_audit']['possible_validator_bugs'])}.", "", "## Interpretation", "", result["primary_bottleneck"], "", "The compiler, ParameterSolver, executor, and HardVerifier were not reached by any frozen candidate, so this pilot cannot support a claim about their effectiveness on ARC transformations.", ""]
    return "\n".join(lines).rstrip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--result", type=Path, default=ROOT / "experiments" / "results" / "LLM_PROGRAM_SYNTHESIS_V2_FAILURE_FORENSICS.json")
    parser.add_argument("--report", type=Path, default=ROOT / "reports" / "llm_program_synthesis_v2_failure_forensics.md")
    parser.add_argument("--candidate-csv", type=Path, default=ROOT / "experiments" / "diagnostics" / "v2_failure_forensics_candidates.csv")
    args = parser.parse_args()
    symbolic = read(args.artifact_root / "LLM_PROGRAM_SYNTHESIS_V2_PILOT_50_QWEN3_8B_SYMBOLIC_FROZEN.json")
    direct = read(args.artifact_root / "LLM_PROGRAM_SYNTHESIS_V2_PILOT_50_QWEN3_8B_DIRECT_FROZEN.json")
    published = read(ROOT / "experiments" / "results" / "LLM_PROGRAM_SYNTHESIS_V2_PILOT_50.json")
    expected_ids = set(published["pilot_config"]["task_ids"])
    expected_sha = published["frozen_config_sha256"]
    checks = {
        "symbolic_parameter_solver": integrity(symbolic, expected_ids, 134, {"MACRO_SCHEMA_INVALID": 79, "MACRO_TYPE_INVALID": 53, "PROVIDER_FAILED": 2}),
        "direct_parameter_ablation": integrity(direct, expected_ids, 141, {"MACRO_SCHEMA_INVALID": 105, "MACRO_TYPE_INVALID": 32, "PROVIDER_FAILED": 4}),
    }
    if any(not all(value for key, value in check.items() if key in {"complete", "prediction_frozen", "declared_ids_match", "candidate_count_matches", "status_counts_match", "all_record_task_ids_match_keys", "no_candidate_silently_missing"}) or check["duplicate_task_count"] != 0 or check["config_sha256"] != expected_sha for check in checks.values()):
        raise SystemExit(f"frozen-data reconciliation failed: {checks}")
    rows_by_condition = {
        "symbolic_parameter_solver": classify_checkpoint(symbolic, condition="symbolic_parameter_solver"),
        "direct_parameter_ablation": classify_checkpoint(direct, condition="direct_parameter_ablation"),
    }
    conditions = {}
    for condition, rows in rows_by_condition.items():
        payload = aggregate_rows(rows)
        payload["nearest_to_type_valid"] = nearest_rows(rows)
        payload["completion_tokens_by_primary_failure"] = {kind: {"count": len(items), "mean": sum(int(item["completion_tokens"]) for item in items) / len(items)} for kind in sorted({row["primary_failure_subtype"] for row in rows}) if (items := [row for row in rows if row["primary_failure_subtype"] == kind])}
        conditions[condition] = payload
    audit = validator_audit(rows_by_condition)
    symbolic_funnel, direct_funnel = conditions["symbolic_parameter_solver"]["funnel"], conditions["direct_parameter_ablation"]["funnel"]
    result = {"experiment_id": "LLM_PROGRAM_SYNTHESIS_V2_FAILURE_FORENSICS", "source_experiment": "LLM_PROGRAM_SYNTHESIS_V2_PILOT_50", "new_inference": False, "gpu_used": False, "task_count_per_condition": 50, "frozen_data_reconciliation": checks, "conditions": conditions, "symbolic_vs_direct": {"schema_valid_rate": {"symbolic": symbolic_funnel["schema_valid"] / 132, "direct": direct_funnel["schema_valid"] / 137}, "type_valid_rate": {"symbolic": 0.0, "direct": 0.0}, "near_miss_count": {"symbolic": conditions["symbolic_parameter_solver"]["near_miss_count"], "direct": conditions["direct_parameter_ablation"]["near_miss_count"]}, "interpretation": "Symbolic parameterization had higher initial frozen macro-schema throughput (53/132 versus 32/137), but neither condition produced a type-valid program. It shifted some failures to concept-chain mismatches; it did not demonstrate ParameterSolver benefit because no candidate reached that stage."}, "validator_audit": audit, "hypotheses": {"H1_macro_api_learnability": {"status": "SUPPORTED", "evidence": ["Every response candidate failed before type validity.", "The dominant errors are bare parameter values where the frozen interface requires a one-key wrapper."]}, "H2_reasoning_plus_api_overload": {"status": "INCONCLUSIVE", "evidence": ["Responses contain rationales, but no deterministic semantic oracle was applied to judge their ARC reasoning quality."]}, "H3_symbolic_parameter_design": {"status": "NOT_SUPPORTED", "evidence": ["No candidate reached ParameterSolver; failures occurred in Macro parameter-contract validation before symbolic resolution." ]}, "H4_compiler_executor_limitation": {"status": "NOT_SUPPORTED", "evidence": ["No frozen candidate was type-valid or compiler-eligible."]}, "H5_before_arc_reasoning_evaluation": {"status": "SUPPORTED", "evidence": ["Both conditions have zero type-valid, parameter-valid, composition-valid, and compile-eligible candidates."]}}, "primary_bottleneck": "The frozen V2 Pilot failed primarily at the LLM-to-Macro-DSL parameter-wrapper and concept-chain interface before the ParameterSolver, compiler, executor, or HardVerifier could be meaningfully evaluated.", "secondary_bottlenecks": ["Macro concept-chain mismatches", "Provider-side JSON parsing and context-window failures"], "recommended_next_experiment": "MACRO_API_COMPREHENSION_BENCHMARK", "publication_safety": "Aggregate-only output; no raw response, prompt, ARC grid, solution, or checkpoint content is written."}
    args.candidate_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = [row for values in rows_by_condition.values() for row in values]
    with args.candidate_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report(result) + "\n", encoding="utf-8")
    print(json.dumps({"status": "COMPLETE", "result": str(args.result), "report": str(args.report), "candidate_csv": str(args.candidate_csv)}))


if __name__ == "__main__":
    main()
