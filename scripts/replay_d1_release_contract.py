#!/usr/bin/env python3
"""CPU exact replay of the *new* D1 contract using frozen 4+4 raw evidence."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
from inference.d1_release_contract import runtime_manifest
from scripts.build_d1_release_submission import finalize
from inference.d1_release_contract import ReleaseContractError, select_record


def read(path: Path) -> Any: return json.loads(path.read_text(encoding="utf-8"))


def sources_for_task(raw24: dict[str, Any], raw48: dict[str, Any], frozen_pool: dict[str, Any], task_id: str) -> dict[str, Any]:
    """Rehydrate raw source candidates plus their frozen per-output scores."""
    sources = {"TTT24": {"task_id": task_id, "status": "SUCCESS" if raw24["candidates"] else "COMPLETED_EMPTY", "candidates": raw24["candidates"], "per_output_evidence": []}, "TTT48": {"task_id": task_id, "status": "SUCCESS" if raw48["candidates"] else "COMPLETED_EMPTY", "candidates": raw48["candidates"], "per_output_evidence": []}}
    for per_test in frozen_pool["per_test"]:
        by_source: dict[str, dict[int, dict[str, Any]]] = {"TTT24": {}, "TTT48": {}}
        for candidate in per_test["candidates"]:
            for row in candidate["source_rows"]:
                source, index = row["source"], int(row["candidate_index"])
                value = {"candidate_index": index, "original_log_likelihood": float(row["original_log_likelihood"]), "view_negative_log_likelihoods": [float(row["mean_view_nll"])]}
                if index in by_source[source] and by_source[source][index] != value:
                    raise ValueError(f"inconsistent frozen likelihood evidence for {source}:{index}")
                by_source[source][index] = value
        for source in by_source:
            sources[source]["per_output_evidence"].append({"test_index": int(per_test["test_index"]), "candidates": [by_source[source][index] for index in sorted(by_source[source])]})
    return sources


def score(submission: dict[str, Any], solutions: dict[str, Any]) -> tuple[int, int, int]:
    top1 = top2 = total = 0
    for task_id, rows in submission.items():
        for index, row in enumerate(rows):
            target = solutions[task_id][index]
            top1 += int(row["attempt_1"] == target); top2 += int(row["attempt_1"] == target or row["attempt_2"] == target); total += 1
    return top1, top2, total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--challenge", type=Path, default=ROOT / "data" / "raw" / "arc-agi_evaluation_challenges.json")
    parser.add_argument("--solutions", type=Path, default=ROOT / "data" / "raw" / "arc-agi_evaluation_solutions.json")
    parser.add_argument("--config", type=Path, default=ROOT / "release" / "TTT24_TTT48_4PLUS4_D1_BASELINE_V1" / "D1_RELEASE_RUNTIME_CONFIG.json")
    parser.add_argument("--raw24", type=Path, default=ROOT / "artifacts" / "eval60_ttt48_confirmation" / "ttt24_candidates_frozen.json")
    parser.add_argument("--raw48", type=Path, default=ROOT / "artifacts" / "eval60_ttt48_confirmation" / "candidates_frozen.json")
    parser.add_argument("--fixed-pool", type=Path, default=ROOT / "artifacts" / "eval60_4plus4_portfolio_audit" / "fixed_portfolio_candidates_frozen.json")
    parser.add_argument("--expected", type=Path, default=ROOT / "artifacts" / "eval60_4plus4_d1_selector" / "d1_predictions_frozen.json")
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    challenge_all, config, raw24, raw48, pools, expected = read(args.challenge), read(args.config), read(args.raw24), read(args.raw48), read(args.fixed_pool), read(args.expected)["predictions"]
    challenge = {task_id: challenge_all[task_id] for task_id in pools["pools"]}
    manifest = runtime_manifest(challenge, config)
    if set(pools["pools"]) != set(manifest["task_ids"]) or set(raw24["records"]) != set(manifest["task_ids"]) or set(raw48["records"]) != set(manifest["task_ids"]): raise ValueError("frozen source task sets do not match runtime challenge")
    records = {}
    for task_id in manifest["task_ids"]:
        records[task_id] = {"task_id": task_id, "status": "SUCCESS", "release_identity": manifest["release_identity"], "sources": sources_for_task(raw24["records"][task_id], raw48["records"][task_id], pools["pools"][task_id], task_id)}
    selection_artifact, submission, provenance = finalize(challenge, config, {"release_identity": manifest["release_identity"], "records": records})
    selection_records = selection_artifact["records"]
    pool_mismatches = []
    for task_id, selection in selection_records.items():
        for index, output in enumerate(selection["outputs"]):
            actual_pool = {row["grid_key"]: row for row in output["candidate_pool"]}
            expected_pool = {row["grid_key"]: row for row in pools["pools"][task_id]["per_test"][index]["candidates"]}
            if set(actual_pool) != set(expected_pool):
                pool_mismatches.append({"task_id": task_id, "test_index": index, "field": "candidate_keys"})
                continue
            for token, actual in actual_pool.items():
                frozen = expected_pool[token]
                actual_rows = sorted((row["source"], row["candidate_index"], row["selected_support_count"], tuple(row["slot_tags"])) for row in actual["source_rows"])
                frozen_rows = sorted((row["source"], row["candidate_index"], row["selected_support_count"], tuple(row["slot_tags"])) for row in frozen["source_rows"])
                if actual_rows != frozen_rows or abs(actual["rrf_score"] - frozen["rrf_score"]) > 1e-12:
                    pool_mismatches.append({"task_id": task_id, "test_index": index, "field": "support_or_rrf", "grid_key": token})
    empty_outputs = [{"task_id": task_id, "test_index": index} for task_id, selection in selection_records.items() for index, row in enumerate(selection["outputs"]) if row.get("selection_source") == "COMPLETED_EMPTY_INPUT_COPY"]
    historical_submission = {task_id: [{"attempt_1": expected[task_id]["attempt_1"][index], "attempt_2": expected[task_id]["attempt_2"][index]} for index in range(len(expected[task_id]["attempt_1"]))] for task_id in manifest["task_ids"]}
    mismatches = []
    for task_id in manifest["task_ids"]:
        actual = submission[task_id]
        for index, row in enumerate(actual):
            if {"task_id": task_id, "test_index": index} not in empty_outputs and (row["attempt_1"] != expected[task_id]["attempt_1"][index] or row["attempt_2"] != expected[task_id]["attempt_2"][index]): mismatches.append({"task_id": task_id, "test_index": index})
    # Solutions are deliberately opened only after all outputs and comparisons are frozen.
    solutions = read(args.solutions)
    historical_top1, historical_top2, total = score(historical_submission, solutions)
    fallback_top1, fallback_top2, _ = score(submission, solutions)
    report = {"status": "PASS" if not mismatches and not pool_mismatches and (historical_top1, historical_top2, total) == (20, 28, 89) and provenance["completed_empty_fallback_output_count"] == len(empty_outputs) else "FAIL", "new_contract": "pool_for_output_to_select_record_to_finalizer", "candidate_pool_membership_checked": not pool_mismatches, "source_local_rank_and_b_rrf_checked": not pool_mismatches, "d1_tie_break_checked": True, "pool_mismatches": pool_mismatches, "attempt_mismatches": mismatches, "historical_nonempty_top1": historical_top1, "historical_nonempty_top2": historical_top2, "fallback_inclusive_top1": fallback_top1, "fallback_inclusive_top2": fallback_top2, "pool_oracle": 30, "output_count": total, "empty_pool_outputs": empty_outputs, "finalizer_provenance": provenance, "selection": selection_records, "submission": submission}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "historical_nonempty_top1", "historical_nonempty_top2", "fallback_inclusive_top1", "fallback_inclusive_top2", "pool_oracle", "output_count")}, sort_keys=True))


if __name__ == "__main__": main()
