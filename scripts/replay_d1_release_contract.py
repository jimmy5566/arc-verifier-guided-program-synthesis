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
    sources = {"TTT24": {"task_id": task_id, "candidates": raw24["candidates"], "per_output_evidence": []}, "TTT48": {"task_id": task_id, "candidates": raw48["candidates"], "per_output_evidence": []}}
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
    selection_records: dict[str, Any] = {}; submission: dict[str, Any] = {}; empty_outputs = []
    for task_id in manifest["task_ids"]:
        try:
            selection = select_record(records[task_id], manifest["tasks"][task_id]); selection_records[task_id] = selection
            submission[task_id] = [{"attempt_1": row["attempt_1"], "attempt_2": row["attempt_2"]} for row in selection["outputs"]]
        except ReleaseContractError as exc:
            expected_empty = all(value is None for value in expected[task_id]["attempt_1"] + expected[task_id]["attempt_2"])
            if not expected_empty: raise
            empty_outputs.append({"task_id": task_id, "error": str(exc)})
            submission[task_id] = [{"attempt_1": None, "attempt_2": None} for _ in manifest["tasks"][task_id]["test_outputs"]]
    # The strict live finalizer must reject this historical source fixture if
    # it contains an empty combined pool; recording that failure proves the
    # production policy rather than silently retaining null historical output.
    try:
        finalize(challenge, config, {"release_identity": manifest["release_identity"], "records": records})
        finalizer_status = "PASS"
    except ReleaseContractError as exc:
        finalizer_status = "FAIL_CLOSED_EMPTY_POOL"; finalizer_error = str(exc)
    else:
        finalizer_error = None
    mismatches = []
    for task_id in manifest["task_ids"]:
        actual = submission[task_id]
        for index, row in enumerate(actual):
            if row["attempt_1"] != expected[task_id]["attempt_1"][index] or row["attempt_2"] != expected[task_id]["attempt_2"][index]: mismatches.append({"task_id": task_id, "test_index": index})
    # Solutions are deliberately opened only after all outputs and comparisons are frozen.
    top1, top2, total = score(submission, read(args.solutions))
    report = {"status": "PASS_WITH_STRICT_EMPTY_POOL_BLOCKER" if not mismatches and (top1, top2, total) == (20, 28, 89) and finalizer_status == "FAIL_CLOSED_EMPTY_POOL" else "FAIL", "new_contract": "pool_for_output_to_select_record_to_finalizer", "candidate_pool_membership_checked": True, "source_local_rank_and_b_rrf_checked": True, "d1_tie_break_checked": True, "attempt_mismatches": mismatches, "top1": top1, "top2": top2, "pool_oracle": 30, "output_count": total, "empty_pool_outputs": empty_outputs, "strict_finalizer_status": finalizer_status, "strict_finalizer_error": finalizer_error, "selection": selection_records, "submission": submission}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "top1", "top2", "pool_oracle", "output_count")}, sort_keys=True))


if __name__ == "__main__": main()
