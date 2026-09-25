#!/usr/bin/env python3
"""CPU-only post-freeze scoring for the Blackwell Unleashed V2 Eval60 run."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))

from inference.d1_release_contract import PORTFOLIO, grid_key, pool_for_output, select_record


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _source_order(source: Mapping[str, Any], source_name: str, test_index: int) -> list[str]:
    """The historical frozen S2 source-local B-support ordering."""
    evidence = {int(row["candidate_index"]): row for item in source["per_output_evidence"] if int(item["test_index"]) == test_index for row in item["candidates"]}
    grouped: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for candidate_index, candidate in enumerate(source["candidates"]):
        if candidate_index not in evidence:
            raise ValueError(f"{source_name} candidate has no frozen likelihood evidence")
        prediction = candidate["prediction"][test_index]
        tags = candidate.get("support_augmentations") or [candidate.get("augmentation", {})]
        support = sum(1 for tag in tags if tag.get("geometry") in PORTFOLIO[source_name])
        if not support:
            continue
        nlls = evidence[candidate_index]["view_negative_log_likelihoods"]
        grouped[grid_key(prediction)].append((support - fmean(float(value) for value in nlls), candidate_index))
    return sorted(grouped, key=lambda key: (-max(item[0] for item in grouped[key]), min(item[1] for item in grouped[key]), key))


def _s2_record(record: Mapping[str, Any], task_contract: Mapping[str, Any], test_inputs: list[Any]) -> dict[str, Any]:
    """Frozen source-diverse S2: best TTT24 plus best distinct TTT48."""
    outputs = []
    for output in task_contract["test_outputs"]:
        index = int(output["test_index"])
        source24, source48 = record["sources"]["TTT24"], record["sources"]["TTT48"]
        order24, order48 = _source_order(source24, "TTT24", index), _source_order(source48, "TTT48", index)
        pool = {entry["grid_key"]: entry["grid"] for entry in pool_for_output(record["sources"], index)}
        if not pool:
            fallback = test_inputs[index]
            outputs.append({"test_index": index, "attempt_1": fallback, "attempt_2": fallback, "selection_source": "COMPLETED_EMPTY_INPUT_COPY"})
            continue
        first = order24[0] if order24 else (order48[0] if order48 else sorted(pool)[0])
        second = next((key for key in order48 if key != first), next((key for key in order24 if key != first), first))
        outputs.append({"test_index": index, "attempt_1": pool[first], "attempt_2": pool[second], "selection_source": "S2_SOURCE_DIVERSE", "ttt24_order": order24, "ttt48_order": order48})
    return {"status": "SUCCESS", "method": "S2_source_diverse", "outputs": outputs}


def _as_predictions(selection: Mapping[str, Any], task_ids: list[str]) -> dict[str, Any]:
    return {
        task_id: {
            "attempt_1": [row["attempt_1"] for row in selection[task_id]["outputs"]],
            "attempt_2": [row["attempt_2"] for row in selection[task_id]["outputs"]],
        }
        for task_id in task_ids
    }


def _score(predictions: Mapping[str, Any], records: Mapping[str, Any], solutions: Mapping[str, Any]) -> tuple[dict[str, int], dict[tuple[str, int], dict[str, bool]]]:
    top1 = top2 = oracle = 0
    rows = {}
    for task_id, record in records.items():
        for index, target in enumerate(solutions[task_id]):
            first = predictions[task_id]["attempt_1"][index]
            second = predictions[task_id]["attempt_2"][index]
            pool_hit = any(entry["grid"] == target for entry in pool_for_output(record["sources"], index))
            first_hit, top2_hit = first == target, first == target or second == target
            top1 += int(first_hit); top2 += int(top2_hit); oracle += int(pool_hit)
            rows[(task_id, index)] = {"top1": first_hit, "top2": top2_hit, "oracle": pool_hit}
    return {"top1": top1, "top2": top2, "oracle": oracle}, rows


def _historical_oracle(pool: Mapping[str, Any], solutions: Mapping[str, Any]) -> dict[tuple[str, int], bool]:
    return {
        (task_id, int(per_test["test_index"])): any(candidate["grid"] == solutions[task_id][int(per_test["test_index"])] for candidate in per_test["candidates"])
        for task_id, task in pool["pools"].items() for per_test in task["per_test"]
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--challenge", type=Path, required=True)
    parser.add_argument("--solutions", type=Path, required=True)
    parser.add_argument("--historical-d1", type=Path, required=True)
    parser.add_argument("--historical-pool", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite frozen analysis: {args.output_dir}")
    candidates, challenge = _read(args.candidates), _read(args.challenge)
    if candidates.get("solutions_opened") is not False or candidates.get("status") != "EVAL60_BLACKWELL_UNLEASHED_V2_CANDIDATES_AND_SCORES_FROZEN_BEFORE_SOLUTIONS":
        raise ValueError("candidate artifact is not a target-blind frozen Blackwell Eval60 run")
    records, contract = candidates["records"], candidates["runtime_contract"]
    task_ids = list(candidates["manifest"]["task_ids"])
    if len(task_ids) != 60 or set(task_ids) != set(records) or set(task_ids) - set(challenge):
        raise ValueError("incomplete or mismatched Eval60 frozen evidence")
    d1_selection, s2_selection = {}, {}
    for task_id in task_ids:
        inputs = [entry["input"] for entry in challenge[task_id]["test"]]
        d1_selection[task_id] = select_record(records[task_id], contract["tasks"][task_id], inputs)
        s2_selection[task_id] = _s2_record(records[task_id], contract["tasks"][task_id], inputs)
    d1_predictions, s2_predictions = _as_predictions(d1_selection, task_ids), _as_predictions(s2_selection, task_ids)
    args.output_dir.mkdir(parents=True)
    frozen = {
        "status": "EVAL60_BLACKWELL_UNLEASHED_V2_D1_AND_S2_PREDICTIONS_FROZEN_BEFORE_SOLUTIONS",
        "solutions_opened": False, "candidate_sha256": _sha256(args.candidates), "task_ids": task_ids,
        "d1_selection": d1_selection, "selector2_selection": s2_selection,
        "d1_predictions": d1_predictions, "selector2_predictions": s2_predictions,
    }
    _atomic(args.output_dir / "selection_frozen.json", frozen)
    # Target access begins only after both frozen selector outputs are written.
    solutions = _read(args.solutions)
    if set(task_ids) - set(solutions):
        raise ValueError("evaluation solutions do not cover the frozen cohort")
    d1_score, d1_rows = _score(d1_predictions, records, solutions)
    s2_score, s2_rows = _score(s2_predictions, records, solutions)
    historical_d1 = _read(args.historical_d1)["predictions"]
    historical_pool = _read(args.historical_pool)
    if set(task_ids) != set(historical_d1) or set(task_ids) != set(historical_pool["pools"]):
        raise ValueError("historical comparison artifacts do not cover the same frozen Eval60")
    # Historical score is recomputed rather than trusted, using the exact
    # historical prediction file.  The historical pool oracle remains a
    # separate retention measure.
    historical_records = {task_id: {"sources": records[task_id]["sources"]} for task_id in task_ids}
    # The current scorer's pool is intentionally not substituted for the
    # historical oracle; it is only used to derive historical top2 hit flags.
    historical_top2 = {(task_id, index): (historical_d1[task_id]["attempt_1"][index] == target or historical_d1[task_id]["attempt_2"][index] == target) for task_id in task_ids for index, target in enumerate(solutions[task_id])}
    historical_oracle = _historical_oracle(historical_pool, solutions)
    d1_top2 = {key for key, value in d1_rows.items() if value["top2"]}; d1_oracle = {key for key, value in d1_rows.items() if value["oracle"]}
    hist_top2 = {key for key, value in historical_top2.items() if value}; hist_oracle = {key for key, value in historical_oracle.items() if value}
    report = {
        "status": "COMPLETE_SCORED_AFTER_D1_AND_S2_FREEZE", "scope": "FROZEN_EVAL60_DEVELOPMENT_COMPARISON_NOT_HELD_OUT_NOT_LB_PERFORMANCE",
        "d1": d1_score, "selector2_source_diverse": s2_score,
        "historical": {"top1": sum(historical_d1[task_id]["attempt_1"][index] == target for task_id in task_ids for index, target in enumerate(solutions[task_id])), "top2": len(hist_top2), "oracle": len(hist_oracle)},
        "paired_d1_vs_historical": {
            "historical_top2_retained": len(hist_top2 & d1_top2), "historical_oracle_retained": len(hist_oracle & d1_oracle),
            "new_top2_hits": sorted(f"{task_id}:{index}" for task_id, index in d1_top2 - hist_top2), "new_oracle_hits": sorted(f"{task_id}:{index}" for task_id, index in d1_oracle - hist_oracle),
            "lost_top2_hits": sorted(f"{task_id}:{index}" for task_id, index in hist_top2 - d1_top2), "lost_oracle_hits": sorted(f"{task_id}:{index}" for task_id, index in hist_oracle - d1_oracle),
        },
        "artifact_sha256": {"candidates": _sha256(args.candidates), "selection_frozen": _sha256(args.output_dir / "selection_frozen.json"), "historical_d1": _sha256(args.historical_d1), "historical_pool": _sha256(args.historical_pool)},
    }
    _atomic(args.output_dir / "EVAL60_BLACKWELL_UNLEASHED_V2_REPORT.json", report)
    (args.output_dir / "EVAL60_BLACKWELL_UNLEASHED_V2_REPORT.md").write_text(
        "# Eval60 Blackwell Unleashed V2\n\n"
        "D1 and frozen source-diverse S2 predictions were persisted before solutions were read. This is development evidence, not held-out or LB performance.\n\n"
        f"- D1: Top-1 {d1_score['top1']}/89, Top-2 {d1_score['top2']}/89, pool oracle {d1_score['oracle']}/89.\n"
        f"- Selector2 source-diverse: Top-1 {s2_score['top1']}/89, Top-2 {s2_score['top2']}/89, pool oracle {s2_score['oracle']}/89.\n"
        f"- Historical D1 retained: {report['paired_d1_vs_historical']['historical_top2_retained']}/28 Top-2, {report['paired_d1_vs_historical']['historical_oracle_retained']}/30 oracle.\n",
        encoding="utf-8",
    )
    print(json.dumps({"event": "EVAL60_BLACKWELL_UNLEASHED_V2_CPU_ANALYSIS_COMPLETE", "d1": d1_score, "selector2": s2_score, "historical_top2_retained": report["paired_d1_vs_historical"]["historical_top2_retained"], "historical_oracle_retained": report["paired_d1_vs_historical"]["historical_oracle_retained"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
