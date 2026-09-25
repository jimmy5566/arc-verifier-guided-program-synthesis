#!/usr/bin/env python3
"""CPU-only D1/Selector2 scoring for a frozen 2x3090 Eval60 artifact.

The candidate artifact and challenge are read first.  D1 and source-diverse
Selector2 predictions are written atomically before this program opens the
evaluation solutions file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from inference.d1_release_contract import PORTFOLIO, grid_key, pool_for_output, select_record


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _source_order(source: Mapping[str, Any], source_name: str, test_index: int) -> list[str]:
    evidence = {
        int(row["candidate_index"]): row
        for item in source["per_output_evidence"]
        if int(item["test_index"]) == test_index
        for row in item["candidates"]
    }
    grouped: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for candidate_index, candidate in enumerate(source["candidates"]):
        if candidate_index not in evidence:
            raise ValueError(f"{source_name} candidate lacks frozen evidence")
        tags = candidate.get("support_augmentations") or [candidate.get("augmentation", {})]
        support = sum(1 for tag in tags if tag.get("geometry") in PORTFOLIO[source_name])
        if not support:
            continue
        prediction = candidate["prediction"][test_index]
        nlls = evidence[candidate_index]["view_negative_log_likelihoods"]
        grouped[grid_key(prediction)].append((support - fmean(float(value) for value in nlls), candidate_index))
    return sorted(
        grouped,
        key=lambda key: (-max(value for value, _ in grouped[key]), min(index for _, index in grouped[key]), key),
    )


def _selector2(record: Mapping[str, Any], task_contract: Mapping[str, Any], test_inputs: list[Any]) -> dict[str, Any]:
    outputs = []
    for output in task_contract["test_outputs"]:
        index = int(output["test_index"])
        source24, source48 = record["sources"]["TTT24"], record["sources"]["TTT48"]
        order24 = _source_order(source24, "TTT24", index)
        order48 = _source_order(source48, "TTT48", index)
        pool = {entry["grid_key"]: entry["grid"] for entry in pool_for_output(record["sources"], index)}
        if not pool:
            fallback = test_inputs[index]
            outputs.append({"test_index": index, "attempt_1": fallback, "attempt_2": fallback, "selection_source": "COMPLETED_EMPTY_INPUT_COPY"})
            continue
        first = order24[0] if order24 else (order48[0] if order48 else sorted(pool)[0])
        second = next((key for key in order48 if key != first), next((key for key in order24 if key != first), first))
        outputs.append({"test_index": index, "attempt_1": pool[first], "attempt_2": pool[second], "selection_source": "S2_SOURCE_DIVERSE"})
    return {"status": "SUCCESS", "method": "S2_source_diverse", "outputs": outputs}


def _predictions(selection: Mapping[str, Any], task_ids: list[str]) -> dict[str, Any]:
    return {
        task_id: {
            "attempt_1": [row["attempt_1"] for row in selection[task_id]["outputs"]],
            "attempt_2": [row["attempt_2"] for row in selection[task_id]["outputs"]],
        }
        for task_id in task_ids
    }


def _score(predictions: Mapping[str, Any], records: Mapping[str, Any], solutions: Mapping[str, Any]) -> tuple[dict[str, int], dict[tuple[str, int], dict[str, bool]]]:
    top1 = top2 = oracle = 0
    rows: dict[tuple[str, int], dict[str, bool]] = {}
    for task_id, record in records.items():
        for index, target in enumerate(solutions[task_id]):
            first = predictions[task_id]["attempt_1"][index]
            second = predictions[task_id]["attempt_2"][index]
            pool_hit = any(entry["grid"] == target for entry in pool_for_output(record["sources"], index))
            first_hit = first == target
            top2_hit = first_hit or second == target
            top1 += int(first_hit)
            top2 += int(top2_hit)
            oracle += int(pool_hit)
            rows[(task_id, index)] = {"top1": first_hit, "top2": top2_hit, "oracle": pool_hit}
    return {"top1": top1, "top2": top2, "oracle": oracle}, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--challenge", type=Path, required=True)
    parser.add_argument("--solutions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-candidates-sha256", required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite frozen analysis: {args.output_dir}")
    if _sha256(args.candidates) != args.expected_candidates_sha256:
        raise ValueError("frozen candidate SHA256 mismatch")
    artifact = _read(args.candidates)
    if artifact.get("experiment_id") != "ARC2_EVAL60_FIXED_4PLUS4_D1_AMPERE" or artifact.get("status") != "SUCCESS" or artifact.get("solutions_opened") is not False:
        raise ValueError("not a successful target-blind 2x3090 Eval60 artifact")
    records, manifest = artifact["records"], artifact["manifest"]
    task_ids = list(manifest["task_ids"])
    challenge = _read(args.challenge)
    if len(task_ids) != 60 or set(task_ids) != set(records) or set(task_ids) - set(challenge):
        raise ValueError("incomplete or mismatched Eval60 evidence")
    for task_id in task_ids:
        expected = manifest["tasks"][task_id]
        if hashlib.sha256(_canonical(challenge[task_id]).encode()).hexdigest() != expected["task_sha256"]:
            raise ValueError(f"challenge content mismatch: {task_id}")

    d1_selection, selector2_selection = {}, {}
    for task_id in task_ids:
        test_inputs = [entry["input"] for entry in challenge[task_id]["test"]]
        d1_selection[task_id] = select_record(records[task_id], manifest["tasks"][task_id], test_inputs)
        selector2_selection[task_id] = _selector2(records[task_id], manifest["tasks"][task_id], test_inputs)
    d1_predictions = _predictions(d1_selection, task_ids)
    selector2_predictions = _predictions(selector2_selection, task_ids)
    _atomic(
        args.output_dir / "selection_frozen.json",
        {
            "status": "EVAL60_2X3090_D1_AND_S2_PREDICTIONS_FROZEN_BEFORE_SOLUTIONS",
            "solutions_opened": False,
            "candidate_sha256": _sha256(args.candidates),
            "challenge_sha256": _sha256(args.challenge),
            "task_ids": task_ids,
            "d1_selection": d1_selection,
            "selector2_selection": selector2_selection,
            "d1_predictions": d1_predictions,
            "selector2_predictions": selector2_predictions,
        },
    )

    solutions = _read(args.solutions)
    if set(task_ids) - set(solutions):
        raise ValueError("solutions do not cover Eval60")
    d1_score, d1_rows = _score(d1_predictions, records, solutions)
    selector2_score, selector2_rows = _score(selector2_predictions, records, solutions)
    if len(d1_rows) != 89 or len(selector2_rows) != 89:
        raise ValueError("unexpected Eval60 output count")
    report = {
        "status": "COMPLETE_SCORED_AFTER_D1_AND_S2_FREEZE",
        "scope": "FROZEN_EVAL60_DEVELOPMENT_EVIDENCE_NOT_HELD_OUT_NOT_LB_PERFORMANCE",
        "d1": d1_score,
        "selector2_source_diverse": selector2_score,
        "candidate_sha256": _sha256(args.candidates),
        "selection_sha256": _sha256(args.output_dir / "selection_frozen.json"),
    }
    _atomic(args.output_dir / "REPORT.json", report)
    (args.output_dir / "REPORT.md").write_text(
        "# Eval60 2x3090 frozen selector analysis\n\n"
        "Predictions were frozen before solutions were opened. This is retrospective development evidence, not held-out or leaderboard performance.\n\n"
        f"- D1: Top-1 {d1_score['top1']}/89; Top-2 {d1_score['top2']}/89; pool oracle {d1_score['oracle']}/89.\n"
        f"- Selector2 source-diverse: Top-1 {selector2_score['top1']}/89; Top-2 {selector2_score['top2']}/89; pool oracle {selector2_score['oracle']}/89.\n",
        encoding="utf-8",
    )
    print(json.dumps({"event": "EVAL60_2X3090_SELECTOR_COMPLETE", "d1": d1_score, "selector2": selector2_score}, sort_keys=True))


if __name__ == "__main__":
    main()
