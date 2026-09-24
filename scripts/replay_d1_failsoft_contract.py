#!/usr/bin/env python3
"""CPU replay of frozen Eval60 evidence through the shared V2 finalizer."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))

from inference.d1_failsoft_runtime import FAILSOFT_SCHEMA_VERSION, finalize_failsoft
from inference.d1_release_contract import runtime_manifest
from scripts.replay_d1_release_contract import score, sources_for_task


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    original = ROOT.parent / "ARC2"
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, default=original)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source_root = args.artifact_root
    challenge_all = read(source_root / "data/raw/arc-agi_evaluation_challenges.json")
    solutions = read(source_root / "data/raw/arc-agi_evaluation_solutions.json")
    config = read(source_root / "release/TTT24_TTT48_4PLUS4_D1_BASELINE_V1/D1_RELEASE_RUNTIME_CONFIG.json")
    raw24 = read(source_root / "artifacts/eval60_ttt48_confirmation/ttt24_candidates_frozen.json")
    raw48 = read(source_root / "artifacts/eval60_ttt48_confirmation/candidates_frozen.json")
    pools = read(source_root / "artifacts/eval60_4plus4_portfolio_audit/fixed_portfolio_candidates_frozen.json")
    expected = read(source_root / "artifacts/eval60_4plus4_d1_selector/d1_predictions_frozen.json")["predictions"]
    challenge = {task_id: challenge_all[task_id] for task_id in pools["pools"]}
    manifest = runtime_manifest(challenge, config)
    records = {
        task_id: {
            "task_id": task_id,
            "status": "SUCCESS",
            "release_identity": manifest["release_identity"],
            "sources": sources_for_task(raw24["records"][task_id], raw48["records"][task_id], pools["pools"][task_id], task_id),
        }
        for task_id in manifest["task_ids"]
    }
    artifact = {"schema_version": FAILSOFT_SCHEMA_VERSION, "release_identity": manifest["release_identity"], "manifest": manifest, "records": records, "solutions_opened": False}
    selection, predictions, provenance = finalize_failsoft(challenge, config, artifact)
    mismatches = []
    nonempty_mismatches = []
    historical = {}
    for task_id in manifest["task_ids"]:
        historical[task_id] = []
        for index in range(len(challenge[task_id]["test"])):
            wanted = {"attempt_1": expected[task_id]["attempt_1"][index], "attempt_2": expected[task_id]["attempt_2"][index]}
            historical[task_id].append(wanted)
            if predictions[task_id][index] != wanted:
                mismatch = {"task_id": task_id, "test_index": index, "actual": predictions[task_id][index], "expected": wanted, "selection_source": selection["records"][task_id]["outputs"][index].get("selection_source")}
                mismatches.append(mismatch)
                if mismatch["selection_source"] != "COMPLETED_EMPTY_INPUT_COPY" or wanted != {"attempt_1": None, "attempt_2": None}:
                    nonempty_mismatches.append(mismatch)
    top1, top2, total = score(predictions, solutions)
    historical_top1, historical_top2, _ = score(historical, solutions)
    report = {
        "status": "PASS" if not nonempty_mismatches and (top1, top2, total) == (20, 28, 89) else "FAIL",
        "schema_version": FAILSOFT_SCHEMA_VERSION,
        "attempt_mismatches": mismatches,
        "historical_nonempty_attempt_mismatches": nonempty_mismatches,
        "completed_empty_policy_differences": [row for row in mismatches if row not in nonempty_mismatches],
        "top1": top1,
        "top2": top2,
        "output_count": total,
        "pool_oracle": 30,
        "historical_top1": historical_top1,
        "historical_top2": historical_top2,
        "selection": selection,
        "provenance": provenance,
        "solutions_opened_after_freeze": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "top1", "top2", "output_count", "pool_oracle")}, sort_keys=True))


if __name__ == "__main__":
    main()
