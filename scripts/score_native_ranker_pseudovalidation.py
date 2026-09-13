"""Score pseudo-test rankings from train pairs after candidate freeze only."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from inference.native_ranker import RANKER_METHODS, select_method_from_pseudovalidation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen", type=Path, required=True); parser.add_argument("--pseudo-cohort", type=Path, required=True)
    parser.add_argument("--pseudo-manifest", type=Path, required=True); parser.add_argument("--challenge-path", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError("refusing to overwrite pseudo-validation score")
    frozen = json.loads(args.frozen.read_text(encoding="utf-8")); cohort = json.loads(args.pseudo_cohort.read_text(encoding="utf-8")); records = frozen.get("records")
    if frozen.get("status") != "CANDIDATES_AND_RANKED_PREDICTIONS_FROZEN_BEFORE_EXACT_SCORING" or frozen.get("task_ids_hash") != cohort["task_ids_hash"] or not isinstance(records, dict) or set(records) != set(cohort["task_ids"]):
        raise ValueError("requires complete frozen pseudo candidate artifact before opening train pairs")
    # Original train outputs are the pseudo labels; no test solution file is opened.
    manifest = json.loads(args.pseudo_manifest.read_text(encoding="utf-8")); challenges = json.loads(args.challenge_path.read_text(encoding="utf-8"))
    exact = {method: 0 for method in RANKER_METHODS}; details = {}
    for case_id, record in records.items():
        origin = manifest[case_id]; expected = challenges[origin["source_task_id"]]["train"][origin["held_out_train_index"]]["output"]
        candidates = [item["prediction"] for item in record.get("candidates", ())]
        rankings = record.get("ranking_indices", {})
        results = {}
        for method in RANKER_METHODS:
            order = rankings.get(method, rankings.get("likelihood", ()))
            correct = bool(order and candidates[int(order[0])] == [expected])
            exact[method] += int(correct); results[method] = correct
        details[case_id] = {"ranker_top1": results, "candidate_count": len(candidates)}
    selected = select_method_from_pseudovalidation(exact)
    output = {"experiment_id": "ARC2_NATIVE_RANKER_DIAGNOSIS_AND_REPAIR_V1", "status": "PSEUDOVALIDATION_COMPLETE_TRAIN_PAIRS_ONLY", "pseudo_case_count": len(records), "ranker_top1_exact": exact, "selected_ranker": selected, "selection_rule": "maximum pseudo-test train-pair exact count; fixed declared order breaks ties", "details": details, "leakage_audit": "Only source training pairs were opened after frozen pseudo candidate verification; test solution files were never read."}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"ranker_top1_exact": exact, "selected_ranker": selected}, sort_keys=True))


if __name__ == "__main__": main()
