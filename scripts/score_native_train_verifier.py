"""Score a verifier-reranked artifact only after its immutable rerank freeze."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--frozen", type=Path, required=True); parser.add_argument("--challenge-path", type=Path, required=True); parser.add_argument("--solutions-path", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError("refusing to overwrite verifier score")
    frozen = json.loads(args.frozen.read_text(encoding="utf-8")); records = frozen.get("records")
    if frozen.get("status") != "CANDIDATES_RERANKED_BY_TRAIN_ONLY_VERIFIER_FROZEN_BEFORE_EXACT_SCORING" or not isinstance(records, dict) or not records:
        raise ValueError("requires fully frozen train-verifier artifact before opening solutions")
    from arc.io import load_solutions
    solutions = load_solutions(args.solutions_path); methods = ("likelihood", "verifier", "likelihood_plus_verifier")
    exact = {method: 0 for method in methods}; any_hit = 0; rescued = harmed = 0
    details = {}
    for task_id, record in records.items():
        expected = solutions[task_id]; candidates = [item["prediction"] for item in record["candidates"]]
        rankings = record["verifier_ranking_indices"]
        result = {method: bool(rankings[method] and candidates[rankings[method][0]] == expected) for method in methods}
        for method, value in result.items(): exact[method] += int(value)
        any_hit += int(any(item == expected for item in candidates)); rescued += int(not result["likelihood"] and result["likelihood_plus_verifier"]); harmed += int(result["likelihood"] and not result["likelihood_plus_verifier"])
        details[task_id] = result
    output = {"experiment_id": "ARC2_NATIVE_TTT_AND_STRONG_VERIFIER_V1", "status": "VERIFIER_SCORED_AFTER_RERANK_FREEZE", "methods_top1_exact": exact, "any_of_k_exact": any_hit, "rescued_vs_likelihood": rescued, "harmed_vs_likelihood": harmed, "details": details, "leakage_audit": "Solutions were imported only after the verifier-reranked artifact passed its frozen status gate."}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: output[key] for key in ("methods_top1_exact", "any_of_k_exact", "rescued_vs_likelihood", "harmed_vs_likelihood")}, sort_keys=True))


if __name__ == "__main__": main()
