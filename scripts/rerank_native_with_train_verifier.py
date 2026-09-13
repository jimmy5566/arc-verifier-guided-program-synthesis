"""Apply a train-only verifier to a frozen native candidate artifact."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from inference.native_train_verifier import rank_with_verifier, train_verifier_scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen", type=Path, required=True); parser.add_argument("--challenge-path", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError("refusing to overwrite verifier-frozen artifact")
    frozen = json.loads(args.frozen.read_text(encoding="utf-8")); records = frozen.get("records")
    if frozen.get("status") != "CANDIDATES_AND_RANKED_PREDICTIONS_FROZEN_BEFORE_EXACT_SCORING" or not isinstance(records, dict) or not records:
        raise ValueError("requires complete frozen candidate artifact before opening train challenges")
    # Challenge JSON supplies public task inputs and their train outputs only.
    challenges = json.loads(args.challenge_path.read_text(encoding="utf-8")); result = copy.deepcopy(frozen)
    for task_id, record in result["records"].items():
        task = challenges[task_id]; candidates = record["candidates"]
        if len(task["test"]) != 1: raise ValueError("verifier v1 currently requires one test input per task")
        likelihood_by_index = dict(zip(record["ranked_candidate_indices"], record["candidate_scores"], strict=True))
        likelihood = [float(likelihood_by_index[index]) for index in range(len(candidates))]
        verifier = train_verifier_scores([(item["input"], item["output"]) for item in task["train"]], task["test"][0]["input"], [item["prediction"][0] for item in candidates])
        record["train_verifier"] = {"scores": verifier, "leave_one_train_pair_out": True}
        record["verifier_ranking_indices"] = rank_with_verifier(likelihood, verifier)
    result["status"] = "CANDIDATES_RERANKED_BY_TRAIN_ONLY_VERIFIER_FROZEN_BEFORE_EXACT_SCORING"
    result["verifier_source_frozen_sha256"] = hashlib.sha256(args.frozen.read_bytes()).hexdigest()
    result["verifier_protocol"] = "generic train input/output relation signatures plus model likelihood; no test outputs or solution file"
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "task_count": len(records)}, sort_keys=True))


if __name__ == "__main__": main()
