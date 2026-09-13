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
    allowed_statuses = {
        "CANDIDATES_AND_RANKED_PREDICTIONS_FROZEN_BEFORE_EXACT_SCORING",
        "CANDIDATES_COMBINED_FROZEN_BEFORE_TRAIN_VERIFIER_RERANK",
    }
    if frozen.get("status") not in allowed_statuses or not isinstance(records, dict) or not records:
        raise ValueError("requires complete frozen candidate artifact before opening train challenges")
    # Challenge JSON supplies public task inputs and their train outputs only.
    challenges = json.loads(args.challenge_path.read_text(encoding="utf-8")); result = copy.deepcopy(frozen)
    for task_id, record in result["records"].items():
        task = challenges[task_id]; candidates = record["candidates"]
        likelihood_by_index = dict(zip(record["ranked_candidate_indices"], record["candidate_scores"], strict=True))
        likelihood = [float(likelihood_by_index[index]) for index in range(len(candidates))]
        train_pairs = [(item["input"], item["output"]) for item in task["train"]]
        test_inputs = [item["input"] for item in task["test"]]
        verifier = []
        for candidate in candidates:
            prediction = candidate["prediction"]
            if len(prediction) != len(test_inputs):
                raise ValueError(f"candidate test-output count does not match task test-input count: {task_id}")
            per_test = train_verifier_scores(train_pairs, test_inputs[0], [prediction[0]])
            scores = [per_test[0]]
            for test_input, test_prediction in zip(test_inputs[1:], prediction[1:], strict=True):
                scores.append(train_verifier_scores(train_pairs, test_input, [test_prediction])[0])
            verifier.append(sum(scores) / len(scores))
        record["train_verifier"] = {"scores": verifier, "leave_one_train_pair_out": True, "test_input_count": len(test_inputs), "aggregation": "mean_per_test_relation_consistency"}
        record["verifier_ranking_indices"] = rank_with_verifier(likelihood, verifier)
    result["status"] = "CANDIDATES_RERANKED_BY_TRAIN_ONLY_VERIFIER_FROZEN_BEFORE_EXACT_SCORING"
    result["verifier_source_frozen_sha256"] = hashlib.sha256(args.frozen.read_bytes()).hexdigest()
    result["verifier_protocol"] = "generic train input/output relation signatures plus model likelihood; no test outputs or solution file"
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "task_count": len(records)}, sort_keys=True))


if __name__ == "__main__": main()
