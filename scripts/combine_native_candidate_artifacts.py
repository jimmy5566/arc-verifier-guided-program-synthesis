"""Combine two already-frozen native candidate pools without opening solutions.

Likelihoods from independently adapted models are not directly comparable.
This utility normalizes likelihood within each source/task, deduplicates exact
grid predictions, and leaves train-only structural reranking to the separate
frozen verifier step.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from math import sqrt
from pathlib import Path
from typing import Any


FROZEN_STATUS = "CANDIDATES_AND_RANKED_PREDICTIONS_FROZEN_BEFORE_EXACT_SCORING"
COMBINED_STATUS = "CANDIDATES_COMBINED_FROZEN_BEFORE_TRAIN_VERIFIER_RERANK"


def _z(values: list[float]) -> list[float]:
    mean = sum(values) / len(values)
    scale = sqrt(sum((value - mean) ** 2 for value in values) / len(values))
    return [(value - mean) / scale if scale else 0.0 for value in values]


def _likelihood_by_candidate(record: dict[str, Any]) -> list[float]:
    candidates = record.get("candidates", [])
    indices, scores = record.get("ranked_candidate_indices", []), record.get("candidate_scores", [])
    if not candidates or len(indices) != len(scores) or set(indices) != set(range(len(candidates))):
        raise ValueError("source record requires one likelihood score for every candidate")
    source = dict(zip(indices, scores, strict=True))
    return _z([float(source[index]) for index in range(len(candidates))])


def _key(candidate: dict[str, Any]) -> str:
    return json.dumps(candidate["prediction"], separators=(",", ":"), sort_keys=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--ttt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite combined frozen artifact")
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    ttt = json.loads(args.ttt.read_text(encoding="utf-8"))
    if baseline.get("status") != FROZEN_STATUS or ttt.get("status") != FROZEN_STATUS:
        raise ValueError("both inputs must be candidate artifacts frozen before exact scoring")
    base_records, ttt_records = baseline.get("records"), ttt.get("records")
    if not isinstance(base_records, dict) or set(base_records) != set(ttt_records or {}):
        raise ValueError("baseline and TTT must contain exactly the same task IDs")
    records: dict[str, dict[str, Any]] = {}
    for task_id in sorted(base_records):
        merged: dict[str, tuple[dict[str, Any], float, list[str]]] = {}
        for source_name, record in (("baseline", base_records[task_id]), ("ttt", ttt_records[task_id])):
            for candidate, score in zip(record["candidates"], _likelihood_by_candidate(record), strict=True):
                item = copy.deepcopy(candidate); key = _key(item)
                prior = merged.get(key)
                if prior is None or score > prior[1]:
                    merged[key] = (item, score, [source_name] if prior is None else sorted(set(prior[2] + [source_name])))
                else:
                    merged[key] = (prior[0], prior[1], sorted(set(prior[2] + [source_name])))
        candidates, likelihood = [], []
        for _key_value, (candidate, score, provenance) in sorted(merged.items()):
            candidate["combined_provenance"] = provenance
            candidates.append(candidate); likelihood.append(score)
        ranked_indices = sorted(range(len(candidates)), key=lambda index: (-likelihood[index], index))
        records[task_id] = {
            "task_id": task_id,
            "status": "SUCCESS" if candidates else "NO_VALID_NATIVE_CANDIDATE",
            "candidates": candidates,
            "ranked_candidate_indices": ranked_indices,
            "candidate_scores": [likelihood[index] for index in ranked_indices],
            "combined_candidate_count": len(candidates),
            "baseline_unique_candidate_count": len(base_records[task_id]["candidates"]),
            "ttt_unique_candidate_count": len(ttt_records[task_id]["candidates"]),
        }
    result = {
        "experiment_id": baseline.get("experiment_id"),
        "status": COMBINED_STATUS,
        "protocol": "Source-local likelihood z-normalization, exact prediction deduplication, and deterministic tie ordering; no train targets or solutions opened.",
        "baseline_frozen_sha256": hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
        "ttt_frozen_sha256": hashlib.sha256(args.ttt.read_bytes()).hexdigest(),
        "task_ids_hash": baseline.get("task_ids_hash"),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "task_count": len(records), "candidate_count": sum(len(item["candidates"]) for item in records.values())}, sort_keys=True))


if __name__ == "__main__":
    main()
