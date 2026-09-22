"""Extract a target-scored CSV from frozen local candidate/ranking records only."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys

THIS = Path(__file__).resolve()
PILOT_ROOT = THIS.parents[1]
REPO_ROOT = THIS.parents[3]
sys.path.insert(0, str(THIS.parent))
from metrics import failure_source

ARTIFACT = REPO_ROOT / "artifacts" / "ARC2_STRATEGY_AWARE_RANKER_FROZEN30_STRATEGY_ONLY_RERANKED_FROZEN.json"
SOLUTIONS = REPO_ROOT / "data" / "raw" / "arc-agi_training_solutions.json"
OUT = PILOT_ROOT / "data" / "arc30_pilot.csv"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if not ARTIFACT.exists() or not SOLUTIONS.exists():
        missing = [str(path) for path in (ARTIFACT, SOLUTIONS) if not path.exists()]
        raise FileNotFoundError("required cached data unavailable; refusing regeneration: " + ", ".join(missing))
    frozen = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    records = frozen.get("records", {})
    if len(records) != 30:
        raise ValueError(f"expected frozen 30 records, found {len(records)}")
    # Candidate/ranking records are already frozen; only now open local labels
    # to score their exact grids.
    solutions = json.loads(SOLUTIONS.read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []
    for task_id, record in sorted(records.items()):
        if task_id not in solutions:
            raise KeyError(f"frozen task absent from cached development solutions: {task_id}")
        candidates = record["candidates"]
        ranked = record["ranked_candidate_indices"]
        if not candidates or not ranked:
            raise ValueError(f"empty cached candidates/ranking for {task_id}")
        target = solutions[task_id]
        candidate_hits = [candidate["prediction"] == target for candidate in candidates]
        selected = int(ranked[0])
        if selected < 0 or selected >= len(candidates):
            raise ValueError(f"invalid selected index for {task_id}")
        scores = [float(value) for value in record.get("candidate_scores", [])]
        if len(scores) != len(candidates):
            raise ValueError(f"candidate-score mismatch for {task_id}")
        # Scores are in cached candidate order; ranking is the frozen likelihood
        # order, so selected score and runner-up are indexed through `ranked`.
        top1_score = scores[selected]
        second_score = scores[int(ranked[1])] if len(ranked) > 1 else None
        any_correct = any(candidate_hits)
        top1_correct = bool(candidate_hits[selected])
        rows.append({
            "task_id": task_id,
            "any_of_k_correct": int(any_correct),
            "top1_correct": int(top1_correct),
            "selected_candidate_id": selected,
            "number_of_candidates": len(candidates),
            "top1_score": top1_score,
            "second_score": second_score if second_score is not None else "",
            "score_margin": top1_score - second_score if second_score is not None else "",
            "candidate_score_min": min(scores),
            "candidate_score_max": max(scores),
            "candidate_score_mean": sum(scores) / len(scores),
            "candidate_score_std": (sum((score - sum(scores) / len(scores)) ** 2 for score in scores) / len(scores)) ** 0.5,
            "correct_candidate_count": sum(candidate_hits),
            "frozen_cohort": "ARC2_NATIVE_RANKER_FROZEN30",
            "artifact_sha256": sha256(ARTIFACT),
            "failure_source": failure_source(any_correct, top1_correct),
        })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    any_k = sum(int(row["any_of_k_correct"]) for row in rows)
    top1 = sum(int(row["top1_correct"]) for row in rows)
    print(json.dumps({"output": str(OUT), "tasks": len(rows), "any_of_k": any_k, "top1": top1, "artifact_sha256": sha256(ARTIFACT)}, indent=2))


if __name__ == "__main__":
    main()
