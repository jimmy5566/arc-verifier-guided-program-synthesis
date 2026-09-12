"""Build the development-only capability gate for Grid Recognition V1.

This is intentionally an oracle *audit*, not recognizer training. It performs
all train-only compilation/hard-verification first, freezes only hashes of
test predictions, and opens development solutions only after every candidate
prediction is frozen.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arc.io import load_dataset
from recognition.downstream_v1_capability_audit import find_train_consistent_oracle, prediction_hash


def _hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _exposed_config_ids() -> set[str]:
    result: set[str] = set()
    for path in ROOT.joinpath("configs").glob("*.json"):
        # The full-catalog and held-out manifests are not prompt-tuning
        # exposure sets. Only actual historical LLM prompt-study subsets are
        # excluded from this controlled development cohort.
        if not any(token in path.name.lower() for token in ("pilot_exposed", "confirmation_dev", "program_synthesis_v2_pilot", "retrieval_diagnostic")):
            continue
        try: payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError: continue
        def visit(value: Any, key: str = "") -> None:
            if isinstance(value, dict):
                for child_key, child in value.items(): visit(child, str(child_key))
            elif isinstance(value, list) and key in {"task_ids", "development_task_ids", "pilot_task_ids"}:
                result.update(item for item in value if isinstance(item, str) and len(item) == 8)
        visit(payload)
    return result


def _audit_one(challenge_path: str, task_id: str) -> dict[str, Any]:
    task = load_dataset(challenge_path)[task_id]
    oracle, stats = find_train_consistent_oracle(task)
    return {"task_id": task_id, "stats": stats, "oracle": None if oracle is None else {"skeleton_id": oracle.skeleton_id, "prediction": oracle.predictions, "prediction_hash": prediction_hash(oracle.predictions)}}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--workers", type=int, default=4); args = parser.parse_args()
    split_rows = list(csv.DictReader((ROOT / "data/splits/task_splits.csv").open(encoding="utf-8")))
    exposed = _exposed_config_ids(); development = sorted(row["task_id"] for row in split_rows if row["split"] == "development" and row["task_id"] not in exposed)
    challenge_path = ROOT / "data/raw/arc-agi_training_challenges.json"; predictions: dict[str, dict[str, Any]] = {}; totals = Counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        future_map = {pool.submit(_audit_one, str(challenge_path), task_id): task_id for task_id in development}
        for index, future in enumerate(as_completed(future_map), 1):
            record = future.result(); totals.update(record["stats"]); oracle = record["oracle"]
            if oracle is not None: predictions[record["task_id"]] = oracle
            if index % 25 == 0 or index == len(development): print(json.dumps({"event": "capability_gate_progress", "completed": index, "total": len(development), "train_consistent": len(predictions)}), flush=True)
    # Prediction hashes are the frozen boundary. Only after all development
    # candidates are complete do we load any development solution file.
    freeze_hash = _hash({key: value["prediction_hash"] for key, value in sorted(predictions.items())})
    solutions = json.loads((ROOT / "data/raw/arc-agi_training_solutions.json").read_text(encoding="utf-8"))
    exact = {task_id: item for task_id, item in predictions.items() if item["prediction"] == solutions.get(task_id)}
    private = {"protocol": "all train predictions frozen before solution load", "prediction_freeze_hash": freeze_hash, "train_consistent": predictions, "test_exact": exact}
    private_path = ROOT / "artifacts/grid_recognition_diagnostic_v1_capability_gate_private.json"; private_path.write_text(json.dumps(private, indent=2), encoding="utf-8")
    # The hard gate requires a full controlled cohort, selected independently
    # of model outcomes from the eligible exact oracle pool only.
    selected = sorted(exact, key=lambda task_id: hashlib.sha256(f"GRID_RECOGNITION_DIAGNOSTIC_V1:{task_id}".encode()).hexdigest())[:24]
    status = "READY_FOR_FORMAL_A_B_C" if len(selected) == 24 else "BLOCKED_CAPABILITY_COVERAGE"
    public = {
        "config_id": "GRID_RECOGNITION_DIAGNOSTIC_V1_TASKS", "status": status, "split": "development", "task_count_required": 24,
        "selected_task_ids": selected if status == "READY_FOR_FORMAL_A_B_C" else [],
        "selection": "deterministic SHA-256 sampling from exact oracle-capability pool after excluding historical LLM/Macro/parameter prompt-study task IDs",
        "selection_salt": "GRID_RECOGNITION_DIAGNOSTIC_V1", "development_pool_size_after_exposure_exclusion": len(development), "excluded_historical_prompt_study_count": len(exposed),
        "train_consistent_oracle_pool_size": len(predictions), "test_exact_oracle_pool_size": len(exact), "candidate_audit_totals": dict(sorted(totals.items())),
        "prediction_freeze_hash": freeze_hash, "oracle_programs_public": False, "raw_grids_public": False, "test_outputs_public": False,
        "gate_reason": "24/24 exact-oracle downstream coverage required" if status == "READY_FOR_FORMAL_A_B_C" else "fewer than 24 development tasks have a train-consistent and test-exact frozen DOWNSTREAM_V1 oracle; formal A/B/C is prohibited",
    }
    output = ROOT / "configs/GRID_RECOGNITION_DIAGNOSTIC_V1_TASKS.json"; output.write_text(json.dumps(public, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "eligible": len(exact), "output": str(output), "private": str(private_path)}))


if __name__ == "__main__": main()
