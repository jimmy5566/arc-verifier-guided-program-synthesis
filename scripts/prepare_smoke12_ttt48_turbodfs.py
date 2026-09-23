"""Freeze a development-only Smoke12 cohort before TurboDFS inference."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))

from inference.kaggle_l4_parallel_runner import atomic_write_json
from scripts.run_eval3_reference_ttt import _read, _task_hash


EXPERIMENT_ID = "ARC2_SMOKE12_TTT48_FAITHFUL_BOUNDED_TURBODFS"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _partial_rule_match(record: dict[str, Any], target: list[list[list[int]]]) -> bool:
    for candidate in record["candidates"]:
        for grid, truth in zip(candidate["prediction"], target, strict=True):
            if len(grid) != len(truth) or not grid or not truth or len(grid[0]) != len(truth[0]):
                continue
            same = sum(left == right for row, target_row in zip(grid, truth, strict=True) for left, right in zip(row, target_row, strict=True))
            if 0 < same < len(grid) * len(grid[0]):
                return True
    return False


def _any_output_hit(record: dict[str, Any], target: list[list[list[int]]]) -> bool:
    return any(candidate["prediction"][index] == truth for candidate in record["candidates"] for index, truth in enumerate(target))


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("challenge_path", "solutions_path", "ttt24_candidates", "ttt48_candidates", "ttt48_predictions", "output_dir"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.output_dir / "smoke12_manifest.json"
    config_path = args.output_dir / "turbodfs_config_frozen.json"
    if manifest_path.exists() or config_path.exists():
        raise FileExistsError("refusing to overwrite frozen TurboDFS Smoke12 inputs")
    challenge, solutions, ttt24, ttt48, predictions = map(_read, (args.challenge_path, args.solutions_path, args.ttt24_candidates, args.ttt48_candidates, args.ttt48_predictions))
    ids = list(ttt24["task_ids"])
    if len(ids) != 60 or ttt48.get("task_ids") != ids or set(predictions.get("records", {}).get("TTT48", ())) != set(ids):
        raise ValueError("incomplete paired Eval60 source artifacts")
    # This uses historic development outcomes only to define a development
    # cohort.  The inference executable receives only the resulting IDs.
    eligible = [
        task_id for task_id in ids
        if not _any_output_hit(ttt24["records"][task_id], solutions[task_id])
        and _partial_rule_match(ttt24["records"][task_id], solutions[task_id])
        and not _any_output_hit(ttt48["records"][task_id], solutions[task_id])
    ]
    task_ids = sorted(eligible, key=lambda task_id: (hashlib.sha256(task_id.encode()).hexdigest(), task_id))[:12]
    if len(task_ids) != 12:
        raise ValueError(f"need 12 TTT24 pool-miss partial-rule TTT48-unsolved tasks, found {len(task_ids)}")
    config = {
        "algorithm": "public_nvarc_turbo_dfs_score_bounded",
        "reference": {"repo_commit": "846d0198efa752534594e321fc3289fc0a06c657", "max_cumulative_nll": 1.6094379124341003},
        "ttt_steps": 48, "generation_augmentation_count": 8,
        "max_new_tokens": 931, "max_wall_seconds_per_task": 360.0,
        "max_wall_seconds_per_prompt": 45.0, "max_expanded_branches_per_prompt": 64,
        "max_complete_candidates_per_prompt": 4, "max_generated_tokens_per_task": 8000,
        "max_complete_candidates_per_task": 128, "total_smoke_wall_clock_cap_seconds": 5400,
        "parser": "existing_strict_native_parser", "target_blind": True,
    }
    config["config_hash"] = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    manifest = {
        "experiment_id": EXPERIMENT_ID, "status": "SMOKE12_TTT48_TURBODFS_COHORT_FROZEN",
        "development_only": True,
        "selection_provenance": "12 smallest SHA256 task IDs among prior Eval60 TTT24 output-pool misses with PARTIAL_RULE_MATCH and no TTT48 greedy output hit. Solutions were used only for historic development cohort selection, never supplied to inference.",
        "task_ids": task_ids, "task_ids_hash": _task_hash(task_ids),
        "source_challenge_sha256": _sha256(args.challenge_path),
        "ttt24_candidate_sha256": _sha256(args.ttt24_candidates), "ttt48_candidate_sha256": _sha256(args.ttt48_candidates),
        "ttt48_prediction_sha256": _sha256(args.ttt48_predictions),
        "source_solution_sha256_used_only_for_cohort_selection": _sha256(args.solutions_path),
        "eligible_task_count": len(eligible), "turbodfs_config_hash": config["config_hash"],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(manifest_path, manifest); atomic_write_json(config_path, config)
    print(json.dumps({"event": "SMOKE12_TTT48_TURBODFS_COHORT_FROZEN", "task_ids": task_ids, "task_ids_hash": manifest["task_ids_hash"], "config_hash": config["config_hash"], "solutions_used_only_for_historic_cohort_selection": True}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
