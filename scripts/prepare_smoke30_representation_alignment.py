"""Freeze a 30-task development cohort for representation-only comparison."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))

from arc.io import load_dataset
from inference.kaggle_l4_parallel_runner import atomic_write_json
from inference.nvarc_reference_like_augmentation import reference_like_variants
from scripts.run_eval3_reference_ttt import _read, _task_hash


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _id_hash(task_id: str) -> str:
    return hashlib.sha256(task_id.encode("utf-8")).hexdigest()


def _pool_miss_ids(candidates: dict[str, Any], solutions: dict[str, Any]) -> list[str]:
    return [
        task_id for task_id, record in candidates["records"].items()
        if not any(candidate["prediction"] == solutions[task_id] for candidate in record["candidates"])
    ]


def build(*, challenge_path: Path, solutions_path: Path, baseline_candidates: Path, baseline_predictions: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    challenge, solutions, candidates, predictions = map(_read, (challenge_path, solutions_path, baseline_candidates, baseline_predictions))
    all_ids = list(candidates.get("task_ids", ()))
    if len(all_ids) != 60 or set(all_ids) != set(candidates.get("records", ())) or set(all_ids) != set(predictions.get("records", {}).get("TTT24", ())):
        raise ValueError("current TTT24 baseline artifacts do not cover the frozen Eval60 cohort")
    pool_misses = _pool_miss_ids(candidates, solutions)
    if len(pool_misses) != 48:
        raise ValueError(f"expected 48 existing TTT24 pool misses, found {len(pool_misses)}")
    task_ids = sorted(pool_misses, key=lambda task_id: (_id_hash(task_id), task_id))[:30]
    tasks = load_dataset(challenge_path)
    variants = {
        task_id: [variant.to_dict() for variant in reference_like_variants(task_id, len(tasks[task_id].train))]
        for task_id in task_ids
    }
    task_ids_hash = _task_hash(task_ids)
    variant_config = {
        "experiment_id": "ARC2_SMOKE30_REFERENCE_LIKE_REPRESENTATION_ALIGNMENT",
        "status": "FROZEN_TARGET_BLIND_VARIANTS_BEFORE_INFERENCE",
        "task_ids": task_ids,
        "task_ids_hash": task_ids_hash,
        "variant_count": 8,
        "operator_family": ["pair_order_permutation", "dihedral_geometry", "arbitrary_color_permutation"],
        "seed_rule": "SHA256(task_id + ':' + variant_id), reduced to uint32 for NumPy RandomState",
        "tasks": variants,
    }
    variant_config["config_hash"] = hashlib.sha256(json.dumps(variant_config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    manifest = {
        "experiment_id": variant_config["experiment_id"],
        "status": "SMOKE30_REFERENCE_LIKE_REPRESENTATION_COHORT_FROZEN",
        "development_only": True,
        "selection_provenance": "The 30 smallest SHA256(task_id) among the already-known TTT24 Eval60 task-level candidate-pool misses; target labels are not available to the G1 generation executable.",
        "source_challenge_sha256": _sha256(challenge_path),
        "source_solution_sha256_used_only_for_existing_pool_miss_cohort_selection": _sha256(solutions_path),
        "baseline_ttt24_candidate_sha256": _sha256(baseline_candidates),
        "baseline_ttt24_per_output_prediction_sha256": _sha256(baseline_predictions),
        "baseline_pool_miss_count": len(pool_misses),
        "task_ids": task_ids,
        "task_ids_hash": task_ids_hash,
        "variant_config_hash": variant_config["config_hash"],
        "generation_contract": {"count_per_task": 8, "decode": "greedy", "no_beam": True, "no_dfs": True, "representation_change_only": True},
    }
    return manifest, variant_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--challenge-path", type=Path, required=True)
    parser.add_argument("--solutions-path", type=Path, required=True)
    parser.add_argument("--baseline-candidates", type=Path, required=True)
    parser.add_argument("--baseline-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if any((args.output_dir / name).exists() for name in ("smoke30_manifest.json", "reference_variant_config.json")):
        raise FileExistsError("refusing to overwrite frozen Smoke30 representation inputs")
    manifest, variants = build(
        challenge_path=args.challenge_path,
        solutions_path=args.solutions_path,
        baseline_candidates=args.baseline_candidates,
        baseline_predictions=args.baseline_predictions,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output_dir / "smoke30_manifest.json", manifest)
    atomic_write_json(args.output_dir / "reference_variant_config.json", variants)
    print(json.dumps({"event": "SMOKE30_COHORT_AND_VARIANTS_FROZEN", "task_count": len(manifest["task_ids"]), "task_ids_hash": manifest["task_ids_hash"], "variant_config_hash": variants["config_hash"], "solutions_used_only_for_existing_baseline_pool_miss_selection": True}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
