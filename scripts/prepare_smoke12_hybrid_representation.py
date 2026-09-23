"""Freeze the target-blind Smoke12 hybrid-representation cohort and views."""
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
from inference.nvarc_hybrid_representation import hybrid_views, redundant_geometry_indices
from scripts.run_eval3_reference_ttt import _read, _task_hash


EXPERIMENT_ID = "ARC2_SMOKE12_HYBRID_REPRESENTATION"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _id_hash(task_id: str) -> str:
    return hashlib.sha256(task_id.encode("utf-8")).hexdigest()


def _pool_misses(candidates: dict[str, Any], solutions: dict[str, Any]) -> list[str]:
    return [
        task_id for task_id, record in candidates["records"].items()
        if not any(candidate["prediction"] == solutions[task_id] for candidate in record["candidates"])
    ]


def build(*, challenge_path: Path, solutions_path: Path, baseline_candidates: Path, baseline_predictions: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    challenge, solutions, candidates, predictions = map(_read, (challenge_path, solutions_path, baseline_candidates, baseline_predictions))
    all_ids = list(candidates.get("task_ids", ()))
    if len(all_ids) != 60 or set(all_ids) != set(candidates.get("records", ())) or set(all_ids) != set(predictions.get("records", {}).get("TTT24", ())):
        raise ValueError("current TTT24 artifacts do not cover the original Eval60 cohort")
    misses = _pool_misses(candidates, solutions)
    if len(misses) != 48:
        raise ValueError(f"expected 48 original TTT24 pool misses, found {len(misses)}")
    task_ids = sorted(misses, key=lambda task_id: (_id_hash(task_id), task_id))[:12]
    tasks = load_dataset(challenge_path)
    views = {task_id: [view.to_dict() for view in hybrid_views(task_id, len(tasks[task_id].train), candidates["records"][task_id])] for task_id in task_ids}
    replaced = {task_id: list(redundant_geometry_indices(candidates["records"][task_id])) for task_id in task_ids}
    task_ids_hash = _task_hash(task_ids)
    variants = {
        "experiment_id": EXPERIMENT_ID,
        "status": "FROZEN_TARGET_BLIND_HYBRID_VIEWS_BEFORE_INFERENCE",
        "task_ids": task_ids,
        "task_ids_hash": task_ids_hash,
        "generation_count": 8,
        "geometry_view_count": 6,
        "reference_like_view_count": 2,
        "operator_family": ["baseline_dihedral_geometry", "pair_order_permutation", "arbitrary_color_permutation"],
        "reference_seed_rule": "SHA256(task_id + ':' + variant_id), reduced to uint32 for NumPy RandomState",
        "replaced_geometry_indices": replaced,
        "replacement_rule": "Choose the two highest frozen baseline duplicate-support redundancy scores among non-identity geometry views; ties use original Aug8 order.",
        "tasks": views,
    }
    variants["config_hash"] = hashlib.sha256(json.dumps(variants, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "status": "SMOKE12_HYBRID_REPRESENTATION_COHORT_FROZEN",
        "development_only": True,
        "selection_provenance": "The 12 smallest SHA256(task_id) among the original known TTT24 Eval60 task-level candidate-pool misses. Targets are unavailable to G1 generation.",
        "source_challenge_sha256": _sha256(challenge_path),
        "source_solution_sha256_used_only_for_original_pool_miss_cohort_selection": _sha256(solutions_path),
        "baseline_ttt24_candidate_sha256": _sha256(baseline_candidates),
        "baseline_ttt24_per_output_prediction_sha256": _sha256(baseline_predictions),
        "baseline_pool_miss_count": len(misses),
        "task_ids": task_ids,
        "task_ids_hash": task_ids_hash,
        "variant_config_hash": variants["config_hash"],
        "generation_contract": {"count_per_task": 8, "decode": "greedy", "no_beam": True, "no_dfs": True, "hybrid_representation_only": True},
    }
    return manifest, variants


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("challenge_path", "solutions_path", "baseline_candidates", "baseline_predictions", "output_dir"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    if any((args.output_dir / name).exists() for name in ("smoke12_manifest.json", "hybrid_variant_config.json")):
        raise FileExistsError("refusing to overwrite frozen Smoke12 hybrid inputs")
    manifest, variants = build(challenge_path=args.challenge_path, solutions_path=args.solutions_path, baseline_candidates=args.baseline_candidates, baseline_predictions=args.baseline_predictions)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output_dir / "smoke12_manifest.json", manifest)
    atomic_write_json(args.output_dir / "hybrid_variant_config.json", variants)
    print(json.dumps({"event": "SMOKE12_HYBRID_COHORT_AND_VIEWS_FROZEN", "task_count": 12, "task_ids_hash": manifest["task_ids_hash"], "variant_config_hash": variants["config_hash"], "solutions_used_only_for_existing_pool_miss_selection": True}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
