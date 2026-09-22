"""CPU-only post-freeze Any-of-K scoring for the Smoke12 TTT cost ablation."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))

from inference.kaggle_l4_parallel_runner import atomic_write_json
from scripts.run_eval3_reference_ttt import _read, _task_hash
from scripts.run_smoke12_ttt_cost_ablation import FROZEN_STATUS, _frozen_task_ids


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_artifact(value: dict[str, Any], task_ids: list[str], label: str) -> None:
    if value.get("task_ids") != task_ids or value.get("task_ids_hash") != _task_hash(task_ids):
        raise ValueError(f"{label}: task cohort mismatch")
    records = value.get("records")
    if not isinstance(records, dict) or set(records) != set(task_ids):
        raise ValueError(f"{label}: incomplete candidate records")
    for task_id in task_ids:
        record = records[task_id]
        if record.get("task_id") != task_id or not isinstance(record.get("candidates"), list):
            raise ValueError(f"{label}: invalid candidate record {task_id}")


def _pool_hit(record: dict[str, Any], target: list[list[list[int]]]) -> bool:
    return any(candidate.get("prediction") == target for candidate in record["candidates"])


def _cost(artifact: dict[str, Any], *, is_reused: bool = False) -> dict[str, Any]:
    records = list(artifact["records"].values())
    task_seconds = [float(item.get("elapsed_seconds", 0.0)) for item in records]
    task_gpu = float(artifact.get("task_gpu_seconds", sum(task_seconds)))
    model_load = float(artifact.get("model_load_gpu_seconds", 0.0))
    worker_loads = [sum(float(item.get("elapsed_seconds", 0.0)) for item in records if int(item.get("worker_id", -1)) == worker_id) for worker_id in range(4)]
    max_worker = max(worker_loads, default=0.0)
    # Model startup is paid once per 240-task production run.  The remaining
    # workload extrapolates the observed dynamic-queue tail rather than
    # incorrectly restarting the model for every 12-task wave.
    runtime = float(artifact.get("runtime_seconds", 0.0))
    startup_wall = max(0.0, runtime - max_worker) if runtime else 0.0
    projected_wall = startup_wall + max_worker * 20.0
    return {
        "wall_clock_seconds": runtime if runtime else None,
        "task_gpu_seconds": task_gpu,
        "model_load_gpu_seconds": model_load,
        "gpu_seconds_total": float(artifact.get("gpu_seconds_total", task_gpu + model_load)),
        "seconds_per_task_gpu": (task_gpu + model_load) / len(records),
        "peak_vram_gb": max((float(item.get("peak_allocated_vram_mb", 0.0)) / 1024.0 for item in records), default=0.0),
        "projected_240_wall_seconds": projected_wall if runtime else None,
        "projected_240_gpu_seconds": (task_gpu + model_load) / len(records) * 240.0,
        "reused_historical_estimate": is_reused,
    }


def _reused_subset_cost(
    source: dict[str, Any],
    *,
    task_ids: list[str],
    expected_source_sha256: str,
    source_path: Path,
) -> dict[str, Any]:
    """Recover task-local S0 timing without mutating its frozen subset pool.

    S0 deliberately reuses a subset of the already-frozen Eval60 run.  Its
    subset artifact contains candidates only, while the cited source artifact
    retains per-task timing.  This function reports only reconstructable
    task-local cost; model startup was not recorded in that historic artifact
    and is explicitly kept at zero rather than invented.
    """
    if _sha256(source_path) != expected_source_sha256:
        raise ValueError("S0 source artifact hash does not match frozen provenance")
    records = source.get("records")
    if not isinstance(records, dict) or set(task_ids) - set(records):
        raise ValueError("S0 source artifact omits frozen Smoke12 task timing")
    selected = [records[task_id] for task_id in task_ids]
    if any(item.get("task_id") != task_id for task_id, item in zip(task_ids, selected, strict=True)):
        raise ValueError("S0 source task record mismatch")
    # The historical Eval60 run performed ranking as well as candidate
    # generation.  S1/S2 intentionally stop after candidate freezing, so
    # remove the explicitly instrumented ranking stages rather than comparing
    # an all-in historical run with candidate-generation-only runs.
    def _candidate_stage_seconds(item: dict[str, Any]) -> float:
        elapsed = float(item.get("elapsed_seconds", 0.0))
        excluded = float(item.get("original_likelihood_seconds", 0.0)) + float(item.get("b_support_scoring_seconds", 0.0))
        if excluded > elapsed:
            raise ValueError("S0 source has invalid ranking-stage telemetry")
        return elapsed - excluded

    task_seconds = [_candidate_stage_seconds(item) for item in selected]
    worker_loads = [
        sum(_candidate_stage_seconds(item) for item in selected if int(item.get("worker_id", -1)) == worker_id)
        for worker_id in range(4)
    ]
    task_gpu_seconds = sum(task_seconds)
    observed_task_wall = max(worker_loads, default=0.0)
    return {
        "wall_clock_seconds": observed_task_wall,
        "task_gpu_seconds": task_gpu_seconds,
        "model_load_gpu_seconds": 0.0,
        "gpu_seconds_total": task_gpu_seconds,
        "seconds_per_task_gpu": task_gpu_seconds / len(selected),
        "peak_vram_gb": max((float(item.get("peak_allocated_vram_mb", 0.0)) / 1024.0 for item in selected), default=0.0),
        "projected_240_wall_seconds": observed_task_wall * 20.0,
        "projected_240_gpu_seconds": task_gpu_seconds / len(selected) * 240.0,
        "reused_historical_estimate": True,
        "cost_provenance": "candidate-stage task timings reconstructed from exact frozen Eval60 source; historical original-likelihood and B-support stages excluded, and source did not record model startup",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("manifest", "s0", "s1", "s2", "s3", "s0_source", "solutions_path", "output_dir"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    manifest, s0, s1, s2, s3 = (_read(args.manifest), _read(args.s0), _read(args.s1), _read(args.s2), _read(args.s3))
    try:
        task_ids, task_ids_hash = _frozen_task_ids(manifest)
    except ValueError as exc:
        raise ValueError("invalid frozen Smoke12 manifest") from exc
    if manifest.get("status") != "SMOKE12_TTT_COST_ABLATION_COHORT_FROZEN":
        raise ValueError("invalid frozen Smoke12 manifest")
    expected = {
        "S0": "SMOKE12_S0_REUSED_STRONG_TTT24_GREEDY_CANDIDATES_FROZEN",
        "S1": FROZEN_STATUS,
        "S2": FROZEN_STATUS,
        "S3": "SMOKE12_S3_BASE_TTT_UNION_CANDIDATES_FROZEN",
    }
    artifacts = {"S0": s0, "S1": s1, "S2": s2, "S3": s3}
    for label, artifact in artifacts.items():
        if artifact.get("status") != expected[label]:
            raise ValueError(f"{label}: unexpected candidate freeze status")
        _validate_artifact(artifact, task_ids, label)
    if s1.get("condition") != "ttt24_beam2" or s2.get("condition") != "ttt48_greedy":
        raise ValueError("new condition artifacts do not match the frozen Smoke12 design")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite Smoke12 score output")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # All candidate pools, including the CPU-only union, are now immutable.
    freeze = {
        "experiment_id": "ARC2_SMOKE12_TTT_COST_ACCURACY_ABLATION",
        "status": "SMOKE12_ALL_CANDIDATE_POOLS_CONFIRMED_FROZEN_BEFORE_EXACT_SCORING",
        "task_ids": task_ids, "task_ids_hash": task_ids_hash, "solutions_opened": False,
        "candidate_artifact_sha256": {label: _sha256(path) for label, path in {"S0": args.s0, "S1": args.s1, "S2": args.s2, "S3": args.s3}.items()},
    }
    atomic_write_json(args.output_dir / "candidate_pools_frozen.json", freeze)
    # Explicit solution boundary: the frozen manifest exists before this read.
    solutions = _read(args.solutions_path)
    if set(task_ids) - set(solutions):
        raise ValueError("solutions omit a Smoke12 task")
    metrics: dict[str, dict[str, Any]] = {}
    s0_hits: set[str] = set()
    for label, artifact in artifacts.items():
        hits = {task_id for task_id in task_ids if _pool_hit(artifact["records"][task_id], solutions[task_id])}
        if label == "S0":
            s0_hits = hits
        metrics[label] = {
            "any_of_k": f"{len(hits)}/12",
            "any_of_k_count": len(hits),
            "any_of_k_percent": 100.0 * len(hits) / len(task_ids),
            "hit_task_ids": sorted(hits),
            "new_recoveries_vs_s0": sorted(hits - s0_hits) if label != "S0" else [],
            "new_recovery_count_vs_s0": len(hits - s0_hits) if label != "S0" else 0,
            "mean_unique_candidates_per_task": statistics.fmean(len(artifact["records"][task_id]["candidates"]) for task_id in task_ids),
        }
    s0_source = _read(args.s0_source)
    s0_cost = _reused_subset_cost(
        s0_source,
        task_ids=task_ids,
        expected_source_sha256=str(s0["strong_ttt_candidates_sha256"]),
        source_path=args.s0_source,
    )
    s1_cost, s2_cost = _cost(s1), _cost(s2)
    s3_cost = {"cpu_only": True, "new_gpu_seconds": 0.0, "candidate_generation_reused": True}
    for label, cost in (("S0", s0_cost), ("S1", s1_cost), ("S2", s2_cost)):
        recovered = metrics[label]["new_recovery_count_vs_s0"]
        metrics[label]["cost"] = cost
        metrics[label]["gpu_seconds_per_new_recovery"] = None if not recovered else cost["gpu_seconds_total"] / recovered
    metrics["S3"]["cost"] = s3_cost
    # Compare task work only: S0's historic source omitted startup telemetry,
    # and a production run pays the same four-model startup once per method.
    beam_extra = s1_cost["task_gpu_seconds"] - s0_cost["task_gpu_seconds"]
    ttt48_extra = s2_cost["task_gpu_seconds"] - s0_cost["task_gpu_seconds"]
    extras = {
        "BEAM2_EXTRA_GPU_COST": beam_extra,
        "TTT48_EXTRA_GPU_COST": ttt48_extra,
        "GPU_SECONDS_PER_RECOVERY_BEAM2": None if not metrics["S1"]["new_recovery_count_vs_s0"] else beam_extra / metrics["S1"]["new_recovery_count_vs_s0"],
        "GPU_SECONDS_PER_RECOVERY_TTT48": None if not metrics["S2"]["new_recovery_count_vs_s0"] else ttt48_extra / metrics["S2"]["new_recovery_count_vs_s0"],
        "PROJECTED_240_BEAM2": s1_cost["projected_240_wall_seconds"],
        "PROJECTED_240_TTT48": s2_cost["projected_240_wall_seconds"],
    }
    candidates = [(label, metrics[label]["new_recovery_count_vs_s0"], metrics[label].get("cost", {}).get("gpu_seconds_total", 0.0)) for label in ("S1", "S2")]
    positive = [item for item in candidates if item[1] > 0]
    winner = max(positive, key=lambda item: (item[1] / max(item[2] - s0_cost["gpu_seconds_total"], 1e-9), item[1], -item[2]))[0] if positive else "NO_NEW_RECOVERY"
    report = {
        "experiment_id": "ARC2_SMOKE12_TTT_COST_ACCURACY_ABLATION",
        "status": "COMPLETE_SCORED_AFTER_TARGET_BLIND_CANDIDATE_FREEZE",
        "cohort": {
            "task_ids": task_ids,
            "task_ids_hash": task_ids_hash,
            "selection_note": manifest.get("selection_note") or manifest.get("selection", {}).get("rule"),
        },
        "S0_ANYK": metrics["S0"]["any_of_k"], "S1_BEAM2_ANYK": metrics["S1"]["any_of_k"],
        "S2_TTT48_ANYK": metrics["S2"]["any_of_k"], "S3_BASE_TTT_UNION_ANYK": metrics["S3"]["any_of_k"],
        "BEAM2_NEW_RECOVERIES": metrics["S1"]["new_recovery_count_vs_s0"],
        "TTT48_NEW_RECOVERIES": metrics["S2"]["new_recovery_count_vs_s0"],
        "UNION_NEW_RECOVERIES": metrics["S3"]["new_recovery_count_vs_s0"],
        **extras, "best_accuracy_runtime_tradeoff": winner,
        "conditions": metrics, "solutions_opened_only_after_all_candidate_pools_frozen": True,
        "development_only_target_derived_pool_miss_cohort": True,
    }
    atomic_write_json(args.output_dir / "SMOKE12_TTT_COST_ACCURACY_REPORT.json", report)
    (args.output_dir / "SMOKE12_TTT_COST_ACCURACY_REPORT.md").write_text(
        "# Smoke12 TTT cost/accuracy ablation\n\n" + "\n".join(f"- {key} = `{value}`" for key, value in report.items() if key != "conditions") + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
