"""Freeze B-support predictions, then score the TTT48/TTT24 paired replay."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
from inference.kaggle_l4_parallel_runner import atomic_write_json
from scripts.run_eval60_reference_ttt_4gpu import _read, _task_hash
from scripts.run_eval60_ttt48_paired_confirmation import FROZEN_STATUS, MANIFEST_STATUS
from scripts.score_eval60_reference_ttt_4gpu import _select

TTT24_CANDIDATE_STATUS = "EVAL60_REFERENCE_TTT_4GPU_CANDIDATES_FROZEN_BEFORE_EXACT_SCORING"
TTT24_PREDICTION_STATUS = "EVAL60_REFERENCE_TTT_4GPU_PREDICTIONS_FROZEN_BEFORE_EXACT_SCORING"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _score_prediction(prediction: dict[str, Any], target: Any) -> tuple[bool, bool]:
    first = prediction.get("attempt_1") == target
    return first, first or prediction.get("attempt_2") == target


def _bucket(ttt24_top2: bool, ttt48_top2: bool) -> str:
    if ttt24_top2 and ttt48_top2:
        return "BOTH"
    if ttt48_top2:
        return "NEW_48"
    if ttt24_top2:
        return "LOST_48"
    return "UNSOLVED_BOTH"


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("manifest", "ttt24_candidates", "ttt24_predictions", "ttt24_report", "candidates", "challenge_path", "solutions_path", "output_dir"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    required_names = ("b_support_selection_frozen.json", "predictions_frozen.json", "paired_comparison.csv", "runtime_per_task.csv", "EVAL60_TTT48_REPORT.json", "EVAL60_TTT48_REPORT.md")
    if any((args.output_dir / name).exists() for name in required_names):
        raise FileExistsError("refusing to overwrite a frozen TTT48 paired score")
    manifest, ttt24, old_predictions, old_report, candidates = map(_read, (args.manifest, args.ttt24_candidates, args.ttt24_predictions, args.ttt24_report, args.candidates))
    task_ids = list(manifest.get("task_ids", ()))
    if (manifest.get("status") != MANIFEST_STATUS or len(task_ids) != 60 or manifest.get("task_ids_hash") != _task_hash(task_ids)
            or candidates.get("status") != FROZEN_STATUS or set(candidates.get("records", ())) != set(task_ids)
            or candidates.get("task_ids_hash") != manifest["task_ids_hash"]
            or ttt24.get("status") != TTT24_CANDIDATE_STATUS or old_predictions.get("status") != TTT24_PREDICTION_STATUS
            or set(ttt24.get("records", ())) != set(task_ids) or set(old_predictions.get("records", ())) != set(task_ids)):
        raise ValueError("incomplete/mismatched paired Eval60 freeze")
    if candidates["reference_config"].get("ttt_steps") != 48 or ttt24["reference_config"].get("ttt_steps") != 24:
        raise ValueError("paired TTT step contract violated")
    if old_report.get("TTT_ANY_OF_K") != "12/60" or old_report.get("TTT_TOP1") != "12/60" or old_report.get("TTT_TOP2") != "12/60":
        raise ValueError("frozen TTT24 strong baseline is not 12/60")
    if _sha256(args.challenge_path) != manifest["source_challenge_sha256"]:
        raise ValueError("challenge changed from frozen Eval60")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selections: dict[str, Any] = {}
    predictions: dict[str, Any] = {}
    runtime_rows: list[dict[str, Any]] = []
    for task_id in task_ids:
        record = candidates["records"][task_id]
        integrity = record.get("ttt", {})
        if (record.get("adapter_reset_success") is not True or integrity.get("train_pairs_only") is not True
                or integrity.get("loss_finite") is not True or integrity.get("base_model_unchanged") is not True):
            raise ValueError(f"{task_id}: invalid TTT48 integrity")
        selection, first, second = _select(record)
        selections[task_id] = {"task_id": task_id, "public_reference_selection": selection, "candidate_artifact_record_sha256": hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}
        predictions[task_id] = {"attempt_1": first, "attempt_2": second, "attempt_candidate_indices": selection["attempt_candidate_indices"], "status": selection["status"], "worker_id": record["worker_id"], "physical_gpu_id": record["physical_gpu_id"]}
        runtime_rows.append({"task_id": task_id, "worker_id": record["worker_id"], "physical_gpu_id": record["physical_gpu_id"], "ttt_seconds": record["ttt_seconds"], "generation_seconds": record["generation_seconds"], "original_likelihood_seconds": record["original_likelihood_seconds"], "b_support_scoring_seconds": record["b_support_scoring_seconds"], "total_task_seconds": record["elapsed_seconds"], "peak_vram_mb": record["peak_allocated_vram_mb"], "generated_candidate_count": record["generated_candidate_count"], "unique_candidate_count": record["unique_candidate_count"], "invalid_candidate_count": record["invalid_candidate_count"]})
    selection_artifact = {"experiment_id": candidates["experiment_id"], "status": "EVAL60_TTT48_B_SUPPORT_SELECTION_FROZEN_BEFORE_EXACT_SCORING", "task_ids": task_ids, "task_ids_hash": manifest["task_ids_hash"], "records": selections, "source_sha256": _sha256(args.candidates)}
    prediction_artifact = {"experiment_id": candidates["experiment_id"], "status": "EVAL60_TTT48_PREDICTIONS_FROZEN_BEFORE_EXACT_SCORING", "solutions_opened": False, "task_ids": task_ids, "task_ids_hash": manifest["task_ids_hash"], "records": predictions, "candidate_artifact_sha256": _sha256(args.candidates), "ttt24_prediction_artifact_sha256": _sha256(args.ttt24_predictions)}
    atomic_write_json(args.output_dir / "b_support_selection_frozen.json", selection_artifact)
    atomic_write_json(args.output_dir / "predictions_frozen.json", prediction_artifact)
    with (args.output_dir / "runtime_per_task.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(runtime_rows[0])); writer.writeheader(); writer.writerows(runtime_rows)
    # The boundary is intentional: every target-blind output above must exist
    # before this next line can access evaluation targets.
    solutions = _read(args.solutions_path)
    if set(task_ids) - set(solutions):
        raise ValueError("solutions omit frozen Eval60 task")
    counts = {"anyk": 0, "top1": 0, "top2": 0, "pool_miss": 0, "selection_miss": 0}
    groups = {"BOTH": [], "NEW_48": [], "LOST_48": [], "UNSOLVED_BOTH": []}
    comparison_rows: list[dict[str, Any]] = []
    for task_id in task_ids:
        target, record = solutions[task_id], candidates["records"][task_id]
        t48_top1, t48_top2 = _score_prediction(predictions[task_id], target)
        t24_top1, t24_top2 = _score_prediction(old_predictions["records"][task_id], target)
        pool_hit = any(candidate["prediction"] == target for candidate in record["candidates"])
        counts["anyk"] += int(pool_hit); counts["top1"] += int(t48_top1); counts["top2"] += int(t48_top2)
        counts["pool_miss"] += int(not pool_hit); counts["selection_miss"] += int(pool_hit and not t48_top2)
        bucket = _bucket(t24_top2, t48_top2); groups[bucket].append(task_id)
        comparison_rows.append({"task_id": task_id, "ttt24_top1": t24_top1, "ttt24_top2": t24_top2, "ttt48_any_of_k": pool_hit, "ttt48_top1": t48_top1, "ttt48_top2": t48_top2, "paired_bucket": bucket})
    with (args.output_dir / "paired_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison_rows[0])); writer.writeheader(); writer.writerows(comparison_rows)
    task_seconds = [float(row["total_task_seconds"]) for row in runtime_rows]
    wall = float(candidates["runtime_seconds"])
    worker_loads = {str(worker): sum(float(row["total_task_seconds"]) for row in runtime_rows if row["worker_id"] == worker) for worker in range(4)}
    ttt24_task_gpu = sum(float(record["elapsed_seconds"]) for record in ttt24["records"].values())
    ttt48_task_gpu = float(candidates.get("task_gpu_seconds", sum(task_seconds)))
    ttt24_wall = float(old_report["TOTAL_WALL_CLOCK_SECONDS"])
    p95 = statistics.quantiles(task_seconds, n=100, method="inclusive")[94]
    projection = 240.0 / 60.0 * wall
    conservative = max(projection, max(worker_loads.values()) * 4.0 + max(0.0, wall - max(worker_loads.values()))) + max(0.0, p95 - statistics.median(task_seconds))
    net = len(groups["NEW_48"]) - len(groups["LOST_48"])
    runtime_safe = conservative <= 10 * 3600
    decision = "TTT48_GO" if net >= 3 and runtime_safe else "TTT48_WEAK_GO" if net in {1, 2} and runtime_safe else "TTT48_NO_GAIN"
    report = {"experiment_id": candidates["experiment_id"], "status": "COMPLETE_SCORED_AFTER_TARGET_BLIND_FREEZE", "TTT24_TOP2": "12/60", "TTT48_ANYK": f"{counts['anyk']}/60", "TTT48_TOP1": f"{counts['top1']}/60", "TTT48_TOP2": f"{counts['top2']}/60", "TTT48_POOL_MISS": counts["pool_miss"], "TTT48_SELECTION_MISS": counts["selection_miss"], "NEW_SOLVES_48": groups["NEW_48"], "REGRESSIONS_48": groups["LOST_48"], "NET_GAIN": net, "TASKS_SOLVED_BY_BOTH": groups["BOTH"], "TASKS_UNSOLVED_BY_BOTH": groups["UNSOLVED_BOTH"], "TTT48_TOTAL_WALL_CLOCK_SECONDS": wall, "TTT48_GPU_SECONDS_TOTAL": float(candidates.get("gpu_seconds_total", ttt48_task_gpu)), "TTT48_MEAN_TTT_SECONDS": statistics.fmean(float(row["ttt_seconds"]) for row in runtime_rows), "TTT48_MEAN_GENERATION_SECONDS": statistics.fmean(float(row["generation_seconds"]) for row in runtime_rows), "TTT48_PEAK_VRAM_GB": max(float(row["peak_vram_mb"]) for row in runtime_rows) / 1024.0, "EXTRA_GPU_SECONDS": ttt48_task_gpu - ttt24_task_gpu, "EXTRA_GPU_SECONDS_SCOPE": "comparable task-stage GPU seconds; TTT24 model-startup telemetry was not persisted", "EXTRA_WALL_CLOCK_SECONDS": wall - ttt24_wall, "GPU_SECONDS_PER_NET_NEW_SOLVE": None if net <= 0 else (ttt48_task_gpu - ttt24_task_gpu) / net, "PROJECTED_240_TTT48_RUNTIME_SECONDS": projection, "PROJECTED_240_TTT48_CONSERVATIVE_SECONDS": conservative, "RUNTIME_RISK": "LOW" if runtime_safe else "MATERIAL", "DECISION": decision, "solutions_opened_only_after_predictions_frozen": True, "candidate_artifact_sha256": _sha256(args.candidates), "ttt24_candidate_artifact_sha256": _sha256(args.ttt24_candidates), "ttt24_prediction_artifact_sha256": _sha256(args.ttt24_predictions)}
    atomic_write_json(args.output_dir / "EVAL60_TTT48_REPORT.json", report)
    (args.output_dir / "EVAL60_TTT48_REPORT.md").write_text("# Eval60 TTT48 paired confirmation\n\n" + "\n".join(f"- {key} = `{value}`" for key, value in report.items()) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
