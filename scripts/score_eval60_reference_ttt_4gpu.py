"""CPU-only post-freeze scoring and unchanged B-support selection for Eval60 TTT."""
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
from inference.nvarc_native_augmentation import NativeAugmentation
from inference.nvarc_public_reference import PublicReferenceEvidence, grouped_public_reference_ranking, prediction_key, two_attempt_indices
from inference.kaggle_l4_parallel_runner import atomic_write_json
from scripts.run_eval60_reference_ttt_4gpu import FROZEN_STATUS, _read, _task_hash


def _sha256(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()


def _valid_grid(grid: Any) -> bool:
    return isinstance(grid, list) and bool(grid) and all(isinstance(row, list) and row and len(row) == len(grid[0]) and all(isinstance(cell, int) and 0 <= cell <= 9 for cell in row) for row in grid)


def _select(record: dict[str, Any]) -> tuple[dict[str, Any], list[list[list[int]]] | None, list[list[list[int]]] | None]:
    candidates = list(record["candidates"]); views = tuple(NativeAugmentation(geometry=name) for name in ("identity", "rot90", "rot180", "rot270", "flip_lr", "flip_ud", "transpose", "anti_transpose"))
    # A task with no parse-valid generated candidate is a completed,
    # target-blind pool miss, not an execution failure.  Preserve that state
    # explicitly so post-freeze reporting can score it as incorrect without
    # fabricating a grid or aborting the other 59 frozen records.
    if not candidates:
        if record.get("status") != "NO_VALID_NATIVE_CANDIDATE":
            raise ValueError(f"{record['task_id']}: empty candidate pool has invalid status")
        return ({"method": "equivalent_output_support_minus_mean_augmentation_view_nll", "ranked_candidate_indices": [], "attempt_candidate_indices": [], "evidence": [], "status": "NO_VALID_NATIVE_CANDIDATE"}, None, None)
    if record.get("b_support_view_spec") != [view.to_dict() for view in views] or len(record.get("b_support_evidence", ())) != len(candidates):
        raise ValueError(f"{record['task_id']}: incomplete cached B-support evidence")
    original = dict(zip(record["ranked_candidate_indices"], record["candidate_scores"], strict=True))
    evidence = []
    for item in record["b_support_evidence"]:
        index = int(item["candidate_index"])
        if float(item["original_log_likelihood"]) != float(original[index]): raise ValueError("original likelihood mismatch")
        evidence.append(PublicReferenceEvidence(index=index, prediction_key=prediction_key(candidates[index]["prediction"]), original_log_likelihood=float(original[index]), view_negative_log_likelihoods=tuple(float(value) for value in item["view_negative_log_likelihoods"]), support_count=int(candidates[index].get("support_count", 1))))
    ranked, attempts = grouped_public_reference_ranking(evidence), two_attempt_indices(grouped_public_reference_ranking(evidence), candidates)
    if not attempts: raise ValueError(f"{record['task_id']}: B-support produced no attempt")
    first, second = candidates[attempts[0]]["prediction"], candidates[attempts[1] if len(attempts) > 1 else attempts[0]]["prediction"]
    if not all(_valid_grid(grid) for output in (first, second) for grid in output): raise ValueError(f"{record['task_id']}: invalid selected grid")
    return {"method": "equivalent_output_support_minus_mean_augmentation_view_nll", "ranked_candidate_indices": ranked, "attempt_candidate_indices": attempts, "evidence": [{"candidate_index": item.index, "support_count": item.support_count, "mean_view_nll": item.mean_view_nll, "original_log_likelihood": item.original_log_likelihood} for item in evidence], "status": "SUCCESS"}, first, second


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("manifest", "baseline", "candidates", "challenge_path", "solutions_path", "output_dir"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    names = ("b_support_selection_frozen.json", "predictions_frozen.json", "runtime_per_task.csv", "worker_summary.json", "EVAL60_TTT_4GPU_REPORT.json", "EVAL60_TTT_4GPU_REPORT.md")
    if any((args.output_dir / name).exists() for name in names): raise FileExistsError("refusing to overwrite an Eval60 TTT score")
    manifest, baseline, candidates = _read(args.manifest), _read(args.baseline), _read(args.candidates)
    task_ids = list(manifest.get("task_ids", ()))
    if manifest.get("status") != "EVAL60_REFERENCE_TTT_4GPU_COHORT_FROZEN" or len(task_ids) != 60 or manifest.get("task_ids_hash") != _task_hash(task_ids) or candidates.get("status") != FROZEN_STATUS or set(candidates.get("records", ())) != set(task_ids) or candidates.get("task_ids_hash") != manifest["task_ids_hash"] or set(baseline.get("records", ())) != set(task_ids):
        raise ValueError("incomplete/mismatched target-blind Eval60 freeze")
    if _sha256(args.challenge_path) != manifest["source_challenge_sha256"]: raise ValueError("challenge changed from frozen Eval60")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selections: dict[str, Any] = {}; predictions: dict[str, Any] = {}; runtime_rows: list[dict[str, Any]] = []
    for task_id in task_ids:
        record = candidates["records"][task_id]
        if record.get("adapter_reset_success") is not True or record["ttt"].get("train_pairs_only") is not True or record["ttt"].get("loss_finite") is not True or record["ttt"].get("base_model_unchanged") is not True:
            raise ValueError(f"{task_id}: invalid TTT integrity")
        selection, first, second = _select(record); selections[task_id] = {**record, "public_reference_selection": selection}
        predictions[task_id] = {"attempt_1": first, "attempt_2": second, "attempt_candidate_indices": selection["attempt_candidate_indices"], "status": selection["status"], "worker_id": record["worker_id"], "physical_gpu_id": record["physical_gpu_id"]}
        runtime_rows.append({"task_id": task_id, "worker_id": record["worker_id"], "physical_gpu_id": record["physical_gpu_id"], "ttt_seconds": record["ttt_seconds"], "generation_seconds": record["generation_seconds"], "total_task_seconds": record["elapsed_seconds"], "initial_ttt_loss": record["ttt"]["first_loss"], "final_ttt_loss": record["ttt"]["last_loss"], "peak_vram_mb": record["peak_allocated_vram_mb"], "generated_candidate_count": record["generated_candidate_count"], "unique_candidate_count": record["unique_candidate_count"], "invalid_candidate_count": record["invalid_candidate_count"], "adapter_reset_success": record["adapter_reset_success"]})
    selection_artifact = {"experiment_id": candidates["experiment_id"], "status": "PUBLIC_REFERENCE_SELECTION_FROZEN_BEFORE_EXACT_SCORING", "task_ids": task_ids, "task_ids_hash": manifest["task_ids_hash"], "records": selections, "source_sha256": _sha256(args.candidates)}
    prediction_artifact = {"experiment_id": candidates["experiment_id"], "status": "EVAL60_REFERENCE_TTT_4GPU_PREDICTIONS_FROZEN_BEFORE_EXACT_SCORING", "solutions_opened": False, "task_ids": task_ids, "task_ids_hash": manifest["task_ids_hash"], "records": predictions, "candidate_artifact_sha256": _sha256(args.candidates)}
    atomic = atomic_write_json
    atomic(args.output_dir / "b_support_selection_frozen.json", selection_artifact); atomic(args.output_dir / "predictions_frozen.json", prediction_artifact)
    with (args.output_dir / "runtime_per_task.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(runtime_rows[0])); writer.writeheader(); writer.writerows(runtime_rows)
    # Explicit target boundary: selection and all predictions were persisted above.
    solutions = _read(args.solutions_path)
    if set(task_ids) - set(solutions): raise ValueError("solutions omit frozen Eval60 task")
    anyk = top1 = top2 = pool_miss = selection_miss = 0; recovered: list[str] = []
    for task_id in task_ids:
        target, record, selected = solutions[task_id], candidates["records"][task_id], predictions[task_id]
        baseline_pool_hit = any(candidate["prediction"] == target for candidate in baseline["records"][task_id]["candidates"])
        pool_hit = any(candidate["prediction"] == target for candidate in record["candidates"]); first_hit, second_hit = selected["attempt_1"] == target, selected["attempt_2"] == target
        anyk += int(pool_hit); top1 += int(first_hit); top2 += int(first_hit or second_hit)
        if pool_hit and not (first_hit or second_hit): selection_miss += 1
        if not pool_hit: pool_miss += 1
        if pool_hit and not baseline_pool_hit: recovered.append(task_id)
    # The baseline denominator/statistics are frozen in the requested protocol.
    worker_counts = {str(index): sum(row["worker_id"] == index for row in runtime_rows) for index in range(4)}
    worker_loads = {str(index): sum(float(row["total_task_seconds"]) for row in runtime_rows if row["worker_id"] == index) for index in range(4)}
    task_seconds = [float(row["total_task_seconds"]) for row in runtime_rows]; wall = float(candidates["runtime_seconds"]); throughput = len(task_ids) * 3600.0 / wall
    p = lambda q: statistics.quantiles(task_seconds, n=100, method="inclusive")[int(q) - 1]
    # Dynamic scheduling's observed tail is the largest accumulated worker
    # load.  Project four equivalent 60-task waves, retain fixed startup /
    # teardown overhead, then add one observed P95-minus-median tail.  The
    # previous expression multiplied a seconds quantity by 60, a unit error
    # that falsely reported a >100-hour conservative projection.
    max_worker_load = max(worker_loads.values())
    startup_overhead = max(0.0, wall - max_worker_load)
    throughput_projection = 240.0 / throughput * 3600.0
    workload_projection = max_worker_load * 4.0 + startup_overhead
    conservative = max(throughput_projection, workload_projection) + max(0.0, p(95) - statistics.median(task_seconds))
    report = {"experiment_id": candidates["experiment_id"], "status": "COMPLETE_SCORED_AFTER_TARGET_BLIND_FREEZE", "BASELINE_ANYK": "2/60", "TTT_ANY_OF_K": f"{anyk}/60", "TTT_TOP1": f"{top1}/60", "TTT_TOP2": f"{top2}/60", "NEW_RECOVERIES": recovered, "POOL_MISS_COUNT": pool_miss, "SELECTION_MISS_COUNT": selection_miss, "ANYK_GAIN": anyk - 2, "TOP1_GAIN": top1 - 2, "TOP2_GAIN": top2 - 2, "TOTAL_WALL_CLOCK_SECONDS": wall, "TASKS_PER_HOUR_4GPU": throughput, "MEAN_TASK_SECONDS": statistics.fmean(task_seconds), "P50_TASK_SECONDS": statistics.median(task_seconds), "P90_TASK_SECONDS": p(90), "P95_TASK_SECONDS": p(95), "MAX_TASK_SECONDS": max(task_seconds), "GPU_TASK_COUNTS": worker_counts, "PROJECTED_240_FROM_THROUGHPUT_SECONDS": throughput_projection, "PROJECTED_240_CONSERVATIVE_SECONDS": conservative, "DECISION": "TTT_STRONG_GO" if anyk >= 10 and conservative < 10 * 3600 else "TTT_GO" if anyk >= 6 and conservative < 10 * 3600 else "TTT_WEAK" if anyk >= 3 else "TTT_PIVOT", "baseline_is_development_not_untouched": True, "solutions_opened_only_after_predictions_frozen": True}
    atomic(args.output_dir / "worker_summary.json", {"worker_task_counts": worker_counts, "worker_runtime_seconds": worker_loads}); atomic(args.output_dir / "EVAL60_TTT_4GPU_REPORT.json", report)
    (args.output_dir / "EVAL60_TTT_4GPU_REPORT.md").write_text("# Eval60 reference-style TTT 4xL4 confirmation\n\n" + "\n".join(f"- {key} = `{value}`" for key, value in report.items()) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__": main()
