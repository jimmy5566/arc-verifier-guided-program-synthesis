"""Post-freeze, solution-aware Evaluation60 Aug8 scorer.

All selection and prediction validation happens before the solution file is
read.  This provides an auditable target-blind boundary for the experiment.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any


FROZEN = "EVAL60_AUG8_PREDICTIONS_FROZEN_BEFORE_EXACT_SCORING"
HISTORICAL = {"any_of_k": 21, "top1": 19, "top2": 19, "runtime_seconds": 1597.772}


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _grid(value: Any) -> bool:
    return (
        isinstance(value, list) and bool(value)
        and all(isinstance(row, list) and row and len(row) == len(value[0]) for row in value)
        and all(isinstance(cell, int) and 0 <= cell <= 9 for row in value for cell in row)
    )


def _diagnosis(any_of_k: int, top2: int) -> str:
    # Predeclared before targets are opened: Frozen60 has AnyK=21/60 and
    # Top2=19/60.  These thresholds make "near zero" and selection collapse
    # explicit rather than being adjusted after the evaluation result.
    if any_of_k <= 3:
        return "SEVERE_GENERATION_GENERALIZATION_FAILURE"
    if any_of_k >= 8 and top2 <= 3:
        return "SEVERE_SELECTION_GENERALIZATION_FAILURE"
    if any_of_k >= 14 and top2 >= 12:
        return "HEALTHY_GENERALIZATION"
    return "MODERATE_GENERALIZATION_GAP"


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("manifest", "candidates", "selection", "predictions", "challenge_path", "solutions_path", "output_dir"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    if any(path.exists() for path in (args.output_dir / "EVAL60_AUG8_REPORT.json", args.output_dir / "EVAL60_AUG8_REPORT.md", args.output_dir / "per_task_results.csv")):
        raise FileExistsError("refusing to overwrite a completed Evaluation60 score report")
    manifest, candidates, selection, frozen = (_read(path) for path in (args.manifest, args.candidates, args.selection, args.predictions))
    task_ids = list(manifest.get("task_ids", ()))
    task_hash = hashlib.sha256(json.dumps(sorted(task_ids), separators=(",", ":")).encode()).hexdigest()
    expected = set(task_ids)
    if (
        len(task_ids) != 60
        or manifest.get("task_ids_hash") != task_hash
        or frozen.get("status") != FROZEN
        or frozen.get("task_ids_hash") != task_hash
        or frozen.get("source_challenge_sha256") != _sha256(args.challenge_path)
        or frozen.get("candidate_artifact_sha256") != _sha256(args.candidates)
        or frozen.get("selection_artifact_sha256") != _sha256(args.selection)
        or set(frozen.get("records", ())) != expected
        or set(candidates.get("records", ())) != expected
        or set(selection.get("records", ())) != expected
    ):
        raise ValueError("predictions are not a complete immutable Evaluation60 target-blind freeze")
    # This is the first permitted target access.
    solutions = _read(args.solutions_path)
    if any(task_id not in solutions for task_id in task_ids):
        raise ValueError("evaluation solutions omit a frozen task")
    rows: list[dict[str, Any]] = []
    output_top1 = output_top2 = output_total = 0
    any_of_k = top1 = top2 = 0
    pool_miss = selection_miss = 0
    generated = unique = invalid = 0
    runtimes: list[float] = []
    for task_id in task_ids:
        target = solutions[task_id]
        prediction = frozen["records"][task_id]
        candidate_record = candidates["records"][task_id]
        attempt_1, attempt_2 = prediction["attempt_1"], prediction["attempt_2"]
        pool = candidate_record.get("candidates", ())
        if not isinstance(target, list) or len(target) != len(attempt_1) or len(target) != len(attempt_2):
            raise ValueError(f"{task_id}: test-output mapping differs from frozen predictions")
        if not all(_grid(grid) for output_set in (attempt_1, attempt_2) for grid in output_set):
            raise ValueError(f"{task_id}: invalid frozen selected grid")
        pool_exact = any(item.get("prediction") == target for item in pool)
        top1_exact = attempt_1 == target
        top2_exact = top1_exact or attempt_2 == target
        if not pool_exact:
            category = "A_POOL_MISS"; pool_miss += 1
        elif not top2_exact:
            category = "B_SELECTION_MISS"; selection_miss += 1
        else:
            category = "C_TOP2_SOLVED"
        per_output_1 = [attempt_1[index] == target[index] for index in range(len(target))]
        per_output_2 = [per_output_1[index] or attempt_2[index] == target[index] for index in range(len(target))]
        output_top1 += sum(per_output_1); output_top2 += sum(per_output_2); output_total += len(target)
        any_of_k += int(pool_exact); top1 += int(top1_exact); top2 += int(top2_exact)
        generated += int(candidate_record.get("generated_candidate_count", 0))
        unique += int(candidate_record.get("unique_candidate_count", len(pool)))
        invalid += int(candidate_record.get("invalid_candidate_count", 0))
        runtimes.append(float(prediction.get("elapsed_seconds", 0.0)))
        rows.append({
            "task_id": task_id, "test_output_count": len(target), "classification": category,
            "any_of_k": pool_exact, "top1": top1_exact, "top2": top2_exact,
            "per_output_top1": json.dumps(per_output_1), "per_output_top2": json.dumps(per_output_2),
            "candidate_count": int(prediction["candidate_count"]), "generated_candidate_count": int(prediction["generated_candidate_count"]),
            "invalid_candidate_count": int(prediction["invalid_candidate_count"]), "worker_id": prediction["worker_id"],
            "physical_gpu_id": prediction["physical_gpu_id"], "elapsed_seconds": prediction["elapsed_seconds"],
            "attempt_candidate_indices": json.dumps(prediction["attempt_candidate_indices"]),
        })
    total_runtime = float(candidates.get("runtime_seconds", sum(runtimes)))
    diagnosis = _diagnosis(any_of_k, top2)
    metrics = {
        "task_count": 60, "output_count": output_total, "any_of_k": any_of_k, "top1": top1, "top2": top2,
        "output_top1": output_top1, "output_top2": output_top2,
        "pool_miss_count": pool_miss, "selection_miss_count": selection_miss,
        "generated_candidate_count": generated, "unique_candidate_count": unique, "invalid_candidate_count": invalid,
        "mean_unique_candidates_per_task": unique / 60.0, "mean_generated_candidates_per_task": generated / 60.0,
        "runtime_seconds": total_runtime, "runtime_seconds_per_task": total_runtime / 60.0,
        "mean_task_elapsed_seconds": statistics.fmean(runtimes), "max_task_elapsed_seconds": max(runtimes),
    }
    report = {
        "experiment_id": "ARC2_EVALUATION60_AUG8_GENERALIZATION_DIAGNOSIS",
        "status": "COMPLETE_SCORED_AFTER_TARGET_BLIND_FREEZE",
        "protocol": "Evaluation60 selected from challenge IDs by SHA-256 before inference. Candidates, B selection, and predictions were frozen before solutions opened.",
        "eval60_hash": task_hash, "source_challenge_sha256": manifest["source_challenge_sha256"],
        "candidate_artifact_sha256": _sha256(args.candidates), "selection_artifact_sha256": _sha256(args.selection),
        "prediction_artifact_sha256": _sha256(args.predictions), "metrics": metrics,
        "historical_frozen60_aug8": HISTORICAL,
        "generalization_drop": {"any_of_k": HISTORICAL["any_of_k"] - any_of_k, "top1": HISTORICAL["top1"] - top1, "top2": HISTORICAL["top2"] - top2},
        "diagnosis": diagnosis,
        "decision_rule": {"healthy": "AnyK >=14 and Top2 >=12", "severe_generation": "AnyK <=3", "severe_selection": "AnyK >=8 and Top2 <=3", "otherwise": "moderate"},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "per_task_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (args.output_dir / "EVAL60_AUG8_REPORT.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "EVAL60_AUG8_REPORT.md").write_text(
        "# Evaluation60 Aug8 generalization diagnosis\n\n"
        f"- Eval60 hash: `{task_hash}`\n- Any-of-K: **{any_of_k}/60**\n- Top-1: **{top1}/60**\n- Top-2: **{top2}/60**\n"
        f"- Output Top-1 / Top-2: {output_top1}/{output_total} / {output_top2}/{output_total}\n"
        f"- Pool misses / selection misses: {pool_miss} / {selection_miss}\n"
        f"- Runtime: {total_runtime:.3f}s ({total_runtime / 60.0:.3f}s/task)\n"
        f"- Generalization drop AnyK / Top1 / Top2: {21-any_of_k} / {19-top1} / {19-top2}\n"
        f"- Diagnosis: **{diagnosis}**\n",
        encoding="utf-8",
    )
    print(json.dumps({"EVAL60_HASH": task_hash, "ANY_OF_K": f"{any_of_k}/60", "TOP1": f"{top1}/60", "TOP2": f"{top2}/60", "TOTAL_RUNTIME": total_runtime, "POOL_MISS_COUNT": pool_miss, "SELECTION_MISS_COUNT": selection_miss, "GENERALIZATION_DROP_TOP2": 19-top2, "DIAGNOSIS": diagnosis}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
