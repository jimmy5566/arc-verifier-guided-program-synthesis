"""Score the frozen Frozen60 augmentation ablation only after all pools freeze.

This script deliberately has no model dependency.  It validates every
candidate/selection artifact before it opens the solution file, then produces
all condition metrics and paired task deltas from identical task membership.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import Any


# The Kaggle orchestrator executes this script by absolute path, which leaves
# only ``scripts/`` on sys.path.  Add the project package explicitly so the
# post-freeze CPU scorer has the same import contract locally and on Kaggle.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


FROZEN_CANDIDATES = "CANDIDATES_AND_RANKED_PREDICTIONS_FROZEN_BEFORE_EXACT_SCORING"
FROZEN_SELECTION = "PUBLIC_REFERENCE_SELECTION_FROZEN_BEFORE_EXACT_SCORING"
FROZEN_SUPPORT = "PREDICTIONS_FROZEN_BEFORE_EXACT_SCORING"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def grid_valid(grid: Any) -> bool:
    return (
        isinstance(grid, list)
        and bool(grid)
        and all(isinstance(row, list) and row and len(row) == len(grid[0]) for row in grid)
        and all(isinstance(cell, int) and 0 <= cell <= 9 for row in grid for cell in row)
    )


def prediction_valid(prediction: Any, expected_outputs: int) -> bool:
    return isinstance(prediction, list) and len(prediction) == expected_outputs and all(grid_valid(grid) for grid in prediction)


def candidate_attempts(selection: dict[str, Any], task_id: str) -> tuple[list[Any], list[int]]:
    record = selection["records"][task_id]
    # Reused 32 B-SUPPORT artifact stores predictions directly.  New 4/8/16
    # selection artifacts retain candidates and deterministic selected indices.
    if "attempt_1" in record:
        indices = [int(value) for value in record.get("attempt_candidate_indices", ())]
        # The historical frozen B-SUPPORT artifact represents a one-candidate
        # portfolio as ``attempt_2: null``.  Production's two-attempt
        # transport duplicates attempt 1 in that case; do the same here
        # without changing the frozen candidate pool or selector outcome.
        attempt_1 = record["attempt_1"]
        attempt_2 = record.get("attempt_2") or attempt_1
        return [attempt_1, attempt_2], indices
    candidates = record.get("candidates", ())
    indices = [int(value) for value in record.get("public_reference_selection", {}).get("attempt_candidate_indices", ())]
    if not candidates or not indices or len(indices) > 2 or any(index < 0 or index >= len(candidates) for index in indices):
        raise ValueError(f"{task_id}: selection lacks valid attempt indices")
    attempts = [candidates[index]["prediction"] for index in indices]
    if len(attempts) == 1:
        attempts.append(attempts[0])
    return attempts, indices


def validate_condition(
    *, label: str,
    candidate_path: Path,
    selection_path: Path,
    task_ids: list[str],
    task_hash: str,
    config_hash: str,
    historical: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidates = read_json(candidate_path)
    selection = read_json(selection_path)
    if candidates.get("status") != FROZEN_CANDIDATES:
        raise ValueError(f"{label}: candidate artifact is not complete/frozen")
    records = candidates.get("records")
    if not isinstance(records, dict) or set(records) != set(task_ids):
        raise ValueError(f"{label}: candidate task IDs differ from Frozen60")
    if candidates.get("task_ids_hash") != task_hash or candidates.get("config_sha256") != config_hash:
        raise ValueError(f"{label}: candidate cohort/config identity mismatch")
    if int(candidates.get("stage_augmentation_count", -1)) != int(label) or int(candidates.get("stage_worker_count", -1)) != 4:
        raise ValueError(f"{label}: candidate stage identity mismatch")
    if historical:
        if selection.get("status") != FROZEN_SUPPORT or selection.get("task_ids_hash") != task_hash:
            raise ValueError("32: reused B-SUPPORT predictions are not frozen or do not match Frozen60")
        if selection.get("source_candidate_artifact_sha256") != sha256(candidate_path):
            raise ValueError("32: reused B-SUPPORT artifact was created from a different candidate pool")
    else:
        if selection.get("status") != FROZEN_SELECTION or set(selection.get("records", ())) != set(task_ids):
            raise ValueError(f"{label}: B-SUPPORT selection is not complete/frozen")
        if selection.get("task_ids_hash") != task_hash or selection.get("config_sha256") != config_hash:
            raise ValueError(f"{label}: selection cohort/config identity mismatch")
        if selection.get("public_reference_source_sha256") != sha256(candidate_path):
            raise ValueError(f"{label}: selection was created from a different candidate artifact")
    return candidates, selection


def task_metrics(candidates: dict[str, Any], selection: dict[str, Any], task_ids: list[str], solutions: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    details: list[dict[str, Any]] = []
    generated = unique = invalid = 0
    support_values: list[int] = []
    output_top1 = output_top2 = output_total = 0
    for task_id in task_ids:
        record = candidates["records"][task_id]
        target = solutions[task_id]
        attempts, indices = candidate_attempts(selection, task_id)
        if not prediction_valid(attempts[0], len(target)) or not prediction_valid(attempts[1], len(target)):
            raise ValueError(f"{task_id}: malformed selected prediction")
        pool = record.get("candidates", ())
        if not isinstance(pool, list):
            raise ValueError(f"{task_id}: malformed candidate pool")
        if not all(prediction_valid(candidate.get("prediction"), len(target)) for candidate in pool):
            raise ValueError(f"{task_id}: malformed candidate grid")
        generated += int(record.get("generated_candidate_count", 0))
        unique += int(record.get("unique_candidate_count", len(pool)))
        invalid += int(record.get("invalid_candidate_count", 0))
        support_values.extend(int(candidate.get("support_count", 1)) for candidate in pool)
        any_of_k = any(candidate["prediction"] == target for candidate in pool)
        top1 = attempts[0] == target
        top2 = top1 or attempts[1] == target
        per_output_top1 = [attempts[0][index] == target[index] for index in range(len(target))]
        per_output_top2 = [per_output_top1[index] or attempts[1][index] == target[index] for index in range(len(target))]
        output_top1 += sum(per_output_top1); output_top2 += sum(per_output_top2); output_total += len(target)
        details.append({
            "task_id": task_id,
            "any_of_k": any_of_k,
            "top1": top1,
            "top2": top2,
            "attempt_candidate_indices": indices,
            "test_output_count": len(target),
            "per_output_top1": per_output_top1,
            "per_output_top2": per_output_top2,
        })
    return {
        "task_count": len(task_ids),
        "output_count": output_total,
        "any_of_k": sum(item["any_of_k"] for item in details),
        "top1": sum(item["top1"] for item in details),
        "top2": sum(item["top2"] for item in details),
        "output_top1": output_top1,
        "output_top2": output_top2,
        "raw_valid_candidate_count": generated - invalid,
        "generated_candidate_count": generated,
        "mean_unique_candidates": statistics.fmean(int(candidates["records"][task_id].get("unique_candidate_count", 0)) for task_id in task_ids),
        "median_unique_candidates": statistics.median(int(candidates["records"][task_id].get("unique_candidate_count", 0)) for task_id in task_ids),
        "mean_support_count": statistics.fmean(support_values) if support_values else 0.0,
        "invalid_candidate_total": invalid,
    }, details


def runtime_metrics(candidates: dict[str, Any], selection: dict[str, Any], historical_selection: dict[str, Any] | None = None) -> dict[str, float]:
    task_count = int(candidates["stage_task_count"])
    candidate_wall = float(candidates["runtime_seconds"])
    # New conditions score B-support inline inside each GPU worker.  The
    # historical 32 artifact predates that transport optimization, so retain
    # its recorded serial B-selection duration rather than estimating it.
    selection_wall = float((historical_selection or selection).get("public_reference_selection_runtime_seconds", 0.0))
    wall = candidate_wall + selection_wall
    generation = float(candidates.get("generation_seconds_sum", 0.0))
    likelihood = float(candidates.get("original_likelihood_seconds_sum", 0.0))
    b_support = float(candidates.get("b_support_scoring_seconds_sum", 0.0))
    return {
        "wall_clock_seconds": wall,
        "candidate_generation_wall_seconds": candidate_wall,
        "selection_wall_seconds": selection_wall,
        "generation_seconds": generation,
        "original_likelihood_seconds": likelihood,
        "b_support_seconds": b_support,
        "tasks_per_hour": task_count / (wall / 3600.0),
        "peak_vram_mb_recorded": float(candidates.get("peak_allocated_vram_mb", 0.0)),
    }


def paired_rows(left: str, right: str, left_details: list[dict[str, Any]], right_details: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_left, by_right = ({row["task_id"]: row for row in rows} for rows in (left_details, right_details))
    rows: list[dict[str, Any]] = []
    counters = {key: [] for key in ("new_top2", "lost_top2", "new_any_of_k", "lost_any_of_k", "ranking_only_gains", "ranking_only_losses")}
    for task_id in sorted(by_left):
        before, after = by_left[task_id], by_right[task_id]
        new_top2 = not before["top2"] and after["top2"]
        lost_top2 = before["top2"] and not after["top2"]
        new_any = not before["any_of_k"] and after["any_of_k"]
        lost_any = before["any_of_k"] and not after["any_of_k"]
        # Candidate recall was already available in the respective pool, so a
        # top-2 change without an Any-of-K change is selection/ranking-only.
        ranking_gain = new_top2 and before["any_of_k"] == after["any_of_k"]
        ranking_loss = lost_top2 and before["any_of_k"] == after["any_of_k"]
        for key, condition in (("new_top2", new_top2), ("lost_top2", lost_top2), ("new_any_of_k", new_any), ("lost_any_of_k", lost_any), ("ranking_only_gains", ranking_gain), ("ranking_only_losses", ranking_loss)):
            if condition:
                counters[key].append(task_id)
        rows.append({
            "transition": f"{left}->{right}", "task_id": task_id,
            f"any_of_k_{left}": before["any_of_k"], f"any_of_k_{right}": after["any_of_k"],
            f"top2_{left}": before["top2"], f"top2_{right}": after["top2"],
            "new_top2": new_top2, "lost_top2": lost_top2,
            "new_any_of_k": new_any, "lost_any_of_k": lost_any,
            "ranking_only_gain": ranking_gain, "ranking_only_loss": ranking_loss,
        })
    return {"from": int(left), "to": int(right), **counters, "net_top2_gain": len(counters["new_top2"]) - len(counters["lost_top2"])}, rows


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--solutions-path", type=Path, required=True)
    parser.add_argument("--aug4-candidates", type=Path, required=True)
    parser.add_argument("--aug4-selection", type=Path, required=True)
    parser.add_argument("--aug8-candidates", type=Path, required=True)
    parser.add_argument("--aug8-selection", type=Path, required=True)
    parser.add_argument("--aug16-candidates", type=Path, required=True)
    parser.add_argument("--aug16-selection", type=Path, required=True)
    parser.add_argument("--aug32-candidates", type=Path, required=True)
    parser.add_argument("--aug32-support-selection", type=Path, required=True)
    parser.add_argument("--aug32-historical-selection", type=Path, required=True, help="historical 32 B-selection artifact holding its recorded runtime")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = read_json(args.manifest)
    task_ids = list(manifest.get("task_ids", ()))
    task_hash = str(manifest.get("task_ids_hash", ""))
    config_hash = str(manifest.get("selection", {}).get("method_a_config_sha256", ""))
    if manifest.get("status") != "COHORT_FROZEN_BEFORE_INFERENCE_AND_TARGET_ACCESS" or len(task_ids) != 60:
        raise ValueError("requires the exact frozen untouched60 manifest")
    if hashlib.sha256(json.dumps(sorted(task_ids), separators=(",", ":")).encode()).hexdigest() != task_hash:
        raise ValueError("Frozen60 task hash mismatch")

    # This complete validation is intentionally before load_solutions.
    paths = {
        4: (args.aug4_candidates, args.aug4_selection, False),
        8: (args.aug8_candidates, args.aug8_selection, False),
        16: (args.aug16_candidates, args.aug16_selection, False),
        32: (args.aug32_candidates, args.aug32_support_selection, True),
    }
    frozen = {count: validate_condition(label=str(count), candidate_path=candidate, selection_path=selection, task_ids=task_ids, task_hash=task_hash, config_hash=config_hash, historical=historical) for count, (candidate, selection, historical) in paths.items()}
    historical_selection = read_json(args.aug32_historical_selection)
    if historical_selection.get("status") != FROZEN_SELECTION or historical_selection.get("task_ids_hash") != task_hash:
        raise ValueError("32: historical selection runtime source is not a Frozen60 selection artifact")
    if historical_selection.get("public_reference_source_sha256") != sha256(args.aug32_candidates):
        raise ValueError("32: historical selection runtime source belongs to another candidate pool")
    print(json.dumps({"event": "FROZEN60_ALL_4_8_16_32_PREDICTIONS_VALIDATED_BEFORE_SCORING", "task_ids_hash": task_hash, "solutions_opened": False}, sort_keys=True), flush=True)

    from arc.io import load_solutions
    solutions = load_solutions(args.solutions_path)
    if set(task_ids) - set(solutions):
        raise ValueError("Frozen60 solutions do not cover every frozen task")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[int, dict[str, Any]] = {}
    details: dict[int, list[dict[str, Any]]] = {}
    for count, (candidates, selection) in frozen.items():
        metric, condition_details = task_metrics(candidates, selection, task_ids, solutions)
        runtime = runtime_metrics(candidates, selection, historical_selection if count == 32 else None)
        metric.update(runtime)
        metric["any_of_k_percent"] = 100.0 * metric["any_of_k"] / 60.0
        metric["top1_percent"] = 100.0 * metric["top1"] / 60.0
        metric["top2_percent"] = 100.0 * metric["top2"] / 60.0
        metric["output_top1_percent"] = 100.0 * metric["output_top1"] / metric["output_count"]
        metric["output_top2_percent"] = 100.0 * metric["output_top2"] / metric["output_count"]
        metric["top2_per_gpu_hour"] = metric["top2"] / (metric["wall_clock_seconds"] * 4.0 / 3600.0)
        metric["any_of_k_per_gpu_hour"] = metric["any_of_k"] / (metric["wall_clock_seconds"] * 4.0 / 3600.0)
        metric["estimated_240_task_wall_seconds"] = metric["wall_clock_seconds"] * 4.0
        metric["candidate_artifact_sha256"] = sha256(paths[count][0])
        metric["selection_artifact_sha256"] = sha256(paths[count][1])
        metric["historical_32_reused"] = count == 32
        results[count], details[count] = metric, condition_details
        payload = {"status": "SCORED_AFTER_ALL_4_8_16_32_PREDICTIONS_FROZEN", "augmentation_count": count, "metrics": metric, "per_task": condition_details}
        name = f"frozen60_aug{count}_results.json" if count != 32 else "frozen60_aug32_reused.json"
        write_json(args.output_dir / name, payload)

    paired: dict[str, Any] = {}
    csv_rows: list[dict[str, Any]] = []
    for left, right in ((4, 8), (8, 16), (16, 32)):
        item, rows = paired_rows(str(left), str(right), details[left], details[right])
        extra_gpu_hours = (results[right]["wall_clock_seconds"] - results[left]["wall_clock_seconds"]) * 4.0 / 3600.0
        item["marginal_top2_per_extra_gpu_hour"] = item["net_top2_gain"] / extra_gpu_hours if extra_gpu_hours > 0 else None
        item["marginal_any_of_k_per_extra_gpu_hour"] = (results[right]["any_of_k"] - results[left]["any_of_k"]) / extra_gpu_hours if extra_gpu_hours > 0 else None
        paired[f"{left}_to_{right}"] = item; csv_rows.extend(rows)
    with (args.output_dir / "paired_task_deltas.csv").open("w", newline="", encoding="utf-8") as handle:
        # Transitions expose different condition-labelled columns (for
        # example ``top2_4`` versus ``top2_16``), therefore the CSV schema is
        # the deterministic union of all rows rather than the first row.
        fieldnames = sorted({key for row in csv_rows for key in row}) if csv_rows else ["transition", "task_id"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(csv_rows)

    max_top2 = max(item["top2"] for item in results.values())
    feasible = [count for count, item in results.items() if item["estimated_240_task_wall_seconds"] <= 10.25 * 3600.0]
    contenders = [count for count in feasible if results[count]["top2"] == max(results[item]["top2"] for item in feasible)] if feasible else []
    recommended = min(contenders) if contenders else max(results, key=lambda item: results[item]["top2"])
    report = {
        "experiment_id": "ARC2_FROZEN60_AUGMENTATION_ABLATION_4_8_16_32",
        "status": "COMPLETE_SCORED_AFTER_ALL_CONDITIONS_FROZEN",
        "frozen60_hash": task_hash,
        "scientific_identity": {"config_sha256": config_hash, "manifest_sha256": sha256(args.manifest), "solutions_opened_only_after_all_predictions_frozen": True},
        "conditions": {str(key): value for key, value in results.items()},
        "paired": paired,
        "recommendation": {"augmentation_count": recommended, "verdict": f"MOVE_TO_{recommended}" if recommended != 32 else "KEEP_32", "decision_rule": "Among augmentation counts whose 240-task linear runtime extrapolation is <=10.25h, select the lowest count with maximal Frozen60 Top-2; if none are feasible, select maximal Frozen60 Top-2.", "note": "240-task values are extrapolations, not measured production runtimes."},
    }
    write_json(args.output_dir / "AUGMENTATION_ABLATION_REPORT.json", report)
    lines = ["# Frozen60 Augmentation Ablation", "", "All 4/8/16 candidate and B-support selection artifacts were validated/frozen before solutions were loaded.", "", "| Aug | Any-of-K | Top-1 | Top-2 | Avg Unique Candidates | Runtime s | Relative Runtime | Top2/GPU-hour |", "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    baseline_runtime = results[32]["wall_clock_seconds"]
    for count in (4, 8, 16, 32):
        item = results[count]
        lines.append(f"| {count} | {item['any_of_k']}/60 | {item['top1']}/60 | {item['top2']}/60 | {item['mean_unique_candidates']:.3f} | {item['wall_clock_seconds']:.3f} | {item['wall_clock_seconds'] / baseline_runtime:.3f}x | {item['top2_per_gpu_hour']:.4f} |")
    lines.extend(["", "## Recommendation", "", f"`{report['recommendation']['verdict']}`. {report['recommendation']['decision_rule']}", "", "The 32-augmentation condition reuses its historical recorded runtime and was not rerun."])
    (args.output_dir / "AUGMENTATION_ABLATION_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for count in (4, 8, 16, 32):
        item = results[count]
        print(f"AUG{count}_ANYK = {item['any_of_k']}/60")
        print(f"AUG{count}_TOP1 = {item['top1']}/60")
        print(f"AUG{count}_TOP2 = {item['top2']}/60")
        print(f"AUG{count}_RUNTIME = {item['wall_clock_seconds']:.3f}s")
    print(f"FROZEN60_HASH = {task_hash}")
    print(f"BEST_ACCURACY = {max_top2}/60")
    print(f"BEST_RUNTIME_ADJUSTED = {recommended}")
    print(f"RECOMMENDED_PRODUCTION_AUG_COUNT = {recommended}")


if __name__ == "__main__":
    main()
