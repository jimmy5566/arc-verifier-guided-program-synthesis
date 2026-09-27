#!/usr/bin/env python3
"""CPU-only retrospective scorer for frozen NONBLIND_DEVELOPMENT_LOO_TRANSFER30."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from inference.kaggle_l4_parallel_runner import atomic_write_json


TOP_LABELS = {1: "LOO_TOP1", 2: "LOO_TOP2", 4: "LOO_TOP4"}
BASELINES = ["FIXED_24_IDENTITY", "FIXED_48_IDENTITY", "STEP1_BEST_FIXED_12_TRANSPOSE"]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def grid_key(grid: Any) -> str | None:
    return canonical(grid) if isinstance(grid, list) and grid else None


def frozen(path: Path) -> dict[str, Any]:
    value = read_json(path)
    if value.get("status") != "FROZEN_BEFORE_ANY_EVAL_SOLUTION_ACCESS" or value.get("solutions_opened") is not False:
        raise RuntimeError(f"unfrozen prediction artifact: {path}")
    return value


def output_target(solutions: dict[str, Any], task_id: str, test_index: int) -> Any:
    return solutions[task_id][test_index]


def prediction_sets(records: list[dict[str, Any]], label: str) -> dict[tuple[str, int], set[str]]:
    result: dict[tuple[str, int], set[str]] = defaultdict(set)
    for row in records:
        if label not in row.get("labels", []):
            continue
        key = grid_key(row.get("prediction"))
        if key is not None:
            result[(str(row["task_id"]), int(row["test_index"]))].add(key)
    return result


def score_sets(*, sets: dict[tuple[str, int], set[str]], keys: list[tuple[str, int]], solutions: dict[str, Any]) -> tuple[int, dict[tuple[str, int], bool]]:
    hits = {}
    for task_id, test_index in keys:
        hits[(task_id, test_index)] = grid_key(output_target(solutions, task_id, test_index)) in sets.get((task_id, test_index), set())
    return sum(hits.values()), hits


def full_tasks(hits: dict[tuple[str, int], bool], keys: list[tuple[str, int]]) -> tuple[int, list[str]]:
    per_task: dict[str, list[bool]] = defaultdict(list)
    for key in keys:
        per_task[key[0]].append(hits[key])
    solved = sorted(task_id for task_id, values in per_task.items() if all(values))
    return len(solved), solved


def main(args: argparse.Namespace) -> None:
    root = args.output.resolve()
    gpu = read_json(root / "GPU_FREEZE_COMPLETE.json")
    cohort = read_json(root / "cohort.json")
    sentinel = read_json(root / "sentinel6.json")
    if gpu.get("solutions_opened") is not False or int(gpu.get("tasks", -1)) != len(cohort["task_ids"]):
        raise RuntimeError("GPU freeze missing, incomplete, or contaminated; refusing solution access")
    main_predictions = frozen(root / "production_predictions_frozen.json")["predictions"]
    baselines = frozen(root / "fixed_baselines_frozen.json")["predictions"]
    full_surface = frozen(root / "sentinel6_full_surface_frozen.json")["predictions"]
    rankings = frozen(root / "loo_rankings_frozen.json")["rankings"]
    solutions = read_json(args.solutions)  # Deliberately after all freeze checks.
    keys = [(task_id, test_index) for task_id in cohort["task_ids"] for test_index in range(next(row["test_output_count"] for row in cohort["entries"] if row["task_id"] == task_id))]
    if any(task_id not in solutions for task_id in cohort["task_ids"]):
        raise RuntimeError("solution file missing a frozen cohort task")

    report: dict[str, Any] = {"experiment_id": cohort["status"], "scope": cohort["scope"], "solutions_opened_after_gpu_freeze": True, "cohort_outputs": len(keys), "cohort_tasks": len(cohort["task_ids"]), "methods": {}}
    all_hits: dict[str, dict[tuple[str, int], bool]] = {}
    for topk, label in TOP_LABELS.items():
        candidate_sets = prediction_sets(main_predictions, label)
        count, hits = score_sets(sets=candidate_sets, keys=keys, solutions=solutions)
        task_count, task_ids = full_tasks(hits, keys)
        name = f"LOO_TOP{topk}"
        all_hits[name] = hits
        report["methods"][name] = {"exact_output_hits": count, "denominator_outputs": len(keys), "fully_solved_tasks": task_count, "fully_solved_task_ids": task_ids}
    for baseline in BASELINES:
        sets: dict[tuple[str, int], set[str]] = defaultdict(set)
        for row in baselines:
            if row.get("baseline") == baseline:
                key = grid_key(row.get("prediction"))
                if key is not None:
                    sets[(str(row["task_id"]), int(row["test_index"]))].add(key)
        count, hits = score_sets(sets=sets, keys=keys, solutions=solutions)
        task_count, task_ids = full_tasks(hits, keys)
        all_hits[baseline] = hits
        report["methods"][baseline] = {"exact_output_hits": count, "denominator_outputs": len(keys), "fully_solved_tasks": task_count, "fully_solved_task_ids": task_ids}

    ranking_map = {row["task_id"]: {int(cell["loo_rank"]): (int(cell["depth"]), str(cell["gen_view"])) for cell in row["all_40_cells"]} for row in rankings}
    sentinel_ids = set(sentinel["task_ids"])
    sentinel_keys = [key for key in keys if key[0] in sentinel_ids]
    full_sets: dict[tuple[str, int], set[str]] = defaultdict(set)
    correct_cells: dict[tuple[str, int], list[tuple[int, str, int]]] = defaultdict(list)
    for row in full_surface:
        key = (str(row["task_id"]), int(row["test_index"]))
        value = grid_key(row.get("prediction"))
        if value is None:
            continue
        full_sets[key].add(value)
        if value == grid_key(output_target(solutions, *key)):
            cell = (int(row["depth"]), str(row["gen_view"]))
            rank = next((rank for rank, item in ranking_map[key[0]].items() if item == cell), None)
            if rank is None:
                raise RuntimeError(f"missing frozen rank for {key} {cell}")
            correct_cells[key].append((cell[0], cell[1], rank))
    oracle_count, oracle_hits = score_sets(sets=full_sets, keys=sentinel_keys, solutions=solutions)
    rank_bins: Counter[str] = Counter()
    for key in sentinel_keys:
        ranks = [item[2] for item in correct_cells.get(key, [])]
        if not ranks: rank_bins["no_correct_cell_exists"] += 1
        elif min(ranks) == 1: rank_bins["rank_1"] += 1
        elif min(ranks) == 2: rank_bins["rank_2"] += 1
        elif min(ranks) <= 4: rank_bins["rank_3_4"] += 1
        elif min(ranks) <= 8: rank_bins["rank_5_8"] += 1
        else: rank_bins["rank_gt_8"] += 1
    retention = {}
    for topk, label in TOP_LABELS.items():
        _, hits = score_sets(sets=prediction_sets(main_predictions, label), keys=sentinel_keys, solutions=solutions)
        retained = sum(hits[key] and oracle_hits[key] for key in sentinel_keys)
        retention[f"TOP{topk}"] = {"hits": retained, "oracle_outputs": oracle_count, "retention": retained / oracle_count if oracle_count else None}
    unique_depth: dict[str, list[str]] = defaultdict(list); unique_view: dict[str, list[str]] = defaultdict(list); unique_cell: dict[str, list[str]] = defaultdict(list)
    for key, cells in correct_cells.items():
        depths = {item[0] for item in cells}; views = {item[1] for item in cells}
        if len(depths) == 1: unique_depth[str(next(iter(depths)))].append(f"{key[0]}:{key[1]}")
        if len(views) == 1: unique_view[next(iter(views))].append(f"{key[0]}:{key[1]}")
        if len(cells) == 1:
            depth, view, _ = cells[0]; unique_cell[f"{depth}:{view}"].append(f"{key[0]}:{key[1]}")
    depth_oracle = sum(bool({item[0] for item in cells}) for cells in correct_cells.values())
    view_oracle = sum(bool({item[1] for item in cells}) for cells in correct_cells.values())
    report["sentinel6"] = {"outputs": len(sentinel_keys), "full_surface_oracle": oracle_count, "loo_topk_oracle_retention": retention, "correct_cell_rank_distribution": dict(sorted(rank_bins.items())), "correct_cells": {f"{task_id}:{test_index}": [{"depth": depth, "view": view, "loo_rank": rank} for depth, view, rank in cells] for (task_id, test_index), cells in correct_cells.items()}, "depth_only_oracle": depth_oracle, "view_only_oracle": view_oracle, "joint_oracle": oracle_count, "unique_depth_contributions": dict(sorted(unique_depth.items())), "unique_view_contributions": dict(sorted(unique_view.items())), "unique_depth_view_contributions": dict(sorted(unique_cell.items()))}
    top1 = report["methods"]["LOO_TOP1"]["exact_output_hits"]
    fixed_best = max(report["methods"][name]["exact_output_hits"] for name in BASELINES)
    top4_retention = retention["TOP4"]["retention"]
    if top1 > fixed_best and top4_retention is not None and top4_retention >= 0.75:
        decision, recommendation = "STRONG_TRANSFER", "EXPAND_LARGE_VALIDATION"
    elif top4_retention is not None and top4_retention >= 0.75 and report["methods"]["LOO_TOP4"]["exact_output_hits"] > top1:
        decision, recommendation = "PORTFOLIO_TRANSFER", "BUILD_LOO_GUIDED_SMALL_PORTFOLIO"
    elif top1 > fixed_best or report["methods"]["LOO_TOP2"]["exact_output_hits"] > fixed_best:
        decision, recommendation = "WEAK_TRANSFER", "REVISE_LOO_AGGREGATION"
    else:
        decision, recommendation = "NO_TRANSFER", "STOP_ADAPTIVE_ROUTING"
    report["decision"] = decision; report["recommendation"] = recommendation
    rows = []
    for task_id, test_index in keys:
        row = {"task_id": task_id, "test_index": test_index, "sentinel": task_id in sentinel_ids}
        for name, hits in all_hits.items(): row[name] = hits[(task_id, test_index)]
        row["sentinel_oracle"] = oracle_hits.get((task_id, test_index))
        rows.append(row)
    with (root / "transfer_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row})); writer.writeheader(); writer.writerows(rows)
    sentinel_rows = [row for row in rows if row["sentinel"]]
    with (root / "sentinel6_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in sentinel_rows for key in row})); writer.writeheader(); writer.writerows(sentinel_rows)
    atomic_write_json(root / "REPORT.json", report)
    (root / "REPORT.md").write_text("# Nonblind Development LOO Transfer30\n\n" + json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    hashes = {path.name: sha_file(path) for path in sorted(root.glob("*.json")) if path.name != "SHA256SUMS.json"}
    hashes.update({"transfer_results.csv": sha_file(root / "transfer_results.csv"), "sentinel6_results.csv": sha_file(root / "sentinel6_results.csv")})
    atomic_write_json(root / "SHA256SUMS.json", hashes)
    print(canonical({"event": "NONBLIND_DEVELOPMENT_LOO_TRANSFER30_SCORED", "decision": decision, "recommendation": recommendation, "solutions_opened_after_gpu_freeze": True}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--solutions", type=Path, required=True)
    main(parser.parse_args())
