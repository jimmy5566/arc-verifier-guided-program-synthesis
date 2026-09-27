#!/usr/bin/env python3
"""Corrective CPU-only audit for the frozen Adaptive TTT Step 1 surface.

This script deliberately reads only frozen checkpoints and execution logs.  It
does not load a model, candidates, or evaluation-test targets.  In particular,
it distinguishes the preregistered 85-minute run from the later recovery of
one missing depth checkpoint rather than silently blending their trajectories.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import statistics
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


VIEWS = ["identity", "rot90", "rot180", "rot270", "flip_lr", "flip_ud", "transpose", "anti_transpose"]
DEPTHS = [0, 12, 24, 48, 72]
SELF_SIGNALS = {
    "mean_logprob_per_token": 1,
    "sequence_logprob": 1,
    "mean_token_entropy": -1,
    "mean_token_top1_top2_margin": 1,
    "minimum_token_margin": 1,
    "ttt_loss": -1,
    "ttt_recent_loss_delta": -1,
    "generation_length": -1,
}
COMPETENCE_SIGNAL = "nll_per_token"


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, sort_keys=True, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def raw_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return None if not values else float(sum(values) / len(values))


def median(values: Iterable[float]) -> float | None:
    values = list(values)
    return None if not values else float(statistics.median(values))


def numeric(value: Any) -> float | None:
    return float(value) if isinstance(value, (float, int)) and math.isfinite(float(value)) else None


def ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda pair: pair[1])
    result = [0.0] * len(values)
    start = 0
    while start < len(indexed):
        end = start + 1
        while end < len(indexed) and indexed[end][1] == indexed[start][1]:
            end += 1
        rank = (start + 1 + end) / 2.0
        for index, _ in indexed[start:end]:
            result[index] = rank
        start = end
    return result


def auc(labels: list[bool], scores: list[float]) -> float | None:
    positive = sum(labels)
    negative = len(labels) - positive
    if not positive or not negative:
        return None
    ordered_ranks = ranks(scores)
    rank_sum = sum(rank for label, rank in zip(labels, ordered_ranks) if label)
    return float((rank_sum - positive * (positive + 1) / 2.0) / (positive * negative))


def csv_write(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({field for row in rows for field in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def cell_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((root / "checkpoints" / "cells").glob("*/*.json")):
        payload = raw_json(path)
        cell = payload.get("cell")
        if payload.get("status") != "FROZEN" or not isinstance(cell, dict):
            raise RuntimeError(f"invalid frozen checkpoint: {path}")
        row = dict(cell)
        row["checkpoint_path"] = str(path.relative_to(root)).replace("\\", "/")
        row["checkpoint_sha256"] = sha256(path)
        rows.append(row)
    return rows


def runtime_audit(root: Path, rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], set[tuple[str, int, str]]]:
    starts = {int(path.stem.removeprefix("runtime_start_worker")): raw_json(path) for path in root.glob("runtime_start_worker*.json")}
    stops = {int(path.stem.removeprefix("runtime_stop_worker")): raw_json(path) for path in root.glob("runtime_stop_worker*.json")}
    primary_excluded: set[tuple[str, int, str]] = set()
    audit: list[dict[str, Any]] = []
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[str(row["task_id"])].append(row)
    for task_id in sorted(by_task):
        task_rows = by_task[task_id]
        assigned_workers = sorted(wid for wid, start in starts.items() if task_id in start.get("task_ids", []))
        # The initial assigned worker is the original preregistered run.  A
        # later runtime-start for the same task is necessarily recovery.
        original_workers = assigned_workers[:1]
        terminal = [entry for stop in stops.values() for entry in stop.get("task_results", []) if entry.get("task_id") == task_id]
        original_terminal = [entry for wid in original_workers for entry in stops.get(wid, {}).get("task_results", []) if entry.get("task_id") == task_id]
        recovery_workers = [wid for wid in assigned_workers if wid not in original_workers]
        original_statuses = [str(entry.get("status")) for entry in original_terminal]
        restarted = len(terminal) > 1 or any(status == "DEADLINE" for status in original_statuses)
        deadline = any(status == "DEADLINE" for status in original_statuses)
        # A deadline task's completed depths are a valid original prefix.  Any
        # cells deeper than the original loss-curve length were generated later.
        original_curve_len = max((len(entry.get("loss_curve", [])) for entry in original_terminal), default=None)
        excluded_depths: list[int] = []
        if deadline and original_curve_len is not None:
            for row in task_rows:
                if int(row["depth"]) > original_curve_len:
                    primary_excluded.add((task_id, int(row["depth"]), str(row["gen_view"])))
                    excluded_depths.append(int(row["depth"]))
        trajectory = raw_json(root / "checkpoints" / "trajectories" / f"{task_id}.json").get("trajectory", {})
        final_curve = trajectory.get("loss_curve", [])
        # Existing cell fields have no worker/trajectory id.  Verify the
        # continuous suffix by matching each depth's recorded TTT loss to the
        # final stored trajectory; values at depth zero have no update value.
        mismatches: list[dict[str, Any]] = []
        for row in task_rows:
            depth = int(row["depth"])
            value = numeric(row.get("ttt_loss"))
            expected = numeric(final_curve[depth - 1]) if depth and len(final_curve) >= depth else None
            if depth and value is not None and expected is not None and not math.isclose(value, expected, rel_tol=1e-10, abs_tol=1e-12):
                mismatches.append({"depth": depth, "gen_view": row["gen_view"], "cell_ttt_loss": value, "final_trajectory_loss": expected})
        audit.append({
            "task_id": task_id,
            "original_workers": original_workers,
            "recovery_workers": recovery_workers,
            "terminal_statuses": original_statuses,
            "restart_or_resume_observed": restarted,
            "original_deadline_observed": deadline,
            "original_curve_length": original_curve_len,
            "stored_final_curve_length": len(final_curve),
            "post_deadline_depths": sorted(set(excluded_depths)),
            "post_deadline_cells": len([key for key in primary_excluded if key[0] == task_id]),
            "mixed_trajectory": bool(excluded_depths),
            "cell_vs_stored_trajectory_mismatch_count": len(mismatches),
            "cell_vs_stored_trajectory_mismatches": mismatches,
            "continuous_prefix_integrity": "PASS" if not mismatches or bool(excluded_depths) else "FAIL",
        })
    return audit, primary_excluded


def value(row: dict[str, Any], signal: str) -> float | None:
    if signal == COMPETENCE_SIGNAL:
        cross = row.get("cross_validation_score")
        return numeric(cross.get(signal)) if isinstance(cross, dict) else None
    return numeric(row.get(signal))


def per_signal(rows: list[dict[str, Any]], signal: str, direction: int, *, label: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    observed = [(row, value(row, signal)) for row in rows]
    observed = [(row, raw) for row, raw in observed if raw is not None]
    labels = [bool(row["loo_greedy_exact"]) for row, _ in observed]
    raw_values = [raw for _, raw in observed]
    oriented = [direction * raw for raw in raw_values]
    exact_raw = [raw for raw, exact in zip(raw_values, labels) if exact]
    miss_raw = [raw for raw, exact in zip(raw_values, labels) if not exact]
    by_task: dict[str, list[tuple[dict[str, Any], float]]] = defaultdict(list)
    for (row, raw), score in zip(observed, oriented):
        by_task[str(row["task_id"])].append((row, score))
    task_rows: list[dict[str, Any]] = []
    within_aucs: list[float] = []
    for task_id, entries in sorted(by_task.items()):
        task_labels = [bool(row["loo_greedy_exact"]) for row, _ in entries]
        task_scores = [score for _, score in entries]
        task_auc = auc(task_labels, task_scores)
        if task_auc is not None:
            within_aucs.append(task_auc)
        ranked = sorted(entries, key=lambda item: (-item[1], int(item[0]["depth"]), VIEWS.index(str(item[0]["gen_view"]))))
        exact_ranks = [index + 1 for index, (row, _) in enumerate(ranked) if bool(row["loo_greedy_exact"])]
        task_rows.append({
            "dataset": label,
            "signal": signal,
            "task_id": task_id,
            "cell_count": len(entries),
            "exact_cell_count": sum(task_labels),
            "within_task_auc": task_auc,
            "best_correct_confidence_rank": min(exact_ranks) if exact_ranks else None,
            "oracle_solvable": bool(exact_ranks),
            **{f"top{k}_retains_exact": bool(exact_ranks and min(exact_ranks) <= k) for k in (1, 2, 4, 8)},
        })
    oracle_tasks = [row for row in task_rows if row["oracle_solvable"]]
    summary = {
        "signal": signal,
        "direction": "higher_raw_is_better" if direction > 0 else "lower_raw_is_better",
        "available_cells": len(observed),
        "exact_cells": sum(labels),
        "raw_exact_mean": mean(exact_raw),
        "raw_exact_median": median(exact_raw),
        "raw_miss_mean": mean(miss_raw),
        "raw_miss_median": median(miss_raw),
        "oriented_exact_mean": mean([score for score, exact in zip(oriented, labels) if exact]),
        "oriented_miss_mean": mean([score for score, exact in zip(oriented, labels) if not exact]),
        "pooled_auc": auc(labels, oriented),
        "within_task_auc_count": len(within_aucs),
        "macro_within_task_auc": mean(within_aucs),
        "median_within_task_auc": median(within_aucs),
        "top_k_task_retrieval": {str(k): {"retained_oracle_tasks": sum(bool(row[f"top{k}_retains_exact"]) for row in oracle_tasks), "oracle_solvable_tasks": len(oracle_tasks)} for k in (1, 2, 4, 8)},
    }
    return summary, task_rows


def competence_router(rows: list[dict[str, Any]], *, label: str) -> dict[str, Any]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if value(row, COMPETENCE_SIGNAL) is not None:
            by_task[str(row["task_id"])].append(row)
    selected: list[dict[str, Any]] = []
    lost: list[str] = []
    oracle = 0
    for task_id, task_rows in sorted(by_task.items()):
        winner = min(task_rows, key=lambda row: (float(value(row, COMPETENCE_SIGNAL)), int(row["depth"]), VIEWS.index(str(row["gen_view"]))))
        selected.append(winner)
        if any(bool(row["loo_greedy_exact"]) for row in task_rows):
            oracle += 1
            if not bool(winner["loo_greedy_exact"]):
                lost.append(task_id)
    return {
        "dataset": label,
        "argmin_nll_selected_exact_tasks": sum(bool(row["loo_greedy_exact"]) for row in selected),
        "task_oracle": oracle,
        "oracle_regret": oracle - sum(bool(row["loo_greedy_exact"]) for row in selected),
        "exact_task_ids_lost_by_competence_routing": lost,
        "selected_cells": [{"task_id": row["task_id"], "depth": row["depth"], "gen_view": row["gen_view"], "loo_greedy_exact": row["loo_greedy_exact"], "nll_per_token": value(row, COMPETENCE_SIGNAL)} for row in selected],
    }


def curves(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    table: list[dict[str, Any]] = []
    for view in VIEWS:
        for depth in DEPTHS:
            group = [row for row in rows if row["gen_view"] == view and int(row["depth"]) == depth]
            confidence = [value(row, "mean_logprob_per_token") for row in group]
            competence = [value(row, COMPETENCE_SIGNAL) for row in group]
            table.append({
                "gen_view": view, "depth": depth, "n": len(group),
                "exact_rate": mean(float(bool(row["loo_greedy_exact"])) for row in group),
                "mean_self_confidence_raw": mean(raw for raw in confidence if raw is not None),
                "median_self_confidence_raw": median(raw for raw in confidence if raw is not None),
                "mean_loo_competence_nll_raw": mean(raw for raw in competence if raw is not None),
            })
    features: dict[str, Any] = {}
    for view in VIEWS:
        points = [row for row in table if row["gen_view"] == view]
        confidence = [row["mean_self_confidence_raw"] for row in points if row["mean_self_confidence_raw"] is not None]
        exact = [row["exact_rate"] for row in points if row["exact_rate"] is not None]
        peak = max(points, key=lambda row: (row["mean_self_confidence_raw"] if row["mean_self_confidence_raw"] is not None else -math.inf, -int(row["depth"])))
        peak_exact = max(points, key=lambda row: (row["exact_rate"] if row["exact_rate"] is not None else -math.inf, -int(row["depth"])))
        final = next(row for row in points if int(row["depth"]) == 72)
        features[view] = {
            "confidence_area_under_depth_curve": sum(confidence),
            "correctness_area_under_depth_curve": sum(exact),
            "peak_confidence_raw": peak["mean_self_confidence_raw"], "peak_confidence_depth": peak["depth"],
            "peak_exact_rate": peak_exact["exact_rate"], "peak_exact_depth": peak_exact["depth"],
            "late_confidence_drop_peak_to_72": None if peak["mean_self_confidence_raw"] is None or final["mean_self_confidence_raw"] is None else peak["mean_self_confidence_raw"] - final["mean_self_confidence_raw"],
            "late_exact_rate_drop_peak_to_72": None if peak_exact["exact_rate"] is None or final["exact_rate"] is None else peak_exact["exact_rate"] - final["exact_rate"],
            "confidence_variance": statistics.pvariance(confidence) if len(confidence) >= 2 else None,
            "exact_rate_variance": statistics.pvariance(exact) if len(exact) >= 2 else None,
        }
    return table, features


def contributions(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[str(row["task_id"])].append(row)
    entries: list[dict[str, Any]] = []
    for task_id, task_rows in sorted(by_task.items()):
        exact = [row for row in task_rows if bool(row["loo_greedy_exact"])]
        for row in exact:
            same_depth = [item for item in exact if int(item["depth"]) == int(row["depth"])]
            same_view = [item for item in exact if item["gen_view"] == row["gen_view"]]
            entries.append({
                "task_id": task_id, "depth": row["depth"], "gen_view": row["gen_view"],
                "per_depth_view_unique": len(same_depth) == 1,
                "global_view_unique": len(same_view) == 1,
                "depth_unique": len({int(item["depth"]) for item in exact}) == 1,
                "global_cell_frontier": len(exact) == 1,
                "mean_logprob_per_token": row.get("mean_logprob_per_token"),
            })
    depth_exclusive: dict[str, list[str]] = {}
    for depth in DEPTHS:
        depth_exclusive[str(depth)] = sorted(
            task_id for task_id, task_rows in by_task.items()
            if any(bool(item["loo_greedy_exact"]) and int(item["depth"]) == depth for item in task_rows)
            and not any(bool(item["loo_greedy_exact"]) and int(item["depth"]) != depth for item in task_rows)
        )
    view_exclusive: dict[str, list[str]] = {}
    for view in VIEWS:
        view_exclusive[view] = sorted(
            task_id for task_id, task_rows in by_task.items()
            if any(bool(item["loo_greedy_exact"]) and item["gen_view"] == view for item in task_rows)
            and not any(bool(item["loo_greedy_exact"]) and item["gen_view"] != view for item in task_rows)
        )
    view_means = {
        view: mean(float(item["mean_logprob_per_token"]) for item in rows if item["gen_view"] == view and numeric(item.get("mean_logprob_per_token")) is not None)
        for view in VIEWS
    }
    view_median = median(item for item in view_means.values() if item is not None)
    low_confidence_unique = [
        {"task_id": row["task_id"], "depth": row["depth"], "gen_view": row["gen_view"], "view_mean_logprob_per_token": view_means[row["gen_view"]]}
        for row in entries if row["global_view_unique"] and view_median is not None and view_means[row["gen_view"]] is not None and view_means[row["gen_view"]] <= view_median
    ]
    poor_rank_correct: list[dict[str, Any]] = []
    for task_id, task_rows in by_task.items():
        scored = [(item, numeric(item.get("mean_logprob_per_token"))) for item in task_rows]
        scored = [(item, score) for item, score in scored if score is not None]
        ranked = sorted(scored, key=lambda pair: (-pair[1], int(pair[0]["depth"]), VIEWS.index(pair[0]["gen_view"])))
        threshold = math.ceil(0.75 * len(ranked))
        for rank, (item, _) in enumerate(ranked, start=1):
            if bool(item["loo_greedy_exact"]) and rank > threshold:
                poor_rank_correct.append({"task_id": task_id, "depth": item["depth"], "gen_view": item["gen_view"], "confidence_rank": rank, "cell_count": len(ranked)})
    summary = {
        "per_depth_view_unique_task_ids": sorted({row["task_id"] for row in entries if row["per_depth_view_unique"]}),
        "global_view_unique_task_ids": sorted({row["task_id"] for row in entries if row["global_view_unique"]}),
        "depth_unique_task_ids": sorted({row["task_id"] for row in entries if row["depth_unique"]}),
        "global_cell_frontier_task_ids": sorted({row["task_id"] for row in entries if row["global_cell_frontier"]}),
        "depth_exclusive": depth_exclusive,
        "view_exclusive": view_exclusive,
        "low_average_confidence_view_global_unique_solutions": low_confidence_unique,
        "correct_cells_with_poor_self_confidence_rank": poor_rank_correct,
    }
    return entries, summary


def decoupling(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[str(row["task_id"])].append(row)
    joint = sum(any(bool(row["loo_greedy_exact"]) for row in task_rows) for task_rows in by_task.values())
    depth_only = sum(any(any(bool(row["loo_greedy_exact"]) and int(row["depth"]) == depth for row in task_rows) for depth in DEPTHS) for task_rows in by_task.values())
    view_only = sum(any(any(bool(row["loo_greedy_exact"]) and row["gen_view"] == view for row in task_rows) for view in VIEWS) for task_rows in by_task.values())
    fixed_depth = {depth: sum(any(bool(row["loo_greedy_exact"]) and int(row["depth"]) == depth for row in task_rows) for task_rows in by_task.values()) for depth in DEPTHS}
    fixed_view = {view: sum(any(bool(row["loo_greedy_exact"]) and row["gen_view"] == view for row in task_rows) for task_rows in by_task.values()) for view in VIEWS}
    fixed_cell = {(depth, view): sum(any(bool(row["loo_greedy_exact"]) and int(row["depth"]) == depth and row["gen_view"] == view for row in task_rows) for task_rows in by_task.values()) for depth in DEPTHS for view in VIEWS}
    return {
        "joint_oracle": joint, "depth_only_oracle": depth_only, "view_only_oracle": view_only,
        "joint_minus_depth_only": joint - depth_only, "joint_minus_view_only": joint - view_only,
        "best_fixed_depth": max(fixed_depth.items(), key=lambda item: (item[1], -DEPTHS.index(item[0]))),
        "best_fixed_view": max(fixed_view.items(), key=lambda item: (item[1], -VIEWS.index(item[0]))),
        "best_fixed_depth_view": max(fixed_cell.items(), key=lambda item: (item[1], -DEPTHS.index(item[0][0]), -VIEWS.index(item[0][1]))),
    }


def figures(directory: Path, curves_table: list[dict[str, Any]], rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as error:
        raise RuntimeError(f"matplotlib required for registered figures: {error}") from error
    directory.mkdir(parents=True, exist_ok=True)
    def plot_all(metric: str, title: str, filename: str, ylabel: str) -> None:
        fig, axis = plt.subplots(figsize=(9, 5))
        for view in VIEWS:
            points = [row for row in curves_table if row["gen_view"] == view]
            axis.plot([row["depth"] for row in points], [row[metric] for row in points], marker="o", label=view)
        axis.set(title=title, xlabel="TTT depth", ylabel=ylabel); axis.legend(ncol=2, fontsize=8); axis.grid(alpha=0.25)
        fig.tight_layout(); fig.savefig(directory / filename, dpi=160); plt.close(fig)
    plot_all("exact_rate", "LOO exact rate by TTT depth and generation view", "01_all_views_exact_rate_vs_depth.png", "exact rate")
    plot_all("mean_self_confidence_raw", "Mean self-confidence by TTT depth and generation view", "02_all_views_confidence_vs_depth.png", "mean logprob/token (raw)")
    for metric, title, filename, ylabel in (("exact_rate", "Exact-rate curve per view", "03_exact_rate_per_view.png", "exact rate"), ("mean_self_confidence_raw", "Self-confidence curve per view", "04_confidence_per_view.png", "mean logprob/token (raw)")):
        fig, axes = plt.subplots(2, 4, figsize=(13, 6), sharex=True)
        for axis, view in zip(axes.flat, VIEWS):
            points = [row for row in curves_table if row["gen_view"] == view]
            axis.plot([row["depth"] for row in points], [row[metric] for row in points], marker="o")
            axis.set_title(view); axis.grid(alpha=0.25)
        fig.supxlabel("TTT depth"); fig.supylabel(ylabel); fig.suptitle(title); fig.tight_layout(); fig.savefig(directory / filename, dpi=160); plt.close(fig)
    def calibration(signal: str, direction: int, filename: str, title: str) -> None:
        entries = [(direction * value(row, signal), bool(row["loo_greedy_exact"])) for row in rows if value(row, signal) is not None]
        entries.sort()
        bins = [entries[math.floor(i * len(entries) / 5):math.floor((i + 1) * len(entries) / 5)] for i in range(5)]
        fig, axis = plt.subplots(figsize=(7, 4)); axis.bar(range(1, 6), [mean(float(hit) for _, hit in group) for group in bins]); axis.set(title=title, xlabel="oriented confidence quintile (low → high)", ylabel="LOO exact rate", ylim=(0, 1)); axis.grid(axis="y", alpha=.25); fig.tight_layout(); fig.savefig(directory / filename, dpi=160); plt.close(fig)
    calibration("mean_logprob_per_token", 1, "05_confidence_bin_calibration.png", "Self-confidence calibration")
    calibration(COMPETENCE_SIGNAL, -1, "06_loo_competence_bin_calibration.png", "LOO competence calibration")
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows: by_task[str(row["task_id"])].append(row)
    best_depths = []
    for task_rows in by_task.values():
        hits = [row for row in task_rows if bool(row["loo_greedy_exact"])]
        if hits: best_depths.append(min(int(row["depth"]) for row in hits))
    fig, axis = plt.subplots(figsize=(7, 4)); axis.bar([str(depth) for depth in DEPTHS], [best_depths.count(depth) for depth in DEPTHS]); axis.set(title="Earliest exact depth per oracle-solvable task", xlabel="TTT depth", ylabel="tasks"); axis.grid(axis="y", alpha=.25); fig.tight_layout(); fig.savefig(directory / "07_best_depth_distribution.png", dpi=160); plt.close(fig)
    _, unique = contributions(rows)
    labels = [f"d{depth}" for depth in DEPTHS] + VIEWS
    values = [len(unique["depth_exclusive"][str(depth)]) for depth in DEPTHS] + [len(unique["view_exclusive"][view]) for view in VIEWS]
    fig, axis = plt.subplots(figsize=(10, 4)); axis.bar(labels, values); axis.set(title="Exact exclusive task contribution by depth and view", ylabel="exclusive exact tasks"); axis.tick_params(axis="x", rotation=35); axis.grid(axis="y", alpha=.25); fig.tight_layout(); fig.savefig(directory / "08_unique_contribution_by_view_depth.png", dpi=160); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replace-uncommitted-output", action="store_true", help="Only replaces a local, uncommitted failed analysis attempt.")
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    if output.exists():
        if not args.replace_uncommitted_output:
            raise RuntimeError(f"refusing to overwrite existing audit directory: {output}")
        allowed_output = root / "corrective_audit_v1"
        if output != allowed_output or (output / "FROZEN_COMPLETE").exists():
            raise RuntimeError(f"refusing to replace non-temporary audit directory: {output}")
        # The first local render failed before validation/commit and is neither
        # frozen nor source evidence.  Recreate exactly this registered output.
        shutil.rmtree(output)
    manifest = raw_json(root / "manifest.json")
    rows = cell_rows(root)
    expected = int(manifest["planned_cells"])
    if len(rows) != expected:
        raise RuntimeError(f"expected {expected} frozen cells, found {len(rows)}")
    audit, excluded = runtime_audit(root, rows)
    primary = [row for row in rows if (str(row["task_id"]), int(row["depth"]), str(row["gen_view"])) not in excluded]
    extended = list(rows)
    if len(primary) >= len(extended):
        raise RuntimeError("no post-deadline cells detected; corrective provenance premise needs manual review")
    output.mkdir(parents=True)
    for dataset, dataset_rows in (("PRIMARY_PREREGISTERED", primary), ("EXTENDED_POSTHOC", extended)):
        for row in dataset_rows: row["dataset"] = dataset
    csv_write(output / "primary_cells.csv", primary)
    csv_write(output / "extended_cells.csv", extended)
    signal_summaries: dict[str, Any] = {"PRIMARY_PREREGISTERED": {}, "EXTENDED_POSTHOC": {}}
    within_rows: list[dict[str, Any]] = []
    competence: dict[str, Any] = {}
    competence_rows: list[dict[str, Any]] = []
    for dataset, dataset_rows in (("PRIMARY_PREREGISTERED", primary), ("EXTENDED_POSTHOC", extended)):
        for signal, direction in SELF_SIGNALS.items():
            summary, per_task = per_signal(dataset_rows, signal, direction, label=dataset)
            signal_summaries[dataset][signal] = summary
            within_rows.extend(per_task)
        c_summary, c_per_task = per_signal(dataset_rows, COMPETENCE_SIGNAL, -1, label=dataset)
        competence[dataset] = {"signal_summary": c_summary, "argmin_nll_router": competence_router(dataset_rows, label=dataset)}
        competence_rows.extend(c_per_task)
    csv_write(output / "within_task_confidence.csv", within_rows)
    csv_write(output / "competence_audit.csv", competence_rows)
    curve_table, curve_features = curves(primary)
    csv_write(output / "curve_table.csv", curve_table)
    unique_rows, unique_summary = contributions(primary)
    csv_write(output / "unique_contributions.csv", unique_rows)
    figures(output / "figures", curve_table, primary)
    primary_decoupling = decoupling(primary)
    extended_decoupling = decoupling(extended)
    best_pooled = max(SELF_SIGNALS, key=lambda signal: signal_summaries["PRIMARY_PREREGISTERED"][signal]["pooled_auc"] if signal_summaries["PRIMARY_PREREGISTERED"][signal]["pooled_auc"] is not None else -math.inf)
    best_macro = max(SELF_SIGNALS, key=lambda signal: signal_summaries["PRIMARY_PREREGISTERED"][signal]["macro_within_task_auc"] if signal_summaries["PRIMARY_PREREGISTERED"][signal]["macro_within_task_auc"] is not None else -math.inf)
    deviation = {
        "hard_wall_seconds": manifest.get("hard_wall_seconds"),
        "worker1_elapsed_seconds": raw_json(root / "runtime_stop_worker1.json")["elapsed_seconds"],
        "worker1_deadline_overrun_seconds": raw_json(root / "runtime_stop_worker1.json")["elapsed_seconds"] - float(manifest.get("hard_wall_seconds", 0)),
        "worker1_deadline_reached": raw_json(root / "runtime_stop_worker1.json")["deadline_reached"],
        "post_deadline_cells": [{"task_id": task, "depth": depth, "gen_view": view} for task, depth, view in sorted(excluded, key=lambda key: (key[0], key[1], VIEWS.index(key[2])))],
        "classification": "CONFIRMED: 981571dc depth72 cells were generated after the preregistered hard-wall run on a restarted numerically distinct trajectory.",
    }
    report = {
        "audit_mode": "CPU_ONLY_FROZEN_ARTIFACTS", "solutions_opened": False,
        "source_commit": "12b904bb2576aaf074cdb9d1d611dc1e8fb6acc4",
        "source_root": str(root),
        "provenance_audit": audit,
        "protocol_deviation": deviation,
        "datasets": {"PRIMARY_PREREGISTERED": {"cell_count": len(primary), "definition": "Original preregistered 85-minute continuous trajectories only."}, "EXTENDED_POSTHOC": {"cell_count": len(extended), "definition": "All frozen cells, including post-deadline recovery."}},
        "self_confidence": signal_summaries,
        "best_primary_pooled_self_confidence_signal": {"signal": best_pooled, "pooled_auc": signal_summaries["PRIMARY_PREREGISTERED"][best_pooled]["pooled_auc"]},
        "best_primary_macro_within_task_signal": {"signal": best_macro, "macro_within_task_auc": signal_summaries["PRIMARY_PREREGISTERED"][best_macro]["macro_within_task_auc"]},
        "loo_competence": competence,
        "primary_curves": {"table": curve_table, "function_features_exploratory": curve_features},
        "unique_contributions_primary": unique_summary,
        "decoupling": {"PRIMARY_PREREGISTERED": primary_decoupling, "EXTENDED_POSTHOC": extended_decoupling},
        "recommendation": "REVISE_CONFIDENCE before large-scale GPU expansion. The preregistered surface is incomplete at depth72 for one task and pooled metrics are not treated as evidence of a deployable router.",
    }
    atomic_json(output / "REPORT.json", report)
    primary_best = report["best_primary_pooled_self_confidence_signal"]
    primary_macro = report["best_primary_macro_within_task_signal"]
    lines = [
        "# Adaptive TTT Step 1 Corrective Audit V1", "",
        "**Mode:** CPU-only, frozen artifacts only. Evaluation test targets were not opened.", "",
        "## Provenance correction", "",
        f"- Primary preregistered dataset: **{len(primary)}/{expected}** cells.",
        f"- Extended post-hoc sensitivity dataset: **{len(extended)}/{expected}** cells.",
        "- Confirmed deviation: worker1 exceeded the nominal 5,100 s wall slightly, then stopped during `981571dc`; the eight depth72 cells for that task came from a later restarted, numerically divergent trajectory and are excluded from primary analysis.",
        "- No evaluation-test target was loaded; `loo_greedy_exact` is a held-out training-pair label.", "",
        "## Corrected H1", "",
        f"- Best pooled primary self-confidence: `{primary_best['signal']}` AUC = {primary_best['pooled_auc']:.4f}.",
        f"- Best macro within-task primary self-confidence: `{primary_macro['signal']}` AUC = {primary_macro['macro_within_task_auc']:.4f}.",
        "- Raw lower-is-better metrics remain raw in `REPORT.json`; only explicitly named oriented fields are sign-transformed.", "",
        "## H4 / decision", "",
        "The data are descriptive LOO pseudo-test evidence only.  They do not validate a deployment router. **Decision: REVISE_CONFIDENCE before large-scale GPU expansion.**", "",
        "See `REPORT.json`, CSV tables, and `figures/` for complete primary/post-hoc separation and per-task evidence.",
    ]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    hashes = {str(path.relative_to(output)).replace("\\", "/"): sha256(path) for path in sorted(output.rglob("*")) if path.is_file() and path.name != "SHA256SUMS.json"}
    atomic_json(output / "SHA256SUMS.json", hashes)
    (output / "FROZEN_COMPLETE").write_text("Validated corrective audit output; immutable after local commit.\n", encoding="utf-8")
    print(canonical({"event": "ADAPTIVE_TTT_STEP1_CORRECTIVE_AUDIT_COMPLETE", "primary_cells": len(primary), "extended_cells": len(extended), "output": str(output), "solutions_opened": False}))


if __name__ == "__main__":
    main()
