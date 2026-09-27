#!/usr/bin/env python3
"""CPU-only analysis for the frozen Adaptive TTT Step 1 LOO surface.

This intentionally consumes only atomically frozen cell checkpoints.  It does
not load a model, candidate generator, or any ARC evaluation test solution.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


VIEW_ORDER = ["identity", "rot90", "rot180", "rot270", "flip_lr", "flip_ud", "transpose", "anti_transpose"]
DEPTH_ORDER = [0, 12, 24, 48, 72]


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, sort_keys=True, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return None if not values else float(sum(values) / len(values))


def median(values: Iterable[float]) -> float | None:
    values = list(values)
    return None if not values else float(statistics.median(values))


def average_ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda row: row[1])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(indexed):
        end = start + 1
        while end < len(indexed) and indexed[end][1] == indexed[start][1]:
            end += 1
        rank = (start + 1 + end) / 2.0
        for index, _ in indexed[start:end]:
            ranks[index] = rank
        start = end
    return ranks


def auc(labels: list[bool], scores: list[float]) -> float | None:
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    ranks = average_ranks(scores)
    rank_sum = sum(rank for label, rank in zip(labels, ranks) if label)
    return float((rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives))


def spearman(labels: list[bool], scores: list[float]) -> float | None:
    if len(labels) < 3 or len(set(labels)) < 2 or len(set(scores)) < 2:
        return None
    rx = average_ranks([float(value) for value in labels])
    ry = average_ranks(scores)
    mx, my = mean(rx), mean(ry)
    assert mx is not None and my is not None
    numerator = sum((x - mx) * (y - my) for x, y in zip(rx, ry))
    dx = math.sqrt(sum((x - mx) ** 2 for x in rx))
    dy = math.sqrt(sum((y - my) ** 2 for y in ry))
    return None if not dx or not dy else float(numerator / (dx * dy))


def quartile_rates(labels: list[bool], oriented: list[float]) -> dict[str, Any] | None:
    if len(labels) < 4:
        return None
    ordered = sorted(zip(oriented, labels), key=lambda item: item[0])
    result: dict[str, Any] = {}
    for number, (left, right) in enumerate(((0, 1), (1, 2), (2, 3), (3, 4)), start=1):
        subset = ordered[math.floor(len(ordered) * left / 4):math.floor(len(ordered) * right / 4)]
        result[f"q{number}"] = {"n": len(subset), "exact_rate": mean(float(label) for _, label in subset)}
    return result


def numeric(cell: dict[str, Any], field: str) -> float | None:
    value = cell.get(field)
    return float(value) if isinstance(value, (int, float)) and math.isfinite(float(value)) else None


def select(rows: list[dict[str, Any]], field: str, sign: int) -> dict[str, Any] | None:
    eligible = [row for row in rows if numeric(row, field) is not None]
    if not eligible:
        return None
    # Tie ordering is deterministic and explicitly analytic-only; it does not
    # modify generation or any production selector.
    return sorted(
        eligible,
        key=lambda row: (-sign * float(row[field]), int(row["depth"]), VIEW_ORDER.index(row["gen_view"])),
    )[0]


def load_cells(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    valid: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    identities: set[str] = set()
    for path in sorted((root / "checkpoints" / "cells").glob("*/*.json")):
        try:
            payload = read_json(path)
        except Exception as error:  # report instead of guessing
            rejected.append({"path": str(path), "reason": f"json:{type(error).__name__}"})
            continue
        cell = payload.get("cell")
        if payload.get("status") != "FROZEN" or not isinstance(payload.get("identity"), str) or not isinstance(cell, dict):
            rejected.append({"path": str(path), "reason": "not_frozen_cell"})
            continue
        required = {"task_id", "depth", "gen_view", "loo_greedy_exact", "parse_valid"}
        if not required.issubset(cell) or cell["gen_view"] not in VIEW_ORDER or cell["depth"] not in DEPTH_ORDER:
            rejected.append({"path": str(path), "reason": "invalid_cell_schema"})
            continue
        identities.add(payload["identity"])
        value = dict(cell)
        value["checkpoint_path"] = str(path)
        value["checkpoint_sha256"] = sha256_file(path)
        valid.append(value)
    if len(identities) > 1:
        raise RuntimeError(f"mixed cell identities: {sorted(identities)}")
    return valid, rejected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True, help="Frozen Step 1 run root")
    parser.add_argument("--analysis-dir", type=Path, help="Defaults to ROOT/analysis")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    analysis = (args.analysis_dir or root / "analysis").resolve()
    manifest = read_json(root / "manifest.json")
    cohort = read_json(root / "cohort.json")
    config = read_json(root / "config_resolved.json")
    cells, rejected = load_cells(root)
    expected_keys = {(entry["task_id"], depth, view) for entry in cohort["entries"] for depth in manifest["depths"] for view in manifest["gen_views"]}
    by_key = {(row["task_id"], int(row["depth"]), row["gen_view"]): row for row in cells}
    unexpected = sorted(set(by_key) - expected_keys)
    duplicates = len(cells) - len(by_key)
    if unexpected or duplicates:
        raise RuntimeError(f"checkpoint identity collision: unexpected={unexpected[:3]} duplicates={duplicates}")
    missing = sorted(expected_keys - set(by_key))
    complete = not missing and not rejected
    if args.require_complete and not complete:
        raise RuntimeError(f"surface incomplete: frozen={len(cells)} expected={len(expected_keys)} rejected={len(rejected)}")

    rows = sorted(cells, key=lambda row: (row["task_id"], int(row["depth"]), VIEW_ORDER.index(row["gen_view"])))
    analysis.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with (analysis / "surface.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    with (analysis / "cells_frozen.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical(row) + "\n")

    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[row["task_id"]].append(row)
    signal_specs = {
        "mean_logprob_per_token": 1,
        "sequence_logprob": 1,
        "mean_token_entropy": -1,
        "mean_token_top1_top2_margin": 1,
        "minimum_token_margin": 1,
        "ttt_loss": -1,
        "ttt_recent_loss_delta": -1,
        "generation_length": -1,
    }
    signal_summary: dict[str, Any] = {}
    router_summary: dict[str, Any] = {}
    exact_labels = [bool(row["loo_greedy_exact"]) for row in rows]
    for field, sign in signal_specs.items():
        observed = [(bool(row["loo_greedy_exact"]), sign * value) for row in rows if (value := numeric(row, field)) is not None]
        labels, scores = [item[0] for item in observed], [item[1] for item in observed]
        positives = [score for label, score in observed if label]
        negatives = [score for label, score in observed if not label]
        signal_summary[field] = {
            "orientation": "higher_is_better" if sign > 0 else "lower_is_better",
            "available_cells": len(observed), "exact_cells": sum(labels),
            "exact_mean": mean(positives), "exact_median": median(positives),
            "miss_mean": mean(negatives), "miss_median": median(negatives),
            "auc_exact": auc(labels, scores), "spearman_exact": spearman(labels, scores),
            "quartile_exact_rates": quartile_rates(labels, scores),
        }
        chosen = [choice for task_rows in by_task.values() if (choice := select(task_rows, field, sign)) is not None]
        router_summary[field] = {
            "eligible_tasks": len(chosen), "confidence_selected_exact_tasks": sum(bool(row["loo_greedy_exact"]) for row in chosen),
            "confidence_selected_exact_rate": mean(float(row["loo_greedy_exact"]) for row in chosen),
            "selected_cells": [{"task_id": row["task_id"], "depth": row["depth"], "gen_view": row["gen_view"], "loo_greedy_exact": row["loo_greedy_exact"]} for row in chosen],
        }

    task_oracle = {task: any(bool(row["loo_greedy_exact"]) for row in task_rows) for task, task_rows in by_task.items()}
    random_expectation = mean(mean(float(row["loo_greedy_exact"]) for row in task_rows) or 0.0 for task_rows in by_task.values())
    exact_by_depth: dict[int, Any] = {}
    exact_by_view: dict[str, Any] = {}
    for depth in DEPTH_ORDER:
        rows_at = [row for row in rows if int(row["depth"]) == depth]
        per_task_any = {task: any(bool(row["loo_greedy_exact"]) for row in task_rows if int(row["depth"]) == depth) for task, task_rows in by_task.items()}
        exclusive = sum(value and not any(bool(row["loo_greedy_exact"]) for row in task_rows if int(row["depth"]) != depth) for task, (value, task_rows) in ((task, (per_task_any[task], by_task[task])) for task in by_task))
        exact_by_depth[depth] = {"cells": len(rows_at), "exact_cells": sum(bool(row["loo_greedy_exact"]) for row in rows_at), "task_oracle": sum(per_task_any.values()), "exclusive_task_contribution": exclusive}
    for view in VIEW_ORDER:
        rows_at = [row for row in rows if row["gen_view"] == view]
        per_task_any = {task: any(bool(row["loo_greedy_exact"]) for row in task_rows if row["gen_view"] == view) for task, task_rows in by_task.items()}
        exclusive = sum(value and not any(bool(row["loo_greedy_exact"]) for row in task_rows if row["gen_view"] != view) for task, (value, task_rows) in ((task, (per_task_any[task], by_task[task])) for task in by_task))
        exact_by_view[view] = {"cells": len(rows_at), "exact_cells": sum(bool(row["loo_greedy_exact"]) for row in rows_at), "task_oracle": sum(per_task_any.values()), "exclusive_task_contribution": exclusive}
    interaction = [{"depth": depth, "gen_view": view, "cells": len(group := [row for row in rows if int(row["depth"]) == depth and row["gen_view"] == view]), "exact_cells": sum(bool(row["loo_greedy_exact"]) for row in group), "parse_valid_cells": sum(bool(row["parse_valid"]) for row in group)} for depth in DEPTH_ORDER for view in VIEW_ORDER]
    best_fixed = sorted(interaction, key=lambda row: (-row["exact_cells"], DEPTH_ORDER.index(row["depth"]), VIEW_ORDER.index(row["gen_view"])))[0] if interaction else None
    cv_rows = [row for row in rows if isinstance(row.get("cross_validation_score"), dict)]
    cv_summary = {
        "available_cells": len(cv_rows),
        "nll_per_token": {"exact_mean": mean(float(row["cross_validation_score"]["nll_per_token"]) for row in cv_rows if row["loo_greedy_exact"] and numeric(row["cross_validation_score"], "nll_per_token") is not None), "miss_mean": mean(float(row["cross_validation_score"]["nll_per_token"]) for row in cv_rows if not row["loo_greedy_exact"] and numeric(row["cross_validation_score"], "nll_per_token") is not None)},
        "note": "Teacher-forced held-pair metric; excluded from primary self-confidence routing.",
    }
    report = {
        "experiment_id": manifest.get("experiment_id"), "analysis_mode": "CPU_ONLY_FROZEN_CHECKPOINTS", "solutions_opened": False,
        "source": {"manifest_sha256": sha256_file(root / "manifest.json"), "cohort_sha256": sha256_file(root / "cohort.json"), "config_sha256": sha256_file(root / "config_resolved.json"), "cell_identity": read_json(next((root / "checkpoints" / "cells").glob("*/*.json"))["identity"] if rows else None},
        "completion": {"expected_cells": len(expected_keys), "frozen_cells": len(rows), "missing_cells": len(missing), "missing_cell_keys": [{"task_id": task, "depth": depth, "gen_view": view} for task, depth, view in missing], "rejected_checkpoints": rejected, "complete": complete},
        "outcomes": {"exact_cells": sum(exact_labels), "parse_valid_cells": sum(bool(row["parse_valid"]) for row in rows), "task_count_with_cells": len(by_task), "task_oracle_exact": sum(task_oracle.values()), "random_cell_expected_exact_tasks": random_expectation * len(by_task), "best_fixed_depth_view": best_fixed},
        "primary_self_confidence": signal_summary, "analytic_confidence_router": router_summary,
        "depth_contributions": exact_by_depth, "view_contributions": exact_by_view, "depth_view_interaction": interaction,
        "cross_validation_score_separate": cv_summary,
        "interpretation": "Descriptive LOO pseudo-test evidence only. No real Eval60 test targets were opened and no production selector is changed.",
    }
    write_json(analysis / "REPORT.json", report)
    lines = ["# Adaptive TTT Step 1 LOO Confidence Surface", "", "**Mode:** CPU-only analysis of frozen LOO checkpoints. Real Eval60 test targets were not opened.", "", f"- Frozen cells: **{len(rows)}/{len(expected_keys)}**", f"- Exact cells: **{report['outcomes']['exact_cells']}**", f"- Parse-valid cells: **{report['outcomes']['parse_valid_cells']}**", f"- Per-task cell oracle: **{report['outcomes']['task_oracle_exact']}/{len(by_task)}**", "", "## Primary self-confidence", "", "| Signal | Available cells | AUC exact | Spearman | Router exact tasks |", "| --- | ---: | ---: | ---: | ---: |"]
    for field in signal_specs:
        stat, router = signal_summary[field], router_summary[field]
        def fmt(value: Any) -> str: return "N/A" if value is None else f"{float(value):.4f}"
        lines.append(f"| {field} | {stat['available_cells']} | {fmt(stat['auc_exact'])} | {fmt(stat['spearman_exact'])} | {router['confidence_selected_exact_tasks']}/{router['eligible_tasks']} |")
    lines += ["", "## Coverage", "", "| Dimension | Slot | Exact cells | Task oracle | Exclusive tasks |", "| --- | --- | ---: | ---: | ---: |"]
    for depth, values in exact_by_depth.items(): lines.append(f"| depth | {depth} | {values['exact_cells']} | {values['task_oracle']} | {values['exclusive_task_contribution']} |")
    for view, values in exact_by_view.items(): lines.append(f"| view | {view} | {values['exact_cells']} | {values['task_oracle']} | {values['exclusive_task_contribution']} |")
    if not complete: lines += ["", "## Status", "", "**PARTIAL:** deadline or interruption left missing cells. No completeness-dependent expansion decision is made."]
    (analysis / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    hashes = {str(path.relative_to(root)): sha256_file(path) for path in sorted(analysis.rglob("*")) if path.is_file()}
    write_json(analysis / "SHA256SUMS.json", hashes)
    print(canonical({"event": "ADAPTIVE_TTT_STEP1_ANALYZED", "frozen_cells": len(rows), "expected_cells": len(expected_keys), "complete": complete, "analysis_dir": str(analysis), "solutions_opened": False}))


if __name__ == "__main__":
    main()
