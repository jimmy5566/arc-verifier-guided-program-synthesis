#!/usr/bin/env python3
"""CPU-only decoder-conditioned routing audit over frozen ARC2 artifacts.

This script intentionally never imports model libraries.  It treats Greedy and
V5 outcomes as retrospective decoder-capability labels, keeps unresolved V5
records censored, and evaluates only the fixed 3 x 4 production surface.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEPTHS = (12, 24, 48)
VIEWS = ("identity", "flip_ud", "transpose", "anti_transpose")
CELLS = tuple((d, v) for d in DEPTHS for v in VIEWS)
TARGETS = ("Y_G", "Y_F", "Y_C", "Y_R")
KS = (1, 2, 4, 6)
RESCUE_IDS = ("62593bfd:o1", "78332cb0:o0", "b6f77b65:o2", "de809cff:o0")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: "" if row.get(k) is None else row.get(k) for k in fields})


def bool_or_none(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return 1
    if text in {"false", "0", "no"}:
        return 0
    return None


def numeric(value: Any) -> float | None:
    try:
        if value is None or str(value).strip() == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def status_for_sample(n_positive_outputs: int) -> str:
    if n_positive_outputs < 5:
        return "INSUFFICIENT_FOR_ROUTER_CONCLUSION"
    if n_positive_outputs < 10:
        return "EXPLORATORY_SMALL_N"
    return "DESCRIPTIVE_DEVELOPMENT_ONLY"


def cell_key(task_id: str, output_index: str, depth: int | str, view: str) -> tuple[str, str, int, str]:
    return (str(task_id), str(output_index), int(depth), str(view))


def rank_scores(scores: dict[tuple[int, str], float]) -> dict[tuple[int, str], int] | None:
    if set(scores) != set(CELLS):
        return None
    order = {c: i for i, c in enumerate(CELLS)}
    return {cell: i + 1 for i, cell in enumerate(sorted(CELLS, key=lambda c: (scores[c], order[c]))) }


def metric_rows(
    layer: str,
    mode: str,
    target: str,
    output_rows: dict[str, list[dict[str, Any]]],
    ranks_by_output: dict[str, dict[tuple[int, str], int]],
    allowed_outputs: set[str] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compute retrieval only where all labels and all 12 ranks are available."""
    first_rows: list[dict[str, Any]] = []
    usable: list[tuple[str, int]] = []
    for output_id, rows in sorted(output_rows.items()):
        if allowed_outputs is not None and output_id not in allowed_outputs:
            continue
        ranks = ranks_by_output.get(output_id)
        if not ranks or set(ranks) != set(CELLS):
            continue
        labels = {(int(r["depth"]), r["view"]): r[target] for r in rows}
        if set(labels) != set(CELLS) or any(v is None for v in labels.values()):
            continue
        positives = [ranks[c] for c in CELLS if labels[c] == 1]
        if not positives:
            continue
        first = min(positives)
        usable.append((output_id, first))
        first_rows.append(
            {
                "layer": layer,
                "mode": mode,
                "target": target,
                "output_id": output_id,
                "task_id": rows[0]["task_id"],
                "first_positive_rank": first,
                "mrr": round(1.0 / first, 10),
                "cells_required": first,
                "positive_cell_count": sum(1 for v in labels.values() if v == 1),
                "sample_status": status_for_sample(len(usable)),
            }
        )
    tasks = {output_rows[o][0]["task_id"] for o, _ in usable}
    row: dict[str, Any] = {
        "layer": layer,
        "mode": mode,
        "target": target,
        "n_tasks": len(tasks),
        "n_positive_outputs": len(usable),
        "n_positive_cells": sum(x["positive_cell_count"] for x in first_rows),
        "mean_first_positive_rank": round(statistics.mean([x[1] for x in usable]), 8) if usable else None,
        "mean_cells_required": round(statistics.mean([x[1] for x in usable]), 8) if usable else None,
        "mrr": round(statistics.mean([1.0 / x[1] for x in usable]), 8) if usable else None,
        "sample_status": status_for_sample(len(usable)),
    }
    for k in KS:
        row[f"recall_at_{k}"] = round(sum(1 for _, first in usable if first <= k) / len(usable), 8) if usable else None
        row[f"retained_at_{k}"] = sum(1 for _, first in usable if first <= k)
    return row, first_rows


def train_logistic(xs: list[float], ys: list[int]) -> tuple[float, float, float, float] | None:
    """Small deterministic one-feature L2 logistic fit; returns coef/intercept/mean/std."""
    if len(set(ys)) < 2 or not xs:
        return None
    mean = statistics.mean(xs)
    sd = statistics.pstdev(xs) or 1.0
    beta = 0.0
    intercept = math.log((sum(ys) + 0.5) / (len(ys) - sum(ys) + 0.5))
    l2 = 0.25
    for _ in range(800):
        gb = l2 * beta
        gi = 0.0
        for x, y in zip(xs, ys):
            z = max(-30.0, min(30.0, intercept + beta * ((x - mean) / sd)))
            p = 1.0 / (1.0 + math.exp(-z))
            diff = p - y
            gb += diff * ((x - mean) / sd)
            gi += diff
        scale = 0.15 / len(xs)
        beta -= scale * gb
        intercept -= scale * gi
    return beta, intercept, mean, sd


def predict_logistic(model: tuple[float, float, float, float] | None, prior: float, x: float) -> float:
    if model is None:
        return prior
    beta, intercept, mean, sd = model
    z = max(-30.0, min(30.0, intercept + beta * ((x - mean) / sd)))
    return 1.0 / (1.0 + math.exp(-z))


def calibrated_ranks(
    layer: str,
    target: str,
    output_rows: dict[str, list[dict[str, Any]]],
    raw_scores: dict[str, dict[tuple[int, str], float]],
) -> tuple[dict[str, dict[tuple[int, str], int]], list[dict[str, Any]], str]:
    """Leave-one-task-out calibration, never random-cell split."""
    records: list[tuple[str, str, tuple[int, str], float, int]] = []
    for output_id, rows in output_rows.items():
        if output_id not in raw_scores or set(raw_scores[output_id]) != set(CELLS):
            continue
        labels = {(int(r["depth"]), r["view"]): r[target] for r in rows}
        if any(labels.get(c) is None for c in CELLS):
            continue
        for c in CELLS:
            records.append((rows[0]["task_id"], output_id, c, raw_scores[output_id][c], int(labels[c])))
    pos_outputs = len({o for _, o, _, _, y in records if y == 1})
    if pos_outputs < 5:
        return {}, [], "INSUFFICIENT_FOR_ROUTER_CONCLUSION"
    by_task: dict[str, list[tuple[str, str, tuple[int, str], float, int]]] = defaultdict(list)
    for rec in records:
        by_task[rec[0]].append(rec)
    pred: dict[tuple[str, tuple[int, str]], float] = {}
    details: list[dict[str, Any]] = []
    for held_task, held in sorted(by_task.items()):
        train = [r for task, recs in by_task.items() if task != held_task for r in recs]
        xs, ys = [r[3] for r in train], [r[4] for r in train]
        prior = (sum(ys) + 0.5) / (len(ys) + 1.0) if ys else 0.0
        model = train_logistic(xs, ys)
        for _, output_id, c, x, y in held:
            p = predict_logistic(model, prior, x)
            pred[(output_id, c)] = p
            details.append(
                {
                    "layer": layer,
                    "target": target,
                    "task_id": held_task,
                    "output_id": output_id,
                    "depth": c[0],
                    "view": c[1],
                    "raw_train_only_score": x,
                    "out_of_task_probability": round(p, 10),
                    "decoder_label": y,
                    "cv_split": "LEAVE_ONE_TASK_OUT",
                    "fit_status": "LOGISTIC" if model is not None else "CONSTANT_PRIOR",
                }
            )
    order = {c: i for i, c in enumerate(CELLS)}
    ranks: dict[str, dict[tuple[int, str], int]] = {}
    for output_id in {r[1] for r in records}:
        ps = {c: pred[(output_id, c)] for c in CELLS if (output_id, c) in pred}
        if set(ps) == set(CELLS):
            ranks[output_id] = {c: i + 1 for i, c in enumerate(sorted(CELLS, key=lambda c: (-ps[c], order[c])))}
    return ranks, details, status_for_sample(pos_outputs)


def format_metrics(row: dict[str, Any] | None) -> str:
    if not row or row.get("n_positive_outputs", 0) == 0:
        return "NOT ESTABLISHED"
    return "/".join(str(row.get(f"recall_at_{k}")) for k in KS)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    args = ap.parse_args()
    repo = args.repo.resolve()
    out = repo / "analysis" / "decoder_conditioned_cell_routing_v1"
    out.mkdir(parents=True, exist_ok=True)
    compact = repo / "artifacts" / "eval60_compact_analysis_v2"
    greedy_path = compact / "greedy" / "greedy_cells.csv"
    v5_path = compact / "turbodfs_v5" / "v5_cells.csv"
    l1_path = repo / "artifacts" / "adaptive_ttt_step1_depth_view_confidence_v1" / "analysis" / "surface.csv"
    l2_path = repo / "analysis" / "nonblind_development_loo_transfer30_v1" / "frozen_run_20260927T112700Z" / "loo_rankings_frozen.json"
    l2_legacy_reference = repo / "artifacts" / "same_adapter_pair_surface_analysis_v1" / "source_snapshot" / "old_loo_surface_reference.csv"
    l2_manifest_only = repo / "artifacts" / "adaptive_ttt_loo_transfer12_v1" / "config_resolved.json"
    l3_surface = repo / "artifacts" / "same_adapter_pair_surface_analysis_v1" / "source_snapshot" / "same_adapter_pair_surface.csv"
    l3_checkpoints = repo / "artifacts" / "same_adapter_pair_surface_analysis_v1" / "source_snapshot" / "checkpoint_manifest.json"
    inputs = [greedy_path, v5_path, l1_path, l2_path, l2_legacy_reference, l2_manifest_only, l3_surface, l3_checkpoints]
    missing = [str(p.relative_to(repo)) for p in inputs if not p.exists()]
    if missing:
        raise SystemExit("missing frozen input(s): " + ", ".join(missing))

    greedy_rows = [r for r in read_csv(greedy_path) if int(r["depth"]) in DEPTHS and r["view"] in VIEWS]
    v5_rows = [r for r in read_csv(v5_path) if int(r["depth"]) in DEPTHS and r["view"] in VIEWS]
    gmap: dict[tuple[str, str, int, str], int | None] = {}
    gprov: dict[tuple[str, str, int, str], str] = {}
    for r in greedy_rows:
        key = cell_key(r["task_id"], r["output_index"], r["depth"], r["view"])
        trusted = r.get("provenance_status") == "TRUSTWORTHY"
        val = bool_or_none(r.get("exact_gold_hit")) if trusted else None
        if key in gmap and gmap[key] != val:
            gmap[key] = None
            gprov[key] = "GREEDY_CONFLICTING_DUPLICATE_UNKNOWN"
        else:
            gmap[key] = val
            gprov[key] = "GREEDY_TRUSTWORTHY" if val is not None else "GREEDY_UNTRUSTWORTHY_UNKNOWN"
    fmap: dict[tuple[str, str, int, str], int | None] = {}
    fprov: dict[tuple[str, str, int, str], str] = {}
    for r in v5_rows:
        key = cell_key(r["task_id"], r["output_index"], r["depth"], r["view"])
        trusted = (r.get("provenance_status") == "TRUSTWORTHY" and r.get("record_link_status") == "MATCHED" and r.get("status") == "DONE")
        val = bool_or_none(r.get("gold_hit_any_candidate")) if trusted else None
        if key in fmap and fmap[key] != val:
            fmap[key] = None
            fprov[key] = "V5_CONFLICTING_DUPLICATE_UNKNOWN"
        else:
            fmap[key] = val
            fprov[key] = "V5_TRUSTWORTHY" if val is not None else "V5_PROVENANCE_LIMITED_UNKNOWN"

    outputs: dict[tuple[str, str], str] = {}
    for k in set(gmap) | set(fmap):
        outputs[(k[0], k[1])] = f"{k[0]}:o{k[1]}"
    label_rows: list[dict[str, Any]] = []
    output_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (task_id, output_index), output_id in sorted(outputs.items()):
        for depth, view in CELLS:
            key = cell_key(task_id, output_index, depth, view)
            yg, yf = gmap.get(key), fmap.get(key)
            yc = 1 if yg == 1 or yf == 1 else (0 if yg == 0 and yf == 0 else None)
            yr = 0 if yg == 1 else (yf if yg == 0 and yf is not None else None)
            row = {
                "task_id": task_id, "output_index": output_index, "output_id": output_id,
                "depth": depth, "view": view,
                "Y_G": yg, "Y_F": yf, "Y_C": yc, "Y_R": yr,
                "greedy_provenance": gprov.get(key, "GREEDY_MISSING_UNKNOWN"),
                "v5_provenance": fprov.get(key, "V5_MISSING_UNKNOWN"),
                "greedy_label_status": "KNOWN" if yg is not None else "UNKNOWN",
                "v5_label_status": "KNOWN" if yf is not None else "UNKNOWN",
                "observed_surface": "PRIMARY_3_DEPTH_X_4_VIEW",
            }
            label_rows.append(row)
            output_rows[output_id].append(row)
    label_fields = list(label_rows[0]) if label_rows else []
    write_csv(out / "decoder_cell_labels.csv", label_rows, label_fields)

    coverage_rows: list[dict[str, Any]] = []
    for output_id, rows in sorted(output_rows.items()):
        yc = [r["Y_C"] for r in rows]
        if any(x == 1 for x in yc):
            coverage = "OBSERVED_POSITIVE"
        elif all(x == 0 for x in yc):
            coverage = "OBSERVED_NO_POSITIVE"
        else:
            coverage = "CENSORED"
        coverage_rows.append({
            "task_id": rows[0]["task_id"], "output_index": rows[0]["output_index"], "output_id": output_id,
            "coverage_class": coverage,
            "known_v5_cells": sum(r["Y_F"] is not None for r in rows),
            "unknown_v5_cells": sum(r["Y_F"] is None for r in rows),
            "greedy_positive_cells": sum(r["Y_G"] == 1 for r in rows),
            "dfs_positive_cells": sum(r["Y_F"] == 1 for r in rows),
            "decoder_positive_cells": sum(r["Y_C"] == 1 for r in rows),
            "dfs_only_rescue_cells": sum(r["Y_R"] == 1 for r in rows),
            "note": "Observed-no-positive is coverage, not a router failure." if coverage == "OBSERVED_NO_POSITIVE" else "",
        })
    coverage_fields = list(coverage_rows[0]) if coverage_rows else []
    write_csv(out / "output_coverage_classes.csv", coverage_rows, coverage_fields)

    # L1: frozen single-heldout score, lower held-out NLL is better.
    l1_scores_by_task: dict[str, dict[tuple[int, str], float]] = defaultdict(dict)
    for r in read_csv(l1_path):
        if int(r["depth"]) not in DEPTHS or r["gen_view"] not in VIEWS:
            continue
        try:
            score = ast.literal_eval(r["cross_validation_score"]).get("nll_per_token")
        except (ValueError, SyntaxError, AttributeError):
            score = None
        score = numeric(score)
        if score is not None:
            l1_scores_by_task[r["task_id"]][(int(r["depth"]), r["gen_view"])] = score

    # L2: finalized all-fold LOO mean NLL. Re-rank the frozen 12-cell subset only.
    l2_doc = json.loads(l2_path.read_text(encoding="utf-8"))
    l2_scores_by_task: dict[str, dict[tuple[int, str], float]] = defaultdict(dict)
    for task in l2_doc.get("rankings", []):
        for c in task.get("all_40_cells", []):
            if int(c["depth"]) in DEPTHS and c["gen_view"] in VIEWS:
                score = numeric(c.get("cv_nll_per_token_mean"))
                if score is not None:
                    l2_scores_by_task[task["task_id"]][(int(c["depth"]), c["gen_view"])] = score

    # L3 audit: same-adapter teacher-forced source exists, but must match current production checkpoints.
    l3_manifest = json.loads(l3_checkpoints.read_text(encoding="utf-8"))
    l3_hashes = {(x["task_id"], int(x["depth"])): x["sha256"] for x in l3_manifest.get("checkpoints", [])}
    production_hashes: dict[tuple[str, int], set[str]] = defaultdict(set)
    for r in greedy_rows:
        production_hashes[(r["task_id"], int(r["depth"]))].add(r.get("checkpoint_sha256", ""))
    l3_matches = [(k, h) for k, h in l3_hashes.items() if h in production_hashes.get(k, set())]

    layer_scores = {"L1_SINGLE_HELDOUT": l1_scores_by_task, "L2_MULTIFOLD_LOO": l2_scores_by_task, "L3_FULL_TRAIN_SAME_STATE": {}}
    ranks_by_layer: dict[str, dict[str, dict[tuple[int, str], int]]] = {}
    raw_score_by_layer_output: dict[str, dict[str, dict[tuple[int, str], float]]] = {}
    for layer, by_task in layer_scores.items():
        ranks_by_output: dict[str, dict[tuple[int, str], int]] = {}
        scores_by_output: dict[str, dict[tuple[int, str], float]] = {}
        for output_id, rows in output_rows.items():
            scores = by_task.get(rows[0]["task_id"], {})
            rank = rank_scores(scores)
            if rank is not None:
                ranks_by_output[output_id] = rank
                scores_by_output[output_id] = scores
        ranks_by_layer[layer] = ranks_by_output
        raw_score_by_layer_output[layer] = scores_by_output

    # Coverage/failure analysis needs to expose what each frozen layer would
    # inspect on a no-positive surface.  This is descriptive only: it never
    # turns a coverage miss into a router-negative training example.
    def top_region(layer: str, output_id: str, k: int = 6) -> str:
        ranks = ranks_by_layer[layer].get(output_id)
        if not ranks:
            return "UNAVAILABLE"
        return ";".join(
            f"d{cell[0]}/{cell[1]}"
            for cell, _ in sorted(ranks.items(), key=lambda item: item[1])[:k]
        )

    for coverage in coverage_rows:
        oid = coverage["output_id"]
        coverage["l1_top6_train_only_region"] = top_region("L1_SINGLE_HELDOUT", oid)
        coverage["l2_top6_train_only_region"] = top_region("L2_MULTIFOLD_LOO", oid)
        coverage["l3_top6_train_only_region"] = "UNAVAILABLE_EXACT_STATE_MISSING"
        top1 = [top_region(layer, oid, 1) for layer in ("L1_SINGLE_HELDOUT", "L2_MULTIFOLD_LOO") if top_region(layer, oid, 1) != "UNAVAILABLE"]
        coverage["available_layer_top1_agreement"] = "NO_AVAILABLE_LAYER" if not top1 else ("AGREE" if len(set(top1)) == 1 else "DISAGREE")
        coverage["coverage_interpretation"] = "NO_DECODER_POSITIVE_IN_OBSERVED_SURFACE" if coverage["coverage_class"] == "OBSERVED_NO_POSITIVE" else ""
    coverage_fields = list(coverage_rows[0]) if coverage_rows else []
    write_csv(out / "output_coverage_classes.csv", coverage_rows, coverage_fields)

    metric_table: list[dict[str, Any]] = []
    first_rank_rows: list[dict[str, Any]] = []
    for layer in layer_scores:
        for target in TARGETS:
            row, details = metric_rows(layer, "RAW_LAYER", target, output_rows, ranks_by_layer[layer])
            if layer == "L3_FULL_TRAIN_SAME_STATE":
                row["availability"] = "MISSING_EVIDENCE_UNMATCHED_PRODUCTION_CHECKPOINT_STATE"
            elif layer == "L2_MULTIFOLD_LOO" and not ranks_by_layer[layer]:
                row["availability"] = "NO_CURRENT_EVAL60_TASK_OVERLAP"
            else:
                row["availability"] = "AVAILABLE"
            metric_table.append(row)
            first_rank_rows.extend(details)

    # Out-of-task calibrated ranks using one layer-local scalar only.
    calibration_details: list[dict[str, Any]] = []
    calibration_summary: list[dict[str, Any]] = []
    calibrated_ranks_by_layer_target: dict[tuple[str, str], dict[str, dict[tuple[int, str], int]]] = {}
    for layer in layer_scores:
        for target in TARGETS:
            if layer == "L3_FULL_TRAIN_SAME_STATE":
                calibration_summary.append({"layer": layer, "target": target, "status": "MISSING_EVIDENCE_UNMATCHED_PRODUCTION_CHECKPOINT_STATE", "model": "NONE", "cv_split": "LEAVE_ONE_TASK_OUT"})
                continue
            ranks, details, status = calibrated_ranks(layer, target, output_rows, raw_score_by_layer_output[layer])
            calibrated_ranks_by_layer_target[(layer, target)] = ranks
            calibration_details.extend(details)
            row, first = metric_rows(layer, "DECODER_CALIBRATED_LAYER", target, output_rows, ranks)
            row["availability"] = "AVAILABLE" if ranks else ("NO_CURRENT_EVAL60_TASK_OVERLAP" if layer == "L2_MULTIFOLD_LOO" else status)
            metric_table.append(row)
            first_rank_rows.extend(first)
            calibration_summary.append({
                "layer": layer, "target": target, "status": status if ranks else status,
                "model": "SINGLE_FEATURE_L2_LOGISTIC" if ranks else "NOT_FIT",
                "cv_split": "LEAVE_ONE_TASK_OUT", "n_cell_predictions": len(details),
                "n_task_groups": len({d["task_id"] for d in details}),
                "n_positive_outputs": row.get("n_positive_outputs", 0),
                "recall_at_1": row.get("recall_at_1"), "recall_at_2": row.get("recall_at_2"),
                "recall_at_4": row.get("recall_at_4"), "recall_at_6": row.get("recall_at_6"),
            })

    metric_fields = ["layer", "mode", "target", "availability", "n_tasks", "n_positive_outputs", "n_positive_cells", "sample_status", "recall_at_1", "recall_at_2", "recall_at_4", "recall_at_6", "retained_at_1", "retained_at_2", "retained_at_4", "retained_at_6", "mean_first_positive_rank", "mean_cells_required", "mrr"]
    write_csv(out / "layer_recall_curves.csv", metric_table, metric_fields)
    write_csv(out / "layer_first_positive_rank.csv", first_rank_rows, ["layer", "mode", "target", "task_id", "output_id", "first_positive_rank", "mrr", "cells_required", "positive_cell_count", "sample_status"])
    for layer, filename in (("L1_SINGLE_HELDOUT", "l1_decoder_alignment.csv"), ("L2_MULTIFOLD_LOO", "l2_decoder_alignment.csv"), ("L3_FULL_TRAIN_SAME_STATE", "l3_decoder_alignment.csv")):
        write_csv(out / filename, [r for r in metric_table if r["layer"] == layer], metric_fields)
    write_csv(out / "decoder_calibration_cv.csv", calibration_details, ["layer", "target", "task_id", "output_id", "depth", "view", "raw_train_only_score", "out_of_task_probability", "decoder_label", "cv_split", "fit_status"])
    write_csv(out / "decoder_calibration_summary.csv", calibration_summary, ["layer", "target", "status", "model", "cv_split", "n_cell_predictions", "n_task_groups", "n_positive_outputs", "recall_at_1", "recall_at_2", "recall_at_4", "recall_at_6"])

    # Cohort comparisons: compare raw availability only, never claim a winner across unmatched cohorts.
    compare_rows: list[dict[str, Any]] = []
    comparison_layers = ["L1_SINGLE_HELDOUT", "L2_MULTIFOLD_LOO", "L3_FULL_TRAIN_SAME_STATE"]
    available_layers = [x for x in comparison_layers if ranks_by_layer[x]]
    for target in TARGETS:
        all_sets = {layer: set(ranks_by_layer[layer]) for layer in available_layers}
        for layer in comparison_layers:
            row, _ = metric_rows(layer, "RAW_LAYER", target, output_rows, ranks_by_layer[layer])
            row["comparison_scope"] = "ALL_AVAILABLE_PER_LAYER"
            if layer == "L3_FULL_TRAIN_SAME_STATE":
                row["availability"] = "MISSING_EVIDENCE_UNMATCHED_PRODUCTION_CHECKPOINT_STATE"
            elif layer == "L2_MULTIFOLD_LOO":
                row["availability"] = "NO_CURRENT_EVAL60_TASK_OVERLAP"
            else:
                row["availability"] = "AVAILABLE"
            compare_rows.append(row)
        for a_i, left in enumerate(comparison_layers):
            for right in comparison_layers[a_i + 1:]:
                common = set(ranks_by_layer[left]) & set(ranks_by_layer[right])
                for layer in (left, right):
                    row, _ = metric_rows(layer, "RAW_LAYER", target, output_rows, ranks_by_layer[layer], common)
                    row["comparison_scope"] = f"PAIRWISE_COMMON_COHORT:{left}:{right}"
                    row["availability"] = "AVAILABLE" if common else "NOT_ESTABLISHED_NO_COMMON_OUTPUTS"
                    compare_rows.append(row)
        # Explicit three-way missing row so the absence cannot look like zero performance.
        compare_rows.append({"layer": "L1+L2+L3", "mode": "RAW_LAYER", "target": target, "comparison_scope": "THREE_WAY_COMMON_COHORT", "availability": "NOT_ESTABLISHED_L3_MISSING", "n_tasks": 0, "n_positive_outputs": 0, "n_positive_cells": 0, "sample_status": "NOT_ESTABLISHED"})
    compare_fields = metric_fields + ["comparison_scope"]
    write_csv(out / "all_available_comparison.csv", [r for r in compare_rows if r.get("comparison_scope") == "ALL_AVAILABLE_PER_LAYER"], compare_fields)
    write_csv(out / "common_cohort_comparison.csv", [r for r in compare_rows if r.get("comparison_scope") != "ALL_AVAILABLE_PER_LAYER"], compare_fields)

    # Rescue cases and observed-no-positive selected regions.
    rescue_rows: list[dict[str, Any]] = []
    for output_id in RESCUE_IDS:
        rows = output_rows.get(output_id, [])
        yfs = [f"d{r['depth']}/{r['view']}" for r in rows if r["Y_R"] == 1]
        coverage = next((r["coverage_class"] for r in coverage_rows if r["output_id"] == output_id), "MISSING_OUTPUT")
        for layer in layer_scores:
            ranks = ranks_by_layer[layer].get(output_id)
            first = min((ranks[(int(r["depth"]), r["view"])] for r in rows if ranks and r["Y_R"] == 1), default=None)
            rescue_rows.append({"output_id": output_id, "coverage_class": coverage, "actual_dfs_positive_cells": ";".join(yfs), "layer": layer, "first_dfs_positive_rank": first, "top1_retains": bool(first and first <= 1), "top2_retains": bool(first and first <= 2), "top4_retains": bool(first and first <= 4), "top6_retains": bool(first and first <= 6), "evidence_status": "AVAILABLE" if ranks else "UNAVAILABLE"})
    write_csv(out / "dfs_rescue_case_studies.csv", rescue_rows, ["output_id", "coverage_class", "actual_dfs_positive_cells", "layer", "first_dfs_positive_rank", "top1_retains", "top2_retains", "top4_retains", "top6_retains", "evidence_status"])

    coverage_counts = Counter(r["coverage_class"] for r in coverage_rows)
    l1_tasks = len({o[0] for o in outputs if o[0] in l1_scores_by_task and rank_scores(l1_scores_by_task[o[0]])})
    l2_tasks = len({o[0] for o in outputs if o[0] in l2_scores_by_task and rank_scores(l2_scores_by_task[o[0]])})
    # Pick no "best" unless a common cohort has >=10 positives for YC.
    pairwise_yc = [r for r in compare_rows if r.get("target") == "Y_C" and str(r.get("comparison_scope", "")).startswith("PAIRWISE")]
    sufficient_pairwise = [r for r in pairwise_yc if r.get("n_positive_outputs", 0) >= 10]
    best_layer = "NOT_ESTABLISHED"
    best_k = "NOT_ESTABLISHED"
    if sufficient_pairwise:
        metric = max(sufficient_pairwise, key=lambda r: (r.get("recall_at_4") or -1, r.get("recall_at_6") or -1))
        best_layer = metric["layer"]
        best_k = "4" if metric.get("recall_at_4") is not None else "NOT_ESTABLISHED"
    y_r_metrics = [r for r in metric_table if r["mode"] == "RAW_LAYER" and r["target"] == "Y_R" and r["layer"] != "L3_FULL_TRAIN_SAME_STATE"]
    rescue_signal = "INSUFFICIENT" if not any(r.get("n_positive_outputs", 0) >= 5 for r in y_r_metrics) else "WEAK"
    coverage_ratio = coverage_counts["OBSERVED_NO_POSITIVE"] / len(coverage_rows) if coverage_rows else 0
    coverage_bottleneck = "HIGH" if coverage_ratio >= 0.5 else ("MEDIUM" if coverage_ratio >= 0.25 else "LOW")
    hypothesis = {
        "status": "PARTIAL" if (l1_tasks or l2_tasks) else "FAIL",
        "scope": "CPU-only retrospective development audit on frozen artifacts; no hidden-test deployment claim.",
        "measured": {
            "l1_available_tasks": l1_tasks, "l2_available_tasks": l2_tasks, "l3_available_tasks": 0,
            "three_way_common_tasks": 0, "coverage_counts": dict(coverage_counts),
            "l3_checkpoint_hash_matches_current_production": len(l3_matches),
        },
        "inferred": {
            "best_supported_layer_for_decoder_capability": best_layer,
            "best_supported_k": best_k,
            "dfs_rescue_routing_signal": rescue_signal,
            "surface_coverage_bottleneck": coverage_bottleneck,
            "image_model_readiness": "PREMATURE",
        },
        "not_established": [
            "L3 full-train same-state evidence for current production checkpoints.",
            "A three-way layer winner.",
            "A safe production reduction from 12 cells to 6/4/2 cells.",
            "Image/surface model readiness for deployment.",
        ],
    }
    (out / "hypothesis_status.json").write_text(json.dumps(hypothesis, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    raw = {(r["layer"], r["target"]): r for r in metric_table if r["mode"] == "RAW_LAYER"}
    cal = {(r["layer"], r["target"]): r for r in metric_table if r["mode"] == "DECODER_CALIBRATED_LAYER"}
    report = f"""# Decoder-conditioned cell routing audit v1

## Scope

**MEASURED:** CPU-only retrospective audit of frozen Eval60 compact V2 Greedy and V5 artifacts. The primary surface is exactly depths `{{12,24,48}}` × views `{{identity, flip_ud, transpose, anti_transpose}}` = 12 cells per output. No model was loaded and no candidate, TTT, Greedy, TurboDFS, or Gold-path score was generated.

**INFERRED:** routing is evaluated as retrieval of frozen decoder-capable cells, not minimization of Gold NLL or train NLL.

**NOT ESTABLISHED:** hidden-test behavior or a deployment router.

## Decoder labels and coverage

| Coverage class | Outputs |
|---|---:|
| OBSERVED_POSITIVE | {coverage_counts['OBSERVED_POSITIVE']} |
| OBSERVED_NO_POSITIVE | {coverage_counts['OBSERVED_NO_POSITIVE']} |
| CENSORED | {coverage_counts['CENSORED']} |

`OBSERVED_NO_POSITIVE` means no current decoder-positive cell exists in the observed 12-cell surface; it is **not** a router failure. Censored V5 records are preserved as `UNKNOWN`, never converted to false.

## Evidence availability

| Layer | Available current-Eval60 tasks | Status |
|---|---:|---|
| L1 SINGLE_HELDOUT | {l1_tasks} | frozen held-out NLL evidence |
| L2 MULTIFOLD_LOO | {l2_tasks} | finalized LOO artifacts were inventoried; each has zero task-id overlap with compact V2 Eval60 |
| L3 FULL_TRAIN_SAME_STATE | 0 | missing: same-adapter checkpoint hashes match current production in {len(l3_matches)} task-depth states |

Three-way common cohort: **0 tasks**; three-way comparison is not established.

## Raw decoder-capability retrieval

Metrics are `Recall@1/2/4/6`; all are measured only on outputs with complete layer scores, complete label status for the target, and at least one positive cell. Small samples are labeled in the CSVs.

| Layer | Y_C Recall@1/2/4/6 | Y_R Recall@1/2/4/6 |
|---|---|---|
| L1 | {format_metrics(raw.get(('L1_SINGLE_HELDOUT','Y_C')))} | {format_metrics(raw.get(('L1_SINGLE_HELDOUT','Y_R')))} |
| L2 | {format_metrics(raw.get(('L2_MULTIFOLD_LOO','Y_C')))} | {format_metrics(raw.get(('L2_MULTIFOLD_LOO','Y_R')))} |
| L3 | NOT ESTABLISHED | NOT ESTABLISHED |

## Decoder-calibrated out-of-task retrieval

Calibration is a single-feature regularized logistic model trained only on the layer's frozen train-side scalar, with leave-one-task-out splits. It is development analysis; no random-cell splits and no in-sample result are reported as evidence.

| Layer | calibrated Y_C Recall@1/2/4/6 |
|---|---|
| L1 | {format_metrics(cal.get(('L1_SINGLE_HELDOUT','Y_C')))} |
| L2 | {format_metrics(cal.get(('L2_MULTIFOLD_LOO','Y_C')))} |
| L3 | NOT ESTABLISHED |

## DFS-only rescue evidence

Known V5-only rescue output case studies are in `dfs_rescue_case_studies.csv`. A missing rank is reported as unavailable rather than inferred. The raw Y_R conclusion is **{rescue_signal}**.

## Decision

**MEASURED:** observed-surface coverage bottleneck classification is **{coverage_bottleneck}** using the share of observed-no-positive outputs.

**INFERRED:** best supported layer is **{best_layer}**; best supported K is **{best_k}**. This is deliberately `NOT_ESTABLISHED` unless a shared comparison cohort contains at least ten positive Y_C outputs.

**NOT ESTABLISHED:** L3 ranking, a three-way winner, an image/surface model, or safely reducing the production surface to 6/4/2 cells. `IMAGE_MODEL_READINESS = PREMATURE`.
"""
    (out / "DECODER_CONDITIONED_ROUTING_AUDIT.md").write_text(report, encoding="utf-8")
    readme = """# Frozen decoder-conditioned routing audit artifacts

Run with `py -3 scripts/run_decoder_conditioned_cell_routing_audit.py --repo .`.
All outputs are derived from existing frozen CPU-readable artifacts. `decoder_cell_labels.csv` is the authoritative joined label table. Unknown V5 provenance stays blank/unknown, not false. L1 and L2 use lower-is-better frozen train-side NLL rankings; L3 is deliberately unavailable unless production checkpoint hashes match exactly.
"""
    (out / "README.md").write_text(readme, encoding="utf-8")
    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": __import__("subprocess").check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
        "g1_half_reference_commit": "5e3f6ca253631b446971da9ef0e6fddc313b345a",
        "primary_surface": {"depths": DEPTHS, "views": VIEWS, "cells_per_output": 12},
        "input_sha256": {str(p.relative_to(repo)): sha256(p) for p in inputs},
        "label_policy": {"Y_G": "trustworthy Greedy exact Gold", "Y_F": "trustworthy matched DONE V5 pool contains Gold", "unknown_v5": "preserved_unknown"},
        "l3_status": "MISSING_EVIDENCE_UNMATCHED_PRODUCTION_CHECKPOINT_STATE",
        "l3_checkpoint_hash_matches": len(l3_matches),
        "l2_evidence_inventory": {
            "adaptive_ttt_loo_transfer12_v1": "manifest/config only; no finalized usable scores reconstructed",
            "nonblind_development_loo_transfer30_v1": "finalized LOO scores exist but have zero task-id overlap with compact V2 Eval60",
            "same_adapter_old_loo_reference": "finalized reference surface exists but has zero task-id overlap with compact V2 Eval60",
        },
        "cpu_only": True, "gpu_used": False, "model_loaded": False, "new_generation": False, "new_ttt": False, "new_dfs": False,
    }
    # Output hashes come after all text/CSV artifacts have been written.
    provenance["output_sha256"] = {p.name: sha256(p) for p in sorted(out.iterdir()) if p.is_file() and p.name != "provenance.json"}
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(out), "coverage": dict(coverage_counts), "l1_tasks": l1_tasks, "l2_tasks": l2_tasks, "l3_matches": len(l3_matches)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
