#!/usr/bin/env python3
"""CPU-only post-freeze node separability audit for frozen Phase-3 d24 trees.

Gold is used only to label whether an already-expanded retained node lies on
the exact Gold prefix. Model features are restricted to quantities available
at test time when the retained work item is created/scheduled.

This is a development diagnostic, not a blind benchmark and not a router.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

GEOMS = ("anti_transpose", "flip_lr", "flip_ud", "identity", "rot180", "rot270", "rot90", "transpose")
CONTINUOUS = (
    "branch_depth",
    "cumulative_nll",
    "cumulative_regret",
    "local_token_rank",
    "selected_logprob",
    "top1_logprob",
    "top2_logprob",
    "regret_increment",
    "margin",
    "entropy",
    "discrepancy_count",
    "mean_nll_per_depth",
    "mean_regret_per_depth",
)
FEATURE_NAMES = CONTINUOUS + tuple(f"view={g}" for g in GEOMS[1:])
DECISION_RULE = {
    "strong": {
        "roc_auc_min": 0.70,
        "average_precision_gain_over_prevalence_min": 0.10,
        "recall_at_top20_min": 0.40,
        "min_fold_roc_auc": 0.60,
    },
    "weak": {
        "roc_auc_min": 0.60,
        "average_precision_gain_over_prevalence_min": 0.05,
        "recall_at_top20_min": 0.25,
    },
}


def readj(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def geom_from_aug(value: str) -> str:
    for part in value.split("__"):
        if part.startswith("geom="):
            return part.split("=", 1)[1]
    raise ValueError(value)


def transform(grid: list[list[int]], geom: str) -> list[list[int]]:
    a = [list(r) for r in grid]
    if geom == "identity":
        return a
    if geom == "flip_lr":
        return [list(reversed(r)) for r in a]
    if geom == "flip_ud":
        return list(reversed(a))
    if geom == "transpose":
        return [list(r) for r in zip(*a)]
    if geom == "anti_transpose":
        t = [list(r) for r in zip(*a)]
        return [list(reversed(r)) for r in reversed(t)]
    if geom == "rot180":
        return [list(reversed(r)) for r in reversed(a)]
    if geom == "rot90":
        return [list(r) for r in zip(*a)][::-1]
    if geom == "rot270":
        return [list(r) for r in zip(*a[::-1])]
    raise ValueError(geom)


def grid_tokens(grid: list[list[int]]) -> tuple[int, ...]:
    out: list[int] = []
    for i, row in enumerate(grid):
        if i:
            out.append(10)
        out.extend(int(x) for x in row)
    return tuple(out)


def output_safe(output_id: str) -> str:
    return output_id.replace(":", "_")


def load_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def local_distribution(bp: dict, selected_token: int):
    values = [(int(x["token_id"]), float(x["logprob"])) for x in bp["full_arc_logprobs"]]
    ranked = sorted(values, key=lambda pair: (-pair[1], pair[0]))
    rank = next(i for i, (tok, _lp) in enumerate(ranked, start=1) if tok == selected_token)
    selected_lp = next(lp for tok, lp in values if tok == selected_token)
    return {
        "local_token_rank": float(rank),
        "selected_logprob": selected_lp,
        "top1_logprob": float(bp["top1_logprob"]),
        "top2_logprob": float(bp["top2_logprob"]),
        "regret_increment": float(bp["top1_logprob"]) - selected_lp,
        "margin": float(bp["margin"]),
        "entropy": float(bp["entropy"]),
    }


def feature_vector(base: dict) -> list[float]:
    row = [float(base[name]) for name in CONTINUOUS]
    row.extend(1.0 if base["view"] == geom else 0.0 for geom in GEOMS[1:])
    return row


def extract_output(raw: dict, task_id: str, output_id: str, gold_grid: list[list[int]]):
    rows = []
    cell_counts = []
    missing_bp = 0
    for cell_key, cell in raw["cells"].items():
        view = geom_from_aug(cell["augmentation_id"])
        target = grid_tokens(transform(gold_grid, view))
        nodes = sorted(cell["nodes"], key=lambda n: int(n["node_id"]))
        roots = [n for n in nodes if n.get("parent_node_id") is None and n.get("selected_token") is None and n.get("state") == "root"]
        if len(roots) != 1:
            raise RuntimeError(f"{output_id} {cell_key}: root cardinality {len(roots)}")
        root_id = int(roots[0]["node_id"])
        bp_by_parent = {int(x["parent_node_id"]): x for x in cell.get("branch_probabilities", [])}
        match = {root_id: True}
        discrepancy = {root_id: 0}
        positive = negative = skipped = 0
        for node in nodes:
            if node.get("state") != "expanded" or node.get("selected_token") is None:
                continue
            node_id = int(node["node_id"])
            parent_id = int(node["parent_node_id"])
            token = int(node["selected_token"])
            depth = int(node["branch_depth"])
            bp = bp_by_parent.get(parent_id)
            if bp is None:
                skipped += 1
                missing_bp += 1
                match[node_id] = False
                discrepancy[node_id] = discrepancy.get(parent_id, 0)
                continue
            dist = local_distribution(bp, token)
            parent_match = bool(match.get(parent_id, False))
            is_positive = parent_match and depth <= len(target) and token == int(target[depth - 1])
            match[node_id] = is_positive
            disc = discrepancy.get(parent_id, 0) + (0 if int(dist["local_token_rank"]) == 1 else 1)
            discrepancy[node_id] = disc
            nll = float(node.get("cumulative_score", 0.0) or 0.0)
            regret = float(node.get("cumulative_regret", 0.0) or 0.0)
            base = {
                "task_id": task_id,
                "output_id": output_id,
                "cell_key": cell_key,
                "view": view,
                "y": 1 if is_positive else 0,
                "branch_depth": float(depth),
                "cumulative_nll": nll,
                "cumulative_regret": regret,
                "discrepancy_count": float(disc),
                "mean_nll_per_depth": nll / max(1, depth),
                "mean_regret_per_depth": regret / max(1, depth),
                **dist,
            }
            rows.append(base)
            if is_positive:
                positive += 1
            else:
                negative += 1
        cell_counts.append({
            "task_id": task_id,
            "output_id": output_id,
            "cell_key": cell_key,
            "view": view,
            "positive_nodes": positive,
            "negative_nodes": negative,
            "skipped_missing_branch_probability": skipped,
        })
    return rows, cell_counts, missing_bp


def ranking_at_fraction(y: np.ndarray, score: np.ndarray, fraction: float):
    order = np.argsort(-score, kind="stable")
    k = max(1, int(np.floor(len(order) * fraction)))
    chosen = y[order[:k]]
    positives = int(y.sum())
    tp = int(chosen.sum())
    return {
        "fraction": fraction,
        "k": k,
        "recall": tp / positives if positives else 0.0,
        "precision": tp / k,
        "enrichment_vs_random_recall": (tp / positives) / fraction if positives else 0.0,
    }


def macro_cell_recall(rows: list[dict], scores: np.ndarray, fraction: float):
    by_cell = defaultdict(list)
    for i, row in enumerate(rows):
        by_cell[row["cell_key"]].append(i)
    recalls = []
    for idxs in by_cell.values():
        y = np.array([rows[i]["y"] for i in idxs], dtype=np.int8)
        if not y.any():
            continue
        s = np.array([scores[i] for i in idxs], dtype=float)
        order = np.argsort(-s, kind="stable")
        k = max(1, int(np.floor(len(order) * fraction)))
        recalls.append(float(y[order[:k]].sum() / y.sum()))
    return {
        "cells_with_positive": len(recalls),
        "macro_recall": float(np.mean(recalls)) if recalls else 0.0,
        "median_recall": float(np.median(recalls)) if recalls else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d24", type=Path, required=True)
    ap.add_argument("--score", type=Path, required=True)
    ap.add_argument("--solutions", type=Path, required=True)
    ap.add_argument("--expected-solutions-sha256", required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    observed_gold_sha = sha(args.solutions)
    if observed_gold_sha.lower() != args.expected_solutions_sha256.lower():
        raise RuntimeError("Gold SHA mismatch")
    solutions = readj(args.solutions)
    score_rows = load_csv(args.score)

    rows: list[dict] = []
    cell_counts: list[dict] = []
    missing_bp = 0
    output_count = 0
    for score_row in score_rows:
        output_id = score_row["output_id"]
        task_id = score_row["task_id"]
        output_index = int(output_id.rsplit(":o", 1)[1])
        path = args.d24 / "raw" / f"{output_safe(output_id)}.json.gz"
        if not path.is_file():
            raise RuntimeError(f"missing frozen raw tree: {path}")
        with gzip.open(path, "rt", encoding="utf-8") as f:
            raw = json.load(f)
        gold = solutions[task_id][output_index]
        extracted, counts, missing = extract_output(raw, task_id, output_id, gold)
        rows.extend(extracted)
        cell_counts.extend(counts)
        missing_bp += missing
        output_count += 1

    if not rows:
        raise RuntimeError("no node rows extracted")
    X = np.asarray([feature_vector(row) for row in rows], dtype=np.float64)
    y = np.asarray([row["y"] for row in rows], dtype=np.int8)
    groups = np.asarray([row["task_id"] for row in rows], dtype=object)
    prevalence = float(y.mean())
    unique_tasks = sorted(set(groups.tolist()))

    splitter = GroupKFold(n_splits=5)
    oof = np.full(len(y), np.nan, dtype=np.float64)
    fold_rows = []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups)):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])
        model = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            solver="lbfgs",
            max_iter=500,
            random_state=0,
        )
        model.fit(X_train, y[train_idx])
        pred = model.predict_proba(X_test)[:, 1]
        oof[test_idx] = pred
        fold_y = y[test_idx]
        fold_rows.append({
            "fold": fold,
            "train_nodes": len(train_idx),
            "test_nodes": len(test_idx),
            "test_tasks": len(set(groups[test_idx].tolist())),
            "positive_nodes": int(fold_y.sum()),
            "positive_rate": float(fold_y.mean()),
            "roc_auc": float(roc_auc_score(fold_y, pred)),
            "average_precision": float(average_precision_score(fold_y, pred)),
        })
    if np.isnan(oof).any():
        raise RuntimeError("OOF coverage incomplete")

    roc = float(roc_auc_score(y, oof))
    ap_score = float(average_precision_score(y, oof))
    top = {str(int(frac * 100)): ranking_at_fraction(y, oof, frac) for frac in (0.05, 0.10, 0.20)}
    macro = {str(int(frac * 100)): macro_cell_recall(rows, oof, frac) for frac in (0.05, 0.10, 0.20)}
    min_fold_auc = min(float(row["roc_auc"]) for row in fold_rows)
    ap_gain = ap_score - prevalence
    strong = (
        roc >= DECISION_RULE["strong"]["roc_auc_min"]
        and ap_gain >= DECISION_RULE["strong"]["average_precision_gain_over_prevalence_min"]
        and top["20"]["recall"] >= DECISION_RULE["strong"]["recall_at_top20_min"]
        and min_fold_auc >= DECISION_RULE["strong"]["min_fold_roc_auc"]
    )
    weak = (
        roc >= DECISION_RULE["weak"]["roc_auc_min"]
        and ap_gain >= DECISION_RULE["weak"]["average_precision_gain_over_prevalence_min"]
        and top["20"]["recall"] >= DECISION_RULE["weak"]["recall_at_top20_min"]
    )
    signal = "STRONG_NODE_VALUE_SIGNAL" if strong else ("WEAK_NODE_VALUE_SIGNAL" if weak else "NODE_VALUE_SIGNAL_NOT_ESTABLISHED")

    dataset_summary = {
        "artifact": "SEARCH_ORDER_NODE_SEPARABILITY_DATASET_V1",
        "scope": "POST_FREEZE_NONBLIND_DEVELOPMENT",
        "source": "frozen Phase-3 d24 raw search trees",
        "outputs": output_count,
        "tasks": len(unique_tasks),
        "cells": len(cell_counts),
        "expanded_nodes": len(rows),
        "positive_gold_prefix_nodes": int(y.sum()),
        "negative_expanded_nodes": int((1 - y).sum()),
        "positive_rate": prevalence,
        "missing_branch_probability_rows": missing_bp,
        "label": "1 iff expanded retained node prefix exactly equals Gold prefix through that node; else 0",
        "gold_sha256": observed_gold_sha,
    }
    feature_schema = {
        "artifact": "SEARCH_ORDER_NODE_SEPARABILITY_FEATURE_SCHEMA_V1",
        "model_features": list(FEATURE_NAMES),
        "excluded_as_gold_derived": [
            "Gold token/grid identity",
            "Gold prefix fraction",
            "first Gold divergence",
            "Gold rank counters",
            "Gold NLL/regret summaries",
            "exact hit / future rescue",
        ],
        "excluded_for_identity_leakage": ["task_id", "output_id", "cell_key", "token prefix"],
        "available_at_test_time": True,
        "model": "StandardScaler + LogisticRegression(C=1,class_weight=balanced,lbfgs)",
        "cv": "5-fold GroupKFold grouped by task_id",
        "decision_rule": DECISION_RULE,
    }
    metrics = {
        "artifact": "SEARCH_ORDER_NODE_SEPARABILITY_METRICS_V1",
        "roc_auc": roc,
        "average_precision": ap_score,
        "positive_prevalence": prevalence,
        "average_precision_gain_over_prevalence": ap_gain,
        "min_fold_roc_auc": min_fold_auc,
        "top_fraction_metrics": top,
        "macro_cell_recall": macro,
        "signal": signal,
    }
    decision = {
        "artifact": "SEARCH_ORDER_NODE_SEPARABILITY_DECISION_V1",
        "signal": signal,
        "strong_gate_passed": strong,
        "weak_gate_passed": weak,
        "learned_node_value_live_search_authorized": strong,
        "next": (
            "PREREGISTER_P4_VALUE_GUIDED_SEARCH"
            if strong
            else "DO_NOT_LAUNCH_P4_LEARNED_VALUE_YET; inspect feature/label limitations or collect broader target-blind state features"
        ),
        "classic_lds_context": "P3 Micro12 had 0/9 historical-miss rescues; node audit tests whether richer blind state signals can rank Gold-prefix work better than unit discrepancy.",
    }

    def write_json(name, obj):
        (args.output / name).write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def write_csv(name, output_rows, fields):
        with (args.output / name).open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
            w.writeheader()
            w.writerows(output_rows)

    write_json("DATASET_SUMMARY.json", dataset_summary)
    write_json("FEATURE_SCHEMA.json", feature_schema)
    write_json("METRICS.json", metrics)
    write_json("DECISION.json", decision)
    write_csv("GROUPED_CV_FOLDS.csv", fold_rows, list(fold_rows[0]))
    write_csv(
        "CELL_NODE_COUNTS.csv",
        cell_counts,
        ["task_id", "output_id", "cell_key", "view", "positive_nodes", "negative_nodes", "skipped_missing_branch_probability"],
    )
    report = f"""# Search-order node separability v1

CPU-only post-freeze development audit over frozen Phase-3 d24 raw search trees.

## Dataset

- Outputs: {output_count}
- Unique tasks: {len(unique_tasks)}
- Cells: {len(cell_counts)}
- Expanded retained nodes: {len(rows)}
- Gold-prefix positive nodes: {int(y.sum())}
- Positive prevalence: {prevalence:.4f}
- Missing branch-probability feature rows: {missing_bp}

A positive node is an expanded retained node whose entire prefix exactly matches the transformed Gold prefix through that node.
Gold is used for labels only.

## Target-blind model

Fixed model: StandardScaler + LogisticRegression(C=1, class_weight=balanced), 5-fold GroupKFold by task ID.

Features:
{chr(10).join('- ' + name for name in FEATURE_NAMES)}

No task/output IDs, token-prefix identity, Gold rank, Gold NLL, Gold prefix fraction, exact hit, or future rescue fields are model inputs.

## OOF results

- ROC-AUC: {roc:.4f}
- Average precision: {ap_score:.4f}
- Positive prevalence: {prevalence:.4f}
- AP gain: {ap_gain:.4f}
- Minimum fold ROC-AUC: {min_fold_auc:.4f}
- Recall @ top 5% nodes: {top['5']['recall']:.4f}
- Recall @ top 10% nodes: {top['10']['recall']:.4f}
- Recall @ top 20% nodes: {top['20']['recall']:.4f}

Decision: **{signal}**

Strong authorization requires all preregistered gates in FEATURE_SCHEMA.json. A weak or failed result does not authorize live P4 learned-value search.
"""
    (args.output / "REPORT.md").write_text(report, encoding="utf-8")
    hashes = {p.name: sha(p) for p in sorted(args.output.iterdir()) if p.is_file()}
    write_json("HASHES.json", {"files": hashes})
    print(json.dumps({"dataset": dataset_summary, "metrics": metrics, "decision": decision}, sort_keys=True))


if __name__ == "__main__":
    main()
