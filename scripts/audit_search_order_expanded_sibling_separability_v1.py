#!/usr/bin/env python3
"""CPU-only same-parent expanded-sibling leakage guard for node value.

This audit uses only frozen Phase-3 d24 search trees. A choice group is formed
when an exact Gold-prefix parent has at least two children that were actually
expanded and exactly one expanded child is the next Gold token.

Because siblings share parent, depth, view, and prefix, this removes the main
"shallow node == Gold" shortcut from the global node-prefix classifier.
"""
from __future__ import annotations

import argparse, csv, gzip, hashlib, json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

GATE = {
    "min_choice_groups": 500,
    "min_tasks": 40,
    "roc_auc_min": 0.70,
    "min_fold_roc_auc": 0.60,
    "top1_gain_over_best_baseline_min": 0.05,
    "mrr_gain_over_best_baseline_min": 0.03,
}

BASE_FEATURES = (
    "token_logprob",
    "cumulative_nll",
    "cumulative_regret",
    "regret_increment",
    "is_top1",
    "discrepancy_count",
    "frontier_floor_activated",
    "parent_nll",
    "parent_regret",
    "branch_depth",
    "regret_x_depth",
    "logprob_x_parent_regret",
    "logprob_x_parent_nll",
    "is_top1_x_parent_regret",
)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def readj(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def safe(output_id: str) -> str:
    return output_id.replace(":", "_")


def geom_from_aug(value: str) -> str:
    for part in value.split("__"):
        if part.startswith("geom="):
            return part.split("=", 1)[1]
    raise ValueError(value)


def transform(grid, geom):
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


def grid_tokens(grid):
    out = []
    for i, row in enumerate(grid):
        if i:
            out.append(10)
        out.extend(int(x) for x in row)
    return tuple(out)


def features(row):
    return [float(row[name]) for name in BASE_FEATURES]


def fit_oof(rows):
    X = np.asarray([features(r) for r in rows], dtype=np.float64)
    y = np.asarray([r["y"] for r in rows], dtype=np.int8)
    groups = np.asarray([r["task_id"] for r in rows], dtype=object)
    oof = np.full(len(rows), np.nan, dtype=np.float64)
    folds = []
    splitter = GroupKFold(n_splits=5)
    for fold, (tr, te) in enumerate(splitter.split(X, y, groups)):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[tr])
        Xte = scaler.transform(X[te])
        model = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            solver="lbfgs",
            max_iter=500,
            random_state=0,
        )
        model.fit(Xtr, y[tr])
        pred = model.predict_proba(Xte)[:, 1]
        oof[te] = pred
        folds.append({
            "fold": fold,
            "train_rows": len(tr),
            "test_rows": len(te),
            "test_tasks": len(set(groups[te].tolist())),
            "positive_rows": int(y[te].sum()),
            "roc_auc": float(roc_auc_score(y[te], pred)),
            "average_precision": float(average_precision_score(y[te], pred)),
        })
    if np.isnan(oof).any():
        raise RuntimeError("OOF coverage incomplete")
    return y, oof, folds


def choice_metrics(rows, score=None, mode="learned"):
    by = defaultdict(list)
    for i, row in enumerate(rows):
        by[row["choice_id"]].append(i)
    ranks = []
    for idxs in by.values():
        positives = [i for i in idxs if rows[i]["y"] == 1]
        if len(positives) != 1:
            raise RuntimeError("choice group must have exactly one positive")
        positive = positives[0]
        if mode == "learned":
            ordered = sorted(idxs, key=lambda i: (-float(score[i]), int(rows[i]["node_id"])))
        elif mode == "nll":
            ordered = sorted(idxs, key=lambda i: (float(rows[i]["cumulative_nll"]), int(rows[i]["node_id"])))
        elif mode == "regret":
            ordered = sorted(idxs, key=lambda i: (float(rows[i]["cumulative_regret"]), float(rows[i]["cumulative_nll"]), int(rows[i]["node_id"])))
        elif mode == "lds":
            ordered = sorted(idxs, key=lambda i: (float(rows[i]["discrepancy_count"]), float(rows[i]["cumulative_nll"]), int(rows[i]["node_id"])))
        elif mode == "token_logprob":
            ordered = sorted(idxs, key=lambda i: (-float(rows[i]["token_logprob"]), int(rows[i]["node_id"])))
        else:
            raise ValueError(mode)
        ranks.append(ordered.index(positive) + 1)
    n = len(ranks)
    return {
        "choice_groups": n,
        "top1_accuracy": sum(r == 1 for r in ranks) / n if n else 0.0,
        "top2_accuracy": sum(r <= 2 for r in ranks) / n if n else 0.0,
        "mrr": sum(1.0 / r for r in ranks) / n if n else 0.0,
        "mean_gold_rank": sum(ranks) / n if n else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d24", type=Path, required=True)
    ap.add_argument("--score", type=Path, required=True)
    ap.add_argument("--solutions", type=Path, required=True)
    ap.add_argument("--expected-solutions-sha256", required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)

    gold_sha = sha(a.solutions)
    if gold_sha.lower() != a.expected_solutions_sha256.lower():
        raise RuntimeError("Gold SHA mismatch")
    solutions = readj(a.solutions)
    score_rows = load_csv(a.score)

    rows = []
    group_rows = []
    all_gold_parents = 0
    multi_expanded_gold_parents = 0

    for score_row in score_rows:
        output_id = score_row["output_id"]
        task_id = score_row["task_id"]
        output_index = int(output_id.rsplit(":o", 1)[1])
        with gzip.open(a.d24 / "raw" / f"{safe(output_id)}.json.gz", "rt", encoding="utf-8") as f:
            raw = json.load(f)
        gold = solutions[task_id][output_index]

        for cell_key, cell in raw["cells"].items():
            view = geom_from_aug(cell["augmentation_id"])
            target = grid_tokens(transform(gold, view))
            nodes = sorted(cell["nodes"], key=lambda n: int(n["node_id"]))
            by_id = {int(n["node_id"]): n for n in nodes}
            children = defaultdict(list)
            for node in nodes:
                if node.get("parent_node_id") is not None:
                    children[int(node["parent_node_id"])].append(node)
            roots = [n for n in nodes if n.get("parent_node_id") is None and n.get("selected_token") is None and n.get("state") == "root"]
            if len(roots) != 1:
                raise RuntimeError(f"root cardinality {output_id} {cell_key}")
            root_id = int(roots[0]["node_id"])
            prefix_match = {root_id: True}
            discrepancy = {root_id: 0}

            for node in nodes:
                if node.get("selected_token") is None or node.get("parent_node_id") is None:
                    continue
                node_id = int(node["node_id"])
                parent_id = int(node["parent_node_id"])
                parent_match = bool(prefix_match.get(parent_id, False))
                depth = int(node["branch_depth"])
                token = int(node["selected_token"])
                prefix_match[node_id] = parent_match and depth <= len(target) and token == int(target[depth - 1])
                parent = by_id[parent_id]
                child_regret = float(node.get("cumulative_regret", 0.0) or 0.0)
                parent_regret = float(parent.get("cumulative_regret", 0.0) or 0.0)
                delta = max(0.0, child_regret - parent_regret)
                discrepancy[node_id] = discrepancy.get(parent_id, 0) + (0 if delta <= 1e-6 else 1)

            gold_parent_ids = [node_id for node_id, matched in prefix_match.items() if matched]
            for parent_id in gold_parent_ids:
                parent = by_id[parent_id]
                parent_depth = int(parent.get("branch_depth", 0))
                if parent_depth >= len(target):
                    continue
                all_gold_parents += 1
                expected_token = int(target[parent_depth])
                expanded_children = [
                    n for n in children.get(parent_id, [])
                    if n.get("state") == "expanded" and n.get("selected_token") is not None
                ]
                positive_children = [n for n in expanded_children if int(n["selected_token"]) == expected_token]
                if len(expanded_children) < 2 or len(positive_children) != 1:
                    continue
                multi_expanded_gold_parents += 1
                choice_id = f"{cell_key}:p{parent_id}"
                group_rows.append({
                    "choice_id": choice_id,
                    "task_id": task_id,
                    "output_id": output_id,
                    "cell_key": cell_key,
                    "view": view,
                    "choice_size": len(expanded_children),
                    "branch_depth": parent_depth + 1,
                })
                for child in expanded_children:
                    node_id = int(child["node_id"])
                    child_regret = float(child.get("cumulative_regret", 0.0) or 0.0)
                    parent_regret = float(parent.get("cumulative_regret", 0.0) or 0.0)
                    child_nll = float(child.get("cumulative_score", 0.0) or 0.0)
                    parent_nll = float(parent.get("cumulative_score", 0.0) or 0.0)
                    delta = max(0.0, child_regret - parent_regret)
                    is_top1 = 1.0 if delta <= 1e-6 else 0.0
                    token_logprob = float(child.get("token_logprob", 0.0) or 0.0)
                    depth = float(child.get("branch_depth", parent_depth + 1))
                    rows.append({
                        "task_id": task_id,
                        "output_id": output_id,
                        "cell_key": cell_key,
                        "choice_id": choice_id,
                        "node_id": node_id,
                        "y": int(int(child["selected_token"]) == expected_token),
                        "token_logprob": token_logprob,
                        "cumulative_nll": child_nll,
                        "cumulative_regret": child_regret,
                        "regret_increment": delta,
                        "is_top1": is_top1,
                        "discrepancy_count": float(discrepancy[node_id]),
                        "frontier_floor_activated": 1.0 if child.get("frontier_floor_activated") else 0.0,
                        "parent_nll": parent_nll,
                        "parent_regret": parent_regret,
                        "branch_depth": depth,
                        "regret_x_depth": delta * depth,
                        "logprob_x_parent_regret": token_logprob * parent_regret,
                        "logprob_x_parent_nll": token_logprob * parent_nll,
                        "is_top1_x_parent_regret": is_top1 * parent_regret,
                    })

    if not rows:
        raise RuntimeError("no same-parent expanded-sibling choice groups")
    y, oof, folds = fit_oof(rows)
    learned = choice_metrics(rows, oof, "learned")
    baselines = {
        "CURRENT_DFS_NLL": choice_metrics(rows, mode="nll"),
        "CUMULATIVE_REGRET": choice_metrics(rows, mode="regret"),
        "LDS_UNIT_DISCREPANCY": choice_metrics(rows, mode="lds"),
        "TOKEN_LOGPROB": choice_metrics(rows, mode="token_logprob"),
    }
    roc = float(roc_auc_score(y, oof))
    ap_score = float(average_precision_score(y, oof))
    min_fold = min(float(r["roc_auc"]) for r in folds)
    task_count = len(set(r["task_id"] for r in rows))
    group_count = learned["choice_groups"]
    best_top = max(v["top1_accuracy"] for v in baselines.values())
    best_mrr = max(v["mrr"] for v in baselines.values())
    top_gain = learned["top1_accuracy"] - best_top
    mrr_gain = learned["mrr"] - best_mrr
    strong = (
        group_count >= GATE["min_choice_groups"]
        and task_count >= GATE["min_tasks"]
        and roc >= GATE["roc_auc_min"]
        and min_fold >= GATE["min_fold_roc_auc"]
        and top_gain >= GATE["top1_gain_over_best_baseline_min"]
        and mrr_gain >= GATE["mrr_gain_over_best_baseline_min"]
    )
    signal = "STRONG_EXPANDED_SIBLING_SIGNAL" if strong else "EXPANDED_SIBLING_SIGNAL_NOT_ESTABLISHED"

    summary = {
        "artifact": "SEARCH_ORDER_EXPANDED_SIBLING_DATASET_V1",
        "scope": "POST_FREEZE_NONBLIND_DEVELOPMENT",
        "outputs": len(score_rows),
        "tasks": task_count,
        "all_gold_prefix_parents": all_gold_parents,
        "multi_expanded_gold_parent_groups": multi_expanded_gold_parents,
        "choice_groups": group_count,
        "choice_rows": len(rows),
        "gold_sha256": gold_sha,
        "label": "Within an exact Gold-prefix parent with >=2 expanded children, y=1 iff child token is next Gold token.",
    }
    schema = {
        "artifact": "SEARCH_ORDER_EXPANDED_SIBLING_FEATURE_SCHEMA_V1",
        "features": list(BASE_FEATURES),
        "excluded": [
            "task/output/cell/choice IDs as model inputs",
            "token identity",
            "Gold rank/prefix/future hit",
            "candidate completion",
            "future node state",
        ],
        "model": "StandardScaler + LogisticRegression(C=1,class_weight=balanced,lbfgs)",
        "cv": "5-fold GroupKFold by task_id",
        "gate": GATE,
        "interpretation": "Sibling groups share parent, depth, view and prefix; this is a leakage guard against the global node classifier succeeding only because Gold-prefix nodes are shallow.",
        "limitation": "Only siblings that were actually expanded are reconstructible. Retained-but-never-expanded sibling features are absent from the historical d24 archive.",
    }
    metrics = {
        "artifact": "SEARCH_ORDER_EXPANDED_SIBLING_METRICS_V1",
        "roc_auc": roc,
        "average_precision": ap_score,
        "min_fold_roc_auc": min_fold,
        "learned": learned,
        "baselines": baselines,
        "top1_gain_over_best_baseline": top_gain,
        "mrr_gain_over_best_baseline": mrr_gain,
        "signal": signal,
    }
    decision = {
        "artifact": "SEARCH_ORDER_EXPANDED_SIBLING_DECISION_V1",
        "signal": signal,
        "strong_gate_passed": strong,
        "live_p4_authorized_by_this_guard": strong,
        "next": "PREREGISTER_P4_WITH_TELEMETRY_PRESERVING_CANARY" if strong else "DO_NOT_AUTHORIZE_P4_FROM_GLOBAL_AUC_ALONE",
    }

    def writej(name, obj):
        (a.output / name).write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def writecsv(name, output_rows, fields):
        with (a.output / name).open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
            w.writeheader()
            w.writerows(output_rows)

    writej("DATASET_SUMMARY.json", summary)
    writej("FEATURE_SCHEMA.json", schema)
    writej("METRICS.json", metrics)
    writej("DECISION.json", decision)
    writecsv("GROUPED_CV_FOLDS.csv", folds, list(folds[0]))
    writecsv("CHOICE_GROUPS.csv", group_rows, ["choice_id","task_id","output_id","cell_key","view","choice_size","branch_depth"])
    report = f"""# Same-parent expanded-sibling separability v1

Leakage guard for the global node-prefix classifier.

- choice groups: {group_count}
- unique tasks: {task_count}
- rows: {len(rows)}
- OOF ROC-AUC: {roc:.4f}
- minimum fold ROC-AUC: {min_fold:.4f}
- learned top-1: {learned['top1_accuracy']:.4f}
- best frozen baseline top-1: {best_top:.4f}
- top-1 gain: {top_gain:.4f}
- learned MRR: {learned['mrr']:.4f}
- best frozen baseline MRR: {best_mrr:.4f}
- MRR gain: {mrr_gain:.4f}

Decision: **{signal}**

This result does not cover retained-but-never-expanded siblings because the historical frozen archive did not preserve their diagnostic feature vectors.
"""
    (a.output / "REPORT.md").write_text(report, encoding="utf-8")
    hashes = {p.name: sha(p) for p in sorted(a.output.iterdir()) if p.is_file()}
    writej("HASHES.json", {"files": hashes})
    print(json.dumps({"summary": summary, "metrics": metrics, "decision": decision}, sort_keys=True))


if __name__ == "__main__":
    main()
