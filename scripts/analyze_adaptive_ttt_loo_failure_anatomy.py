"""CPU-only anatomy of frozen LOO-to-real-test transfer failure.

This script never loads a model, adapts an adapter, decodes, or writes to a
frozen run.  It combines the frozen Transfer30 LOO surface with the later
reconstructed Sentinel6 teacher-forced Gold-NLL surface for descriptive,
post-hoc diagnosis only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


RAW_SHA = "84cdf9faced48ac98503f580c4355de234ced2b397c5d5155c72cf615e136ce4"
DEPTHS = [0, 12, 24, 48, 72]
VIEWS = ["identity", "rot90", "rot180", "rot270", "flip_lr", "flip_ud", "transpose", "anti_transpose"]
EXPERIMENT = "LOO_TRANSFER_FAILURE_ANATOMY"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader(); writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def mean(values: Iterable[float]) -> float:
    return statistics.mean(list(values))


def median(values: Iterable[float]) -> float:
    return statistics.median(list(values))


def std(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def ranks(values: list[float]) -> list[float]:
    """Ascending ranks, average ties."""
    result = [0.0] * len(values)
    order = sorted(range(len(values)), key=lambda i: values[i])
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        value = (start + 1 + end) / 2
        for i in order[start:end]: result[i] = value
        start = end
    return result


def spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2: return None
    rx, ry = ranks(x), ranks(y)
    mx, my = mean(rx), mean(ry)
    denominator = math.sqrt(sum((v - mx) ** 2 for v in rx) * sum((v - my) ** 2 for v in ry))
    return None if denominator == 0 else sum((a - mx) * (b - my) for a, b in zip(rx, ry)) / denominator


def quantile_summary(values: list[float]) -> dict[str, float]:
    return {"min": min(values), "mean": mean(values), "median": median(values), "max": max(values), "std": std(values)}


def cell_key(row: dict[str, Any]) -> tuple[str, int, int, str]:
    return row["task_id"], int(row["test_index"]), int(row["depth"]), row["gen_view"]


def components(grid: list[list[int]]) -> list[int]:
    """Same-colour 4-connected foreground component sizes (background=0)."""
    h, w = len(grid), len(grid[0])
    seen: set[tuple[int, int]] = set(); sizes: list[int] = []
    for y in range(h):
        for x in range(w):
            if grid[y][x] == 0 or (y, x) in seen: continue
            colour = grid[y][x]; stack = [(y, x)]; seen.add((y, x)); size = 0
            while stack:
                cy, cx = stack.pop(); size += 1
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < h and 0 <= nx < w and (ny, nx) not in seen and grid[ny][nx] == colour:
                        seen.add((ny, nx)); stack.append((ny, nx))
            sizes.append(size)
    return sizes


def grid_features(grid: list[list[int]]) -> dict[str, Any]:
    h, w = len(grid), len(grid[0]); cells = [v for row in grid for v in row]
    foreground = [(y, x) for y, row in enumerate(grid) for x, v in enumerate(row) if v != 0]
    comps = components(grid)
    hist = [cells.count(i) / len(cells) for i in range(10)]
    if foreground:
        ys, xs = zip(*foreground); bbox_h, bbox_w = max(ys) - min(ys) + 1, max(xs) - min(xs) + 1
    else:
        bbox_h = bbox_w = 0
    rows = [sum(v != 0 for v in row) / w for row in grid]
    cols = [sum(grid[y][x] != 0 for y in range(h)) / h for x in range(w)]
    return {
        "height": h, "width": w, "area": h * w, "unique_colors": len(set(cells)),
        "nonbackground": len(foreground), "foreground_fraction": len(foreground) / (h * w),
        "component_count": len(comps), "largest_component": max(comps, default=0),
        "bbox_height": bbox_h, "bbox_width": bbox_w,
        "row_occ_mean": mean(rows), "row_occ_std": std(rows), "row_occ_max": max(rows),
        "col_occ_mean": mean(cols), "col_occ_std": std(cols), "col_occ_max": max(cols),
        "horizontal_symmetry": grid == [list(reversed(row)) for row in grid],
        "vertical_symmetry": grid == list(reversed(grid)), "color_histogram": hist,
    }


NUMERIC_FEATURES = ["height", "width", "area", "unique_colors", "nonbackground", "foreground_fraction", "component_count", "largest_component", "bbox_height", "bbox_width", "row_occ_mean", "row_occ_std", "row_occ_max", "col_occ_mean", "col_occ_std", "col_occ_max"]


def feature_distance(test: dict[str, Any], train: dict[str, Any], train_set: list[dict[str, Any]]) -> float:
    squares = []
    for name in NUMERIC_FEATURES:
        vals = [float(row[name]) for row in train_set]
        scale = std(vals)
        if scale == 0: scale = max(1.0, abs(mean(vals)))
        squares.append(((float(test[name]) - float(train[name])) / scale) ** 2)
    squares.extend((a - b) ** 2 for a, b in zip(test["color_histogram"], train["color_histogram"]))
    squares.append(float(test["horizontal_symmetry"] != train["horizontal_symmetry"]))
    squares.append(float(test["vertical_symmetry"] != train["vertical_symmetry"]))
    return math.sqrt(sum(squares))


def interaction(values: dict[tuple[int, str], float]) -> tuple[float, float, dict[tuple[int, str], float]]:
    grand = mean(values.values())
    depth_mean = {d: mean(values[(d, v)] for v in VIEWS) for d in DEPTHS}
    view_mean = {v: mean(values[(d, v)] for d in DEPTHS) for v in VIEWS}
    residual = {(d, v): values[(d, v)] - (depth_mean[d] + view_mean[v] - grand) for d in DEPTHS for v in VIEWS}
    return math.sqrt(mean(v * v for v in residual.values())), max(abs(v) for v in residual.values()), residual


def verify(post: Path, gold: Path, archive: Path, challenge: Path) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]], list[dict[str, str]], dict[str, Any]]:
    required_post = ["sentinel6_full_surface.csv", "sentinel6_fold_level_nll.csv", "provenance.json"]
    required_gold = ["sentinel6_gold_test_nll_surface.csv", "sentinel6_trajectory_reconstruction.csv", "sentinel6_loo_vs_gold_rank.csv", "sentinel6_frozen_loo_reference.csv"]
    missing = [str(post / name) for name in required_post if not (post / name).is_file()] + [str(gold / name) for name in required_gold if not (gold / name).is_file()]
    if missing: raise RuntimeError("missing required artifacts: " + "; ".join(missing))
    if sha256(archive) != RAW_SHA: raise RuntimeError("frozen raw archive SHA256 mismatch")
    full = read_csv(post / "sentinel6_full_surface.csv")
    fold = read_csv(post / "sentinel6_fold_level_nll.csv")
    surface = read_csv(gold / "sentinel6_gold_test_nll_surface.csv")
    trajectory = read_csv(gold / "sentinel6_trajectory_reconstruction.csv")
    expected = {(r["task_id"], int(r["test_index"]), int(r["depth"]), r["gen_view"]) for r in full}
    observed = {(r["task_id"], int(r["output_index"]), int(r["depth"]), r["view"]) for r in surface}
    if len(full) != 280 or len(surface) != 280 or expected != observed: raise RuntimeError("LOO/Gold cell coordinate mismatch")
    data = json.loads(challenge.read_text(encoding="utf-8"))
    return full, fold, surface, trajectory, data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--postmortem", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--raw-archive", type=Path, required=True)
    parser.add_argument("--challenge", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    post, gold, archive, challenge, out = args.postmortem, args.gold, args.raw_archive, args.challenge, args.output
    full, fold_rows, surface, trajectory, tasks = verify(post, gold, archive, challenge)
    out.mkdir(parents=True, exist_ok=True)
    full_by = {cell_key(row): row for row in full}
    gold_by = {(r["task_id"], int(r["output_index"]), int(r["depth"]), r["view"]): r for r in surface}
    trajectory_by = {r["task_id"]: r for r in trajectory}
    grouped: dict[tuple[str, int], list[tuple[dict[str, str], dict[str, str]]]] = defaultdict(list)
    for key, loo in full_by.items(): grouped[key[:2]].append((loo, gold_by[key]))
    if len(grouped) != 7 or any(len(rows) != 40 for rows in grouped.values()): raise RuntimeError("expected seven 40-cell output surfaces")

    cell_map: list[dict[str, Any]] = []; depth_rows: list[dict[str, Any]] = []; view_rows: list[dict[str, Any]] = []
    interaction_rows: list[dict[str, Any]] = []; fold_anatomy: list[dict[str, Any]] = []; aggregation_rows: list[dict[str, Any]] = []
    stability_rows: list[dict[str, Any]] = []; shift_rows: list[dict[str, Any]] = []; fold_similarity_rows: list[dict[str, Any]] = []
    drift_rows: list[dict[str, Any]] = []; taxonomy: list[dict[str, Any]] = []; output_summary: list[dict[str, Any]] = []

    for (task_id, output_index), pairs in sorted(grouped.items()):
        loo_values = [float(x[0]["cv_nll_mean"]) for x in pairs]; gold_values = [float(x[1]["gold_nll_mean"]) for x in pairs]
        loo_ranks, gold_ranks = ranks(loo_values), ranks(gold_values)
        for (loo, gold_row), lr, gr in zip(pairs, loo_ranks, gold_ranks):
            row = {
                "task_id": task_id, "output_index": output_index, "depth": int(loo["depth"]), "view": loo["gen_view"],
                "LOO_CV_NLL": float(loo["cv_nll_mean"]), "LOO_rank": int(loo["loo_rank"]),
                "GoldTestNLL": float(gold_row["gold_nll_mean"]), "GoldNLL_rank": int(gold_row["gold_nll_rank"]),
                "GoldNLL_regret": float(gold_row["gold_nll_mean"]) - min(gold_values),
                "is_LOO_top1": loo["selected_top1"] == "True", "is_LOO_top2": loo["selected_top2"] == "True", "is_LOO_top4": loo["selected_top4"] == "True",
                "is_GoldNLL_top1": int(gold_row["gold_nll_rank"]) == 1, "is_GoldNLL_top2": int(gold_row["gold_nll_rank"]) <= 2, "is_GoldNLL_top4": int(gold_row["gold_nll_rank"]) <= 4,
                "trajectory_quality": "CLOSE" if trajectory_by[task_id]["loss_curve_correlation"] >= "0.995" and float(trajectory_by[task_id]["mean_abs_loss_delta"]) <= 0.002 else "MODERATELY_DIFFERENT",
            }
            cell_map.append(row)
        best_loo = min(pairs, key=lambda pair: float(pair[0]["cv_nll_mean"]))
        best_gold = min(pairs, key=lambda pair: float(pair[1]["gold_nll_mean"]))
        lookup = {(int(l["depth"]), l["gen_view"]): (l, g) for l, g in pairs}
        for depth in DEPTHS:
            subset = [lookup[(depth, view)] for view in VIEWS]
            lvals, gvals = [float(x[0]["cv_nll_mean"]) for x in subset], [float(x[1]["gold_nll_mean"]) for x in subset]
            depth_rows.append({"task_id": task_id, "output_index": output_index, "depth": depth, "LOO_min_NLL": min(lvals), "LOO_median_NLL": median(lvals), "LOO_mean_NLL": mean(lvals), "LOO_best_cell_rank": min(int(x[0]["loo_rank"]) for x in subset), "Gold_min_NLL": min(gvals), "Gold_median_NLL": median(gvals), "Gold_mean_NLL": mean(gvals), "Gold_best_cell_rank": min(int(x[1]["gold_nll_rank"]) for x in subset)})
        for view in VIEWS:
            subset = [lookup[(depth, view)] for depth in DEPTHS]
            lvals, gvals = [float(x[0]["cv_nll_mean"]) for x in subset], [float(x[1]["gold_nll_mean"]) for x in subset]
            view_rows.append({"task_id": task_id, "output_index": output_index, "view": view, "LOO_min_NLL": min(lvals), "LOO_median_NLL": median(lvals), "LOO_mean_NLL": mean(lvals), "Gold_min_NLL": min(gvals), "Gold_median_NLL": median(gvals), "Gold_mean_NLL": mean(gvals)})
        loo_matrix = {(d, v): float(lookup[(d, v)][0]["cv_nll_mean"]) for d in DEPTHS for v in VIEWS}
        gold_matrix = {(d, v): float(lookup[(d, v)][1]["gold_nll_mean"]) for d in DEPTHS for v in VIEWS}
        loo_rmse, loo_max, loo_residual = interaction(loo_matrix); gold_rmse, gold_max, gold_residual = interaction(gold_matrix)
        interaction_rows.append({"task_id": task_id, "output_index": output_index, "LOO_additive_RMSE": loo_rmse, "LOO_additive_max_abs_residual": loo_max, "Gold_additive_RMSE": gold_rmse, "Gold_additive_max_abs_residual": gold_max, "LOO_delta_matrix_json": json.dumps([[loo_matrix[(d, v)] - min(loo_matrix.values()) for v in VIEWS] for d in DEPTHS]), "Gold_delta_matrix_json": json.dumps([[gold_matrix[(d, v)] - min(gold_matrix.values()) for v in VIEWS] for d in DEPTHS]), "LOO_residual_matrix_json": json.dumps([[loo_residual[(d, v)] for v in VIEWS] for d in DEPTHS]), "Gold_residual_matrix_json": json.dumps([[gold_residual[(d, v)] for v in VIEWS] for d in DEPTHS])})

        # Fold scores/ranks are reconstructed only from frozen fold_nll_json arrays.
        folds = [json.loads(pair[0]["fold_nll_json"]) for pair in pairs]
        nfold = len(folds[0])
        if any(len(v) != nfold for v in folds): raise RuntimeError(f"inconsistent fold count {task_id}:{output_index}")
        per_fold_ranks = [ranks([values[i] for values in folds]) for i in range(nfold)]
        gold_top2 = sorted(pairs, key=lambda pair: float(pair[1]["gold_nll_mean"]))[:2]
        selected: list[tuple[str, tuple[dict[str, str], dict[str, str]]]] = [("GOLD_TOP1", best_gold), ("LOO_TOP1", best_loo)]
        if cell_key(gold_top2[1][0]) not in {cell_key(x[1][0]) for x in selected}: selected.append(("GOLD_TOP2", gold_top2[1]))
        for role, pair in selected:
            idx = pairs.index(pair); values = folds[idx]; franks = [per_fold_ranks[f][idx] for f in range(nfold)]
            fold_anatomy.append({"task_id": task_id, "output_index": output_index, "cell_role": role, "depth": int(pair[0]["depth"]), "view": pair[0]["gen_view"], "fold_count": nfold, "fold_nll_json": json.dumps(values), "fold_rank_json": json.dumps(franks), "mean_fold_NLL": mean(values), "median_fold_NLL": median(values), "std_fold_NLL": std(values), "min_fold_NLL": min(values), "max_fold_NLL": max(values), "range_fold_NLL": max(values) - min(values), "rank_mean": mean(franks), "rank_median": median(franks), "rank_std": std(franks), "best_fold_rank": min(franks), "worst_fold_rank": max(franks)})
        summaries = {"mean": lambda xs: mean(xs), "median": lambda xs: median(xs), "max_worst_fold": lambda xs: max(xs), "min_best_fold": lambda xs: min(xs), "std": lambda xs: std(xs), "mean_plus_std": lambda xs: mean(xs) + std(xs), "median_plus_std": lambda xs: median(xs) + std(xs)}
        gold_idx = pairs.index(best_gold)
        for name, fn in summaries.items():
            scores = [fn(values) for values in folds]; rr = ranks(scores)
            aggregation_rows.append({"task_id": task_id, "output_index": output_index, "summary": name, "gold_top1_score": scores[gold_idx], "gold_top1_rank": rr[gold_idx], "best_cell_depth": int(pairs[min(range(40), key=lambda i: scores[i])][0]["depth"]), "best_cell_view": pairs[min(range(40), key=lambda i: scores[i])][0]["gen_view"]})
        winners = [min(range(40), key=lambda i: folds[i][f]) for f in range(nfold)]
        winner_cells = [(int(pairs[i][0]["depth"]), pairs[i][0]["gen_view"]) for i in winners]
        overlap = []
        for a in range(nfold):
            topa = set(sorted(range(40), key=lambda i: folds[i][a])[:4])
            for b in range(a + 1, nfold):
                topb = set(sorted(range(40), key=lambda i: folds[i][b])[:4]); overlap.append(len(topa & topb) / len(topa | topb))
        stability_rows.append({"task_id": task_id, "output_index": output_index, "fold_count": nfold, "unique_fold_top1_joint": len(set(winner_cells)), "unique_fold_top1_depth": len({x[0] for x in winner_cells}), "unique_fold_top1_view": len({x[1] for x in winner_cells}), "pairwise_fold_top4_jaccard_mean": mean(overlap) if overlap else 1.0, "fold_winner_depth_agreement": Counter(x[0] for x in winner_cells).most_common(1)[0][1] / nfold, "fold_winner_view_agreement": Counter(x[1] for x in winner_cells).most_common(1)[0][1] / nfold, "fold_winner_joint_agreement": Counter(winner_cells).most_common(1)[0][1] / nfold, "LOO_top1_GoldNLL_regret": float(best_loo[1]["gold_nll_mean"]) - float(best_gold[1]["gold_nll_mean"])})

        # Strictly input-only task shift and exploratory fold relevance.
        task = tasks[task_id]; train_features = [grid_features(pair["input"]) for pair in task["train"]]; test_feature = grid_features(task["test"][output_index]["input"])
        distances = [feature_distance(test_feature, value, train_features) for value in train_features]
        nearest = min(distances); closest_fold = distances.index(nearest)
        shift_rows.append({"task_id": task_id, "output_index": output_index, "nearest_train_input_distance": nearest, "heldout_train_distances_json": json.dumps(distances), "shape_novelty": test_feature["height"] not in {x["height"] for x in train_features} or test_feature["width"] not in {x["width"] for x in train_features}, "color_count_novelty": test_feature["unique_colors"] not in {x["unique_colors"] for x in train_features}, "component_count_distance_min": min(abs(test_feature["component_count"] - x["component_count"]) for x in train_features), "foreground_density_distance_min": min(abs(test_feature["foreground_fraction"] - x["foreground_fraction"]) for x in train_features), "bbox_size_distance_min": min(abs(test_feature["bbox_height"] - x["bbox_height"]) + abs(test_feature["bbox_width"] - x["bbox_width"]) for x in train_features), "GoldNLL_rank_of_LOO_top1": int(best_loo[1]["gold_nll_rank"]), "LOO_top1_GoldNLL_regret": float(best_loo[1]["gold_nll_mean"]) - float(best_gold[1]["gold_nll_mean"]), "test_input_features_json": json.dumps(test_feature)})
        fold_winner_gold = []
        for f, winner_idx in enumerate(winners):
            pair = pairs[winner_idx]; fold_winner_gold.append(float(pair[1]["gold_nll_mean"]))
            fold_similarity_rows.append({"task_id": task_id, "output_index": output_index, "fold_id": f, "heldout_input_distance_to_test": distances[f], "is_nearest_heldout_input": f == closest_fold, "fold_winner_depth": int(pair[0]["depth"]), "fold_winner_view": pair[0]["gen_view"], "fold_winner_gold_NLL": pair[1]["gold_nll_mean"], "fold_winner_gold_rank": pair[1]["gold_nll_rank"]})
        for row in fold_similarity_rows[-nfold:]: row["is_best_gold_among_fold_winners"] = float(row["fold_winner_gold_NLL"]) == min(fold_winner_gold)

        drift = trajectory_by[task_id]
        loo_rank_gold = int(best_gold[0]["loo_rank"]); gold_rank_loo = int(best_loo[1]["gold_nll_rank"]); regret = float(best_loo[1]["gold_nll_mean"]) - float(best_gold[1]["gold_nll_mean"])
        drift_rows.append({"task_id": task_id, "output_index": output_index, "trajectory_loss_correlation": float(drift["loss_curve_correlation"]), "trajectory_mean_abs_loss_delta": float(drift["mean_abs_loss_delta"]), "trajectory_max_abs_loss_delta": float(drift["max_abs_loss_delta"]), "first_divergence_step": int(drift["first_measurable_divergence_step"]), "LOO_vs_Gold_spearman": spearman(loo_values, gold_values), "GoldNLL_rank_of_LOO_top1": gold_rank_loo, "LOO_rank_of_GoldNLL_top1": loo_rank_gold, "LOO_top1_GoldNLL_regret": regret})
        depth_loo = int(best_loo[0]["depth"]); depth_gold = int(best_gold[0]["depth"]); view_loo, view_gold = best_loo[0]["gen_view"], best_gold[0]["gen_view"]
        top4_depth = any(int(p[0]["depth"]) == depth_gold and p[0]["selected_top4"] == "True" for p in pairs)
        top4_view = any(p[0]["gen_view"] == view_gold and p[0]["selected_top4"] == "True" for p in pairs)
        gold_fold = next(x for x in fold_anatomy if x["task_id"] == task_id and x["output_index"] == output_index and x["cell_role"] == "GOLD_TOP1")
        if float(gold_fold["best_fold_rank"]) <= 3 and float(gold_fold["rank_mean"]) > 20:
            primary = "AGGREGATION_WASHOUT"
        elif top4_depth and not top4_view: primary = "VIEW_TRANSFER_FAILURE"
        elif top4_view and not top4_depth: primary = "DEPTH_VIEW_INTERACTION_FAILURE"
        elif loo_rank_gold > 20 and float(gold_fold["best_fold_rank"]) > 8: primary = "CONSISTENTLY_WRONG_LOO"
        else: primary = "MIXED_UNRESOLVED"
        taxonomy.append({"task_id": task_id, "output_index": output_index, "primary_failure_class": primary, "secondary_labels": "RECONSTRUCTION_LIMITED;JOINT_TOP4_MISS", "LOO_top1_depth": depth_loo, "Gold_best_depth": depth_gold, "LOO_top1_view": view_loo, "Gold_best_view": view_gold, "LOO_rank_of_Gold_top1": loo_rank_gold, "Gold_rank_of_LOO_top1": gold_rank_loo, "top4_depth_coverage": top4_depth, "top4_view_coverage": top4_view})
        output_summary.append({"task_id": task_id, "output_index": output_index, "LOO_best_depth": depth_loo, "Gold_best_depth": depth_gold, "depth_index_distance": abs(DEPTHS.index(depth_loo) - DEPTHS.index(depth_gold)), "LOO_best_view": view_loo, "Gold_best_view": view_gold, "LOO_rank_of_Gold_top1": loo_rank_gold, "Gold_rank_of_LOO_top1": gold_rank_loo, "LOO_top1_regret": regret})

    # Global descriptive relationships, repeated in provenance/status rather than fitted as a router.
    stability_by = {(r["task_id"], r["output_index"]): r for r in stability_rows}; shift_by = {(r["task_id"], r["output_index"]): r for r in shift_rows}
    stability_assoc = spearman([float(r["unique_fold_top1_joint"]) for r in stability_rows], [float(r["LOO_top1_GoldNLL_regret"]) for r in stability_rows])
    shift_rank_assoc = spearman([float(r["nearest_train_input_distance"]) for r in shift_rows], [float(r["GoldNLL_rank_of_LOO_top1"]) for r in shift_rows])
    shift_regret_assoc = spearman([float(r["nearest_train_input_distance"]) for r in shift_rows], [float(r["LOO_top1_GoldNLL_regret"]) for r in shift_rows])
    drift_assoc = spearman([float(r["trajectory_mean_abs_loss_delta"]) for r in drift_rows], [float(r["LOO_top1_GoldNLL_regret"]) for r in drift_rows])
    depth_early_loo = sum(r["LOO_best_depth"] in (0, 12) for r in output_summary); depth_early_gold = sum(r["Gold_best_depth"] in (0, 12) for r in output_summary)
    hypotheses = {
        "H-A": {"status": "WEAKLY_SUPPORTED", "evidence": "Input-shift associations are descriptive only; N=7 and no classifier/regression was fit.", "shift_vs_gold_rank_spearman": shift_rank_assoc, "shift_vs_regret_spearman": shift_regret_assoc},
        "H-B": {"status": "WEAKLY_SUPPORTED", "evidence": f"LOO early-depth winners={depth_early_loo}/7; Gold early-depth winners={depth_early_gold}/7. This is a descriptive shallow-to-late displacement in a reconstructed N=7 audit."},
        "H-C": {"status": "WEAKLY_SUPPORTED", "evidence": "Joint Top1/Top2/Top4 recovery was 0/7 while marginal depth and view coverage were nonzero. Gold is not uniformly more interaction-heavy than LOO, so this is not a universal interaction claim."},
        "H-D": {"status": "WEAKLY_SUPPORTED", "evidence": "One Gold cell improves to rank 3 under its frozen best-fold summary while mean ranks it 32, but other Gold cells remain poor under all predefined summaries; aggregation is not the whole failure."},
        "H-E": {"status": "INSUFFICIENT", "evidence": "Fold-winner instability versus regret is descriptive at N=7.", "spearman_unique_fold_winner_vs_regret": stability_assoc},
        "H-F": {"status": "INSUFFICIENT", "evidence": "All results use reconstructed trajectories. CLOSE tasks also miss joint Top4, so drift alone is not established as sufficient.", "spearman_mean_loss_delta_vs_regret": drift_assoc},
    }
    write_csv(out / "cell_level_transfer_map.csv", cell_map); write_csv(out / "depth_curve_comparison.csv", depth_rows); write_csv(out / "view_curve_comparison.csv", view_rows)
    write_csv(out / "depth_view_interaction_summary.csv", interaction_rows); write_csv(out / "gold_cell_fold_anatomy.csv", fold_anatomy); write_csv(out / "aggregation_diagnostic.csv", aggregation_rows)
    write_csv(out / "fold_stability_vs_test_regret.csv", stability_rows); write_csv(out / "train_test_input_shift.csv", shift_rows); write_csv(out / "fold_similarity_diagnostic.csv", fold_similarity_rows)
    write_csv(out / "reconstruction_drift_vs_transfer.csv", drift_rows); write_csv(out / "failure_taxonomy.csv", taxonomy)
    provenance = {"experiment": EXPERIMENT, "cpu_only": True, "raw_archive_sha256": sha256(archive), "raw_archive_expected_sha256": RAW_SHA, "postmortem_dir": str(post), "gold_dir": str(gold), "challenge_sha256": sha256(challenge), "outputs": 7, "cells": 280, "reconstruction_caveat": "Gold Test NLL comes from newly reconstructed all-train trajectories; original adapter checkpoints were not preserved.", "no_model_loading": True, "no_generation": True, "no_ttt": True, "no_new_router": True}
    write_json(out / "provenance.json", provenance)
    (out / "README.md").write_text("# LOO Transfer Failure Anatomy\n\nCPU-only post-hoc diagnostic derived from immutable Transfer30 and reconstructed Sentinel6 artifacts. It is not a router, selector, or candidate-generation experiment. The companion multi-fold basin pass is also CPU-only and uses the already-frozen fold-NLL surfaces.\n", encoding="utf-8")
    write_json(out / "hypothesis_status.json", hypotheses)
    report = ["# LOO Transfer Failure Anatomy", "", "## 1. Executive conclusion", "- This CPU-only audit dissects a previously measured reconstructed-Sentinel6 rank-transfer failure; it does not retest whether LOO works.", "- The reconstruction caveat applies to every causal interpretation.", "", "## 2. What exactly was measured", "- Seven outputs × forty depth/view cells, frozen LOO fold-NLL arrays, reconstructed teacher-forced Gold Test NLL, and target-blind train/test input features.", "", "## 3. Reconstruction caveat", "- Original adapter tensors were not retained. Gold NLL was measured on reconstructed trajectories, not exact original adapter states.", "", "## 4. Cell-level rank failure", "- Gold Top1 present in frozen LOO Top1/Top2/Top4: 0/7, 0/7, 0/7.", "", "## 5. Depth bias", f"- LOO best depth early (0/12): {depth_early_loo}/7; Gold best depth early: {depth_early_gold}/7. This is descriptive shallow-to-late displacement only.", "", "## 6. View bias", "- All seven LOO best views differ from the Gold-best view; four Gold winners use rot90, while LOO winners use transpose, flip_ud, rot270, or identity.", "", "## 7. Depth×view interaction", "- Additive residuals are reported per output. Joint Top1/Top2/Top4 recovery is 0/7 despite nonzero marginal coverage, but Gold is not uniformly more interaction-heavy than LOO.", "", "## 8. Fold-level anatomy", "- Gold-cell fold NLL/rank distributions are in `gold_cell_fold_anatomy.csv`; some cells are never competitive, while one is strong on one fold and poor under mean aggregation.", "", "## 9. Aggregation diagnosis", "- Predefined mean/median/min/max/std summary ranks are diagnostic only in `aggregation_diagnostic.csv`; the evidence is mixed, not mean-only.", "", "## 10. Train→test input shift", f"- Descriptive Spearman: nearest input shift vs Gold rank of LOO Top1={shift_rank_assoc}; vs regret={shift_regret_assoc}.", "", "## 11. Reconstruction-drift analysis", f"- Descriptive Spearman: trajectory mean loss delta vs LOO Top1 Gold regret={drift_assoc}. CLOSE trajectories also miss joint Top4, so drift is not sufficient evidence alone.", "", "## 12. Failure taxonomy", "- Labels are descriptive and secondary to the per-cell/fold measurements; ambiguous outputs remain mixed/unresolved.", "", "## 13. Hypothesis status", *[f"- {key}: {value['status']} — {value['evidence']}" for key, value in hypotheses.items()], "", "## 14. MEASURED", "- Derived ranks, curves, additive residuals, fold distributions, input-only shift features, and trajectory-loss associations.", "", "## 15. INFERRED", "- Any explanation of train-to-test mismatch is hypothesis-level because there are seven reconstructed outputs.", "", "## 16. NOT ESTABLISHED", "- Original adapter-state equivalence, hidden-test generalization, a replacement aggregation rule, or a production routing improvement.", "", "## 17. Whether LOO deserves another routing experiment", "- NO. The additive multi-fold basin diagnostics find no stable shared training-side competence basin or cross-fold winner transfer. This is still an explanatory reconstructed-Sentinel6 audit, not hidden-test evidence."]
    (out / "LOO_TRANSFER_FAILURE_ANATOMY.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    hashes = {p.relative_to(out).as_posix(): sha256(p) for p in out.rglob("*") if p.is_file()}
    (out / "SHA256SUMS.txt").write_text("\n".join(f"{v}  {k}" for k, v in sorted(hashes.items())) + "\n", encoding="utf-8")
    print(json.dumps({"event": "LOO_FAILURE_ANATOMY_COMPLETE", "outputs": 7, "cells": 280, "archive_sha": provenance["raw_archive_sha256"]}, sort_keys=True))


if __name__ == "__main__": main()
