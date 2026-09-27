#!/usr/bin/env python3
"""CPU-only post-mortem of the frozen NONBLIND_DEVELOPMENT_LOO_TRANSFER30_V1 run.

The script deliberately reads frozen predictions before opening the evaluation
solutions, verifies the raw-file manifest, and writes analysis only beneath the
caller-supplied output directory.  It never mutates the frozen run directory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


EXPECTED = {
    "FIXED_24_IDENTITY": 3,
    "LOO_TOP1": 2,
    "LOO_TOP2": 2,
    "LOO_TOP4": 4,
}
TOP_LABELS = {1: "LOO_TOP1", 2: "LOO_TOP2", 4: "LOO_TOP4"}
FROZEN_STATUS = "FROZEN_BEFORE_ANY_EVAL_SOLUTION_ACCESS"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def grid_key(value: Any) -> str | None:
    return canonical(value) if isinstance(value, list) and value else None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frozen(path: Path) -> dict[str, Any]:
    value = read_json(path)
    if value.get("status") != FROZEN_STATUS or value.get("solutions_opened") is not False:
        raise RuntimeError(f"not a target-blind frozen artifact: {path}")
    return value


def verify_manifest(raw: Path) -> dict[str, Any]:
    lines = (raw / "FINAL_SHA256SUMS.txt").read_text(encoding="utf-8").splitlines()
    mismatches: list[str] = []
    for line in lines:
        digest, sep, rel = line.partition("  ./")
        if not sep or len(digest) != 64:
            mismatches.append(f"malformed:{line}")
            continue
        path = raw / rel.replace("/", "\\")
        if not path.is_file():
            mismatches.append(f"missing:{rel}")
        elif sha256(path) != digest:
            mismatches.append(f"digest:{rel}")
    if mismatches:
        raise RuntimeError(f"raw frozen-manifest mismatch: {mismatches[:10]}")
    return {"entries": len(lines), "mismatches": 0, "manifest_sha256": sha256(raw / "FINAL_SHA256SUMS.txt")}


def mean(values: Iterable[float]) -> float:
    value = list(values)
    return sum(value) / len(value) if value else math.nan


def population_std(values: Iterable[float]) -> float:
    value = list(values)
    return statistics.pstdev(value) if len(value) > 1 else 0.0


def rankdata(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    result = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        average_rank = (cursor + 1 + end) / 2.0
        for item in order[cursor:end]:
            result[item] = average_rank
        cursor = end
    return result


def pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    lmean, rmean = mean(left), mean(right)
    numerator = sum((a - lmean) * (b - rmean) for a, b in zip(left, right))
    denominator = math.sqrt(sum((a - lmean) ** 2 for a in left) * sum((b - rmean) ** 2 for b in right))
    return numerator / denominator if denominator else None


def auroc_lower_is_better(scores: list[float], labels: list[int]) -> float | None:
    positives = [score for score, label in zip(scores, labels) if label]
    negatives = [score for score, label in zip(scores, labels) if not label]
    if not positives or not negatives:
        return None
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            wins += 1.0 if positive < negative else 0.5 if positive == negative else 0.0
    return wins / (len(positives) * len(negatives))


def average_precision_lower_is_better(scores: list[float], labels: list[int]) -> float | None:
    total_positive = sum(labels)
    if not total_positive:
        return None
    ordered = sorted(zip(scores, labels), key=lambda item: item[0])
    found = 0
    precision_sum = 0.0
    for index, (_, label) in enumerate(ordered, 1):
        if label:
            found += 1
            precision_sum += found / index
    return precision_sum / total_positive


def truth(solutions: dict[str, Any], task_id: str, test_index: int) -> str:
    return grid_key(solutions[task_id][test_index]) or ""


def prediction_sets(rows: list[dict[str, Any]], label: str | None = None, baseline: str | None = None) -> dict[tuple[str, int], set[str]]:
    result: dict[tuple[str, int], set[str]] = defaultdict(set)
    for row in rows:
        if label is not None and label not in row.get("labels", []):
            continue
        if baseline is not None and row.get("baseline") != baseline:
            continue
        key = grid_key(row.get("prediction"))
        if key is not None:
            result[(str(row["task_id"]), int(row["test_index"]))].add(key)
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({field for row in rows for field in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def selected_pairs(ranking: dict[str, Any], topk: int) -> set[tuple[int, str]]:
    return {(int(row["depth"]), str(row["gen_view"])) for row in ranking[f"top{topk}"]}


def basin_label(exact_count: int) -> str:
    # Fixed pre-analysis descriptive bins: 1 isolated, 2-4 medium, >=5 broad.
    if exact_count <= 1:
        return "ISOLATED"
    if exact_count <= 4:
        return "MEDIUM"
    return "BROAD"


def markdown(report: dict[str, Any]) -> str:
    scores = report["score_reproduction"]["counts"]
    sent = report["sentinel6"]
    lines = [
        "# Adaptive TTT / LOO Transfer30 post-mortem",
        "",
        "## Executive conclusion",
        "",
        "**Recommendation: B — stop LOO as selector, but retain adaptive depth/view as a candidate-generation axis.**",
        "The frozen LOO ranking did not transfer to the two Sentinel outputs for which the 40-cell surface contains an exact candidate: all exact cells ranked outside Top4. The result is a nonblind development mechanism result, not held-out generalization evidence.",
        "",
        "## Final measured score reproduction",
        "",
        f"- Fixed TTT24 identity: {scores['FIXED_24_IDENTITY']}/{report['score_reproduction']['outputs']}",
        f"- LOO Top1: {scores['LOO_TOP1']}/{report['score_reproduction']['outputs']}",
        f"- LOO Top2: {scores['LOO_TOP2']}/{report['score_reproduction']['outputs']}",
        f"- LOO Top4: {scores['LOO_TOP4']}/{report['score_reproduction']['outputs']}",
        "",
        "## Sentinel6 forensic",
        "",
        f"- Outputs: {sent['outputs']}; exact full-surface cells: {sent['exact_cells']}; oracle outputs: {sent['oracle_outputs']}.",
        f"- Exact-cell LOO ranks: {sent['exact_rank_list']}.",
        f"- Exact-cell rank coverage: Top1={sent['topk_exact_cell_coverage']['top1']}, Top2={sent['topk_exact_cell_coverage']['top2']}, Top4={sent['topk_exact_cell_coverage']['top4']}.",
        f"- Lower-NLL-is-better association: AUROC={sent['score_association']['auroc']}, AUPRC={sent['score_association']['auprc']}, Spearman={sent['score_association']['spearman_rank_vs_exact']}.",
        "",
        "| oracle output | exact depth×view cells (LOO ranks) | Top1 depth×view | Top4 depth-family / view-family coverage | basin |",
        "|---|---|---|---|---|",
    ]
    for item in sent["oracle_output_records"]:
        exact = ", ".join(f"{depth}/{view} (r{rank})" for (depth, view), rank in zip(item["exact_pairs"], item["exact_ranks"]))
        coverage = item["topk_coverage"]["top4"]
        lines.append(f"| {item['task_id']}:{item['test_index']} | {exact} | {item['top1_pair'][0]}/{item['top1_pair'][1]} | depth={coverage['depth_family']}; view={coverage['view_family']}; exact={coverage['exact_cell']} | {item['basin']} |")
    lines.extend([
        "",
        "## Why LOO did not transfer",
        "",
        "MEASURED: all real-test exact Sentinel cells were ranked 18, 24, 25, or 32 of 40 by the frozen mean held-out-train NLL. The selected Top4 never contained one. This is direct evidence against transfer of this ranking to the sampled real-test-optimal depth×view states.",
        "",
        "INFERRED: the frozen teacher-forced objective rewards reconstruction of held-out train-pair trajectories, which can prefer states that do not extrapolate to the test input. The evidence is consistent with train-pair competence and test extrapolation being decoupled; it does not isolate a causal mechanism inside the adapter.",
        "",
        "NOT ESTABLISHED: a universal claim that LOO can never be useful, a population estimate from the two positive Sentinel outputs, or a replacement selector.",
        "",
        "## What Step1 actually established",
        "",
        "The previously reported 8/8 Step1 pseudo-test result supports held-out-train reconstruction/competence retrieval in that pseudo-test regime. It did not establish that the same ranking chooses a state that extrapolates to a real test input. Transfer30 directly tests the latter and fails on the available Sentinel oracle outputs.",
        "",
        "## Adaptive state versus router value",
        "",
        f"Adaptive-state value: **{report['hypotheses']['H4']['status']}**. {report['hypotheses']['H4']['evidence']}",
        "",
        "## Failure-layer attribution",
        "",
        f"- **A — wrong ability measured:** not established as the primary failure. The historical Step1 pseudo-test supports a held-out-train reconstruction signal, but Transfer30 does not re-run that historical comparison.",
        f"- **B — train competence versus test extrapolation:** inferred plausible. Exact cells have ranks {sent['exact_rank_list']} despite being real-test correct.",
        f"- **C — aggregation alone:** not sufficient. Correct cells are outside Top4, so a fixed Top1/2/4 aggregation boundary misses them before any final choice among those cells.",
        f"- **D — fold instability:** measured on exact cells in `sentinel6_exact_cell_ranks.csv`; fold-rank standard deviations are nonzero and range from {min(row['fold_rank_std'] for row in sent['exact_cell_rows']):.3f} to {max(row['fold_rank_std'] for row in sent['exact_cell_rows']):.3f}.",
        f"- **E — narrow/exceptional real-test states:** measured descriptively. Each oracle output has 2/40 exact cells, hence MEDIUM under the predeclared 1 / 2-4 / >=5 bins but still only 5% of its surface.",
        f"- **F — state-space value:** weak positive evidence: both oracle outputs are absent from Fixed TTT24 identity and present at adaptive depth/view cells.",
        f"- **G — depth×view interaction:** primary observed selector failure. One oracle output has Top4 correct-view coverage but no correct-depth coverage; the other has Top2 correct-depth coverage but no correct-view coverage; neither has an exact selected cell.",
        "",
        "## Hypotheses",
        "",
    ])
    for key, value in report["hypotheses"].items():
        lines.append(f"- **{key}: {value['status']}** — {value['evidence']}")
    lines.extend([
        "",
        "## Provenance and scope",
        "",
        "This analysis is CPU-only and reads frozen raw artifacts. Evaluation solutions were opened only after the verified target-blind GPU freeze. The cohort is historically exposed but not adaptive-designed; it is not untouched, held-out, or final generalization evidence.",
        "",
    ])
    return "\n".join(lines)


def main(args: argparse.Namespace) -> None:
    raw, output, solutions_path = args.raw_output.resolve(), args.output.resolve(), args.solutions.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing to overwrite nonempty output directory: {output}")
    output.mkdir(parents=True, exist_ok=False)

    # All target-blind status and byte checks precede reading solutions.
    manifest_check = verify_manifest(raw)
    gpu_freeze = read_json(raw / "GPU_FREEZE_COMPLETE.json")
    manifest = read_json(raw / "manifest.json")
    cohort = read_json(raw / "cohort.json")
    sentinel = read_json(raw / "sentinel6.json")
    rankings_artifact = frozen(raw / "loo_rankings_frozen.json")
    predictions_artifact = frozen(raw / "production_predictions_frozen.json")
    baselines_artifact = frozen(raw / "fixed_baselines_frozen.json")
    surface_artifact = frozen(raw / "sentinel6_full_surface_frozen.json")
    if gpu_freeze.get("solutions_opened") is not False or gpu_freeze.get("tasks") != 30 or gpu_freeze.get("folds") != 97:
        raise RuntimeError("incomplete or contaminated GPU freeze")
    if cohort.get("solutions_opened") is not False or sentinel.get("solutions_opened") is not False:
        raise RuntimeError("cohort/sentinel was not frozen target-blind")

    entries = {str(row["task_id"]): row for row in cohort["entries"]}
    keys = [(task_id, index) for task_id in cohort["task_ids"] for index in range(int(entries[task_id]["test_output_count"]))]
    if len(keys) != 41:
        raise RuntimeError(f"expected 41 cohort outputs, got {len(keys)}")

    rankings = {str(row["task_id"]): row for row in rankings_artifact["rankings"]}
    if set(rankings) != set(cohort["task_ids"]):
        raise RuntimeError("ranking task set does not match frozen cohort")

    # Only now may this retrospective analysis read solutions.
    solutions = read_json(solutions_path)
    if any(task not in solutions for task, _ in keys):
        raise RuntimeError("solution file does not cover frozen cohort")

    prediction_rows, baseline_rows, surface_rows = predictions_artifact["predictions"], baselines_artifact["predictions"], surface_artifact["predictions"]
    sets = {name: prediction_sets(prediction_rows, label=name) for name in TOP_LABELS.values()}
    sets["FIXED_24_IDENTITY"] = prediction_sets(baseline_rows, baseline="FIXED_24_IDENTITY")
    hitmaps: dict[str, dict[tuple[str, int], bool]] = {}
    for name, candidate_set in sets.items():
        hitmaps[name] = {key: truth(solutions, *key) in candidate_set.get(key, set()) for key in keys}
    counts = {name: sum(hitmap.values()) for name, hitmap in hitmaps.items()}
    if {name: counts[name] for name in EXPECTED} != EXPECTED:
        raise RuntimeError(f"frozen score reproduction mismatch: {counts}")

    all41_rows: list[dict[str, Any]] = []
    for task_id, test_index in keys:
        fixed, top1, top2, top4 = (hitmaps[name][(task_id, test_index)] for name in ("FIXED_24_IDENTITY", "LOO_TOP1", "LOO_TOP2", "LOO_TOP4"))
        all41_rows.append({
            "task_id": task_id,
            "test_index": test_index,
            "fixed24_identity_exact": fixed,
            "loo_top1_exact": top1,
            "loo_top2_exact": top2,
            "loo_top4_exact": top4,
            "fixed_only": fixed and not top4,
            "adaptive_only_top4": top4 and not fixed,
            "shared_fixed_top4": fixed and top4,
            "top1_to_top2_incremental_rescue": top2 and not top1,
            "top2_to_top4_incremental_rescue": top4 and not top2,
        })
    write_csv(output / "all41_overlap.csv", all41_rows)

    # Fold NLL lookup uses the raw records rather than inferred ordering.
    fold_scores: dict[tuple[str, int], dict[tuple[int, str], float]] = {}
    for line in (raw / "loo_fold_scores.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        fold_scores[(str(row["task_id"]), int(row["fold_id"]))] = {(int(score["depth"]), str(score["gen_view"])): float(score["nll_per_token"]) for score in row["scores"]}

    sentinel_ids = [str(item) for item in sentinel["task_ids"]]
    sentinel_keys = [key for key in keys if key[0] in set(sentinel_ids)]
    if len(sentinel_keys) != 7 or len(surface_rows) != 280:
        raise RuntimeError(f"expected 7 Sentinel outputs and 280 surface rows, got {len(sentinel_keys)}, {len(surface_rows)}")
    surface_by_output: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in surface_rows:
        key = (str(row["task_id"]), int(row["test_index"]))
        if key not in sentinel_keys:
            raise RuntimeError(f"unexpected Sentinel surface key: {key}")
        surface_by_output[key].append(row)

    full_surface_rows: list[dict[str, Any]] = []
    exact_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    all_signal_scores: list[float] = []
    all_signal_labels: list[int] = []
    oracle_output_records: list[dict[str, Any]] = []
    topk_exact_cell_coverage = {"top1": 0, "top2": 0, "top4": 0}
    topk_depth_coverage = {"top1": 0, "top2": 0, "top4": 0}
    topk_view_coverage = {"top1": 0, "top2": 0, "top4": 0}

    for key in sentinel_keys:
        task_id, test_index = key
        ranking = rankings[task_id]
        cells = ranking["all_40_cells"]
        rank_cells = {(int(cell["depth"]), str(cell["gen_view"])): cell for cell in cells}
        observed = surface_by_output[key]
        if len(observed) != 40 or {(int(row["depth"]), str(row["gen_view"])) for row in observed} != set(rank_cells):
            raise RuntimeError(f"surface/ranking mismatch for {key}")
        exact_pairs = {(int(row["depth"]), str(row["gen_view"])) for row in observed if grid_key(row.get("prediction")) == truth(solutions, *key)}
        top_pairs = {topk: selected_pairs(ranking, topk) for topk in TOP_LABELS}
        fixed_exact = hitmaps["FIXED_24_IDENTITY"][key]
        top1_pair = next(iter(top_pairs[1]))
        top1_cell = rank_cells[top1_pair]
        per_fold_ranks: dict[tuple[int, str], list[int]] = defaultdict(list)
        for fold_id in sorted(index for task, index in fold_scores if task == task_id):
            ordered = sorted(fold_scores[(task_id, fold_id)].items(), key=lambda item: item[1])
            for rank, (pair, _) in enumerate(ordered, 1):
                per_fold_ranks[pair].append(rank)
        for pair, cell in rank_cells.items():
            exact = pair in exact_pairs
            row = {
                "task_id": task_id, "test_index": test_index, "depth": pair[0], "gen_view": pair[1],
                "real_test_exact": exact, "loo_rank": int(cell["loo_rank"]),
                "loo_percentile": int(cell["loo_rank"]) / 40.0,
                "cv_nll_mean": float(cell["cv_nll_per_token_mean"]),
                "cv_nll_median": float(cell["cv_nll_per_token_median"]),
                "cv_nll_worst": float(cell["cv_nll_per_token_worst"]),
                "fold_nll_json": canonical(cell["fold_nll_per_token"]),
                "fold_rank_mean": mean(per_fold_ranks[pair]), "fold_rank_std": population_std(per_fold_ranks[pair]),
                "selected_top1": pair in top_pairs[1], "selected_top2": pair in top_pairs[2], "selected_top4": pair in top_pairs[4],
            }
            full_surface_rows.append(row)
            all_signal_scores.append(float(cell["cv_nll_per_token_mean"]))
            all_signal_labels.append(int(exact))
            if exact:
                values = [float(item) for item in cell["fold_nll_per_token"]]
                exact_rows.append({
                    **row,
                    "nll_gap_vs_top1": float(cell["cv_nll_per_token_mean"]) - float(top1_cell["cv_nll_per_token_mean"]),
                    "fold_nll_min": min(values), "fold_nll_max": max(values), "fold_nll_std": population_std(values),
                    "same_depth_top1": pair[0] == top1_pair[0], "same_view_top1": pair[1] == top1_pair[1],
                })
        roles_by_pair: dict[tuple[int, str], list[str]] = defaultdict(list)
        for pair in exact_pairs: roles_by_pair[pair].append("EXACT")
        for topk, pairs in top_pairs.items():
            for pair in pairs: roles_by_pair[pair].append(f"TOP{topk}")
        for pair, roles in roles_by_pair.items():
            for fold_id in sorted(index for task, index in fold_scores if task == task_id):
                fold_rows.append({"task_id": task_id, "test_index": test_index, "depth": pair[0], "gen_view": pair[1], "roles": "|".join(roles), "fold_id": fold_id, "nll_per_token": fold_scores[(task_id, fold_id)][pair], "fold_rank": per_fold_ranks[pair][fold_id]})
        if exact_pairs:
            exact_depths, exact_views = {pair[0] for pair in exact_pairs}, {pair[1] for pair in exact_pairs}
            for topk, pairs in top_pairs.items():
                topk_exact_cell_coverage[f"top{topk}"] += int(bool(exact_pairs & pairs))
                topk_depth_coverage[f"top{topk}"] += int(bool({pair[0] for pair in pairs} & exact_depths))
                topk_view_coverage[f"top{topk}"] += int(bool({pair[1] for pair in pairs} & exact_views))
            coverage = {}
            for topk, pairs in top_pairs.items():
                coverage[f"top{topk}"] = {
                    "exact_cell": bool(exact_pairs & pairs),
                    "depth_family": bool({pair[0] for pair in pairs} & exact_depths),
                    "view_family": bool({pair[1] for pair in pairs} & exact_views),
                }
            oracle_output_records.append({
                "task_id": task_id, "test_index": test_index, "exact_cells": len(exact_pairs),
                "exact_depths": sorted(exact_depths), "exact_views": sorted(exact_views),
                "largest_same_depth_exact_cluster": max(sum(pair[0] == depth for pair in exact_pairs) for depth in exact_depths),
                "largest_same_view_exact_cluster": max(sum(pair[1] == view for pair in exact_pairs) for view in exact_views),
                "basin": basin_label(len(exact_pairs)),
                "fixed24_identity_exact": fixed_exact,
                "top1_pair": list(top1_pair), "top1_rank": int(top1_cell["loo_rank"]),
                "top1_nll": float(top1_cell["cv_nll_per_token_mean"]),
                "exact_pairs": [list(pair) for pair in sorted(exact_pairs)],
                "exact_ranks": [int(rank_cells[pair]["loo_rank"]) for pair in sorted(exact_pairs)],
                "topk_coverage": coverage,
            })

    write_csv(output / "sentinel6_full_surface.csv", full_surface_rows)
    write_csv(output / "sentinel6_exact_cell_ranks.csv", exact_rows)
    write_csv(output / "sentinel6_fold_level_nll.csv", fold_rows)

    association = {
        "positive_exact_cells": sum(all_signal_labels), "total_cells": len(all_signal_labels),
        "auroc": auroc_lower_is_better(all_signal_scores, all_signal_labels),
        "auprc": average_precision_lower_is_better(all_signal_scores, all_signal_labels),
        "base_rate": sum(all_signal_labels) / len(all_signal_labels),
        "spearman_rank_vs_exact": pearson(rankdata([row["loo_rank"] for row in full_surface_rows]), rankdata([float(label) for label in all_signal_labels])),
    }
    exact_rank_list = sorted(row["loo_rank"] for row in exact_rows)
    depth_view_summary = {
        "oracle_outputs": len(oracle_output_records),
        "exact_cells": len(exact_rows),
        "oracle_output_records": oracle_output_records,
        "topk_exact_cell_coverage": topk_exact_cell_coverage,
        "topk_depth_family_coverage": topk_depth_coverage,
        "topk_view_family_coverage": topk_view_coverage,
        "diagnosis": "DEPTH_VIEW_INTERACTION_FAILURE: no Top1/2/4 candidate was an exact depth×view cell; family coverage is reported descriptively and cannot support a population conclusion from two oracle outputs.",
    }
    (output / "depth_view_failure_summary.json").write_text(json.dumps(depth_view_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    fixed_missed_oracle = [item for item in oracle_output_records if not item["fixed24_identity_exact"]]
    hypotheses = {
        "H1": {"status": "SUPPORTED", "evidence": "Historical Step1's reported 8/8 pseudo-test result supports held-out-train competence retrieval in that train-reconstruction regime; it was not re-run here."},
        "H2": {"status": "SUPPORTED", "evidence": "The frozen LOO protocol defines and scores train-pair competence basins. This support is limited to the train-pair teacher-forced objective, not test extrapolation."},
        "H3": {"status": "NOT SUPPORTED", "evidence": f"All {len(exact_rows)} Sentinel exact cells ranked {exact_rank_list}; none was in frozen Top4."},
        "H4": {"status": "WEAK" if fixed_missed_oracle else "NOT ESTABLISHED", "evidence": f"{len(fixed_missed_oracle)}/{len(oracle_output_records)} Sentinel oracle outputs were exact under adaptive depth/view cells while Fixed TTT24 identity was not exact. The sample is only {len(oracle_output_records)} oracle outputs."},
    }
    (output / "hypothesis_status.json").write_text(json.dumps(hypotheses, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    categories = {
        "fixed_only": [f"{row['task_id']}:{row['test_index']}" for row in all41_rows if row["fixed_only"]],
        "top1_hits": [f"{row['task_id']}:{row['test_index']}" for row in all41_rows if row["loo_top1_exact"]],
        "top1_to_top2_incremental_rescues": [f"{row['task_id']}:{row['test_index']}" for row in all41_rows if row["top1_to_top2_incremental_rescue"]],
        "top2_to_top4_incremental_rescues": [f"{row['task_id']}:{row['test_index']}" for row in all41_rows if row["top2_to_top4_incremental_rescue"]],
        "adaptive_only_top4": [f"{row['task_id']}:{row['test_index']}" for row in all41_rows if row["adaptive_only_top4"]],
        "shared_fixed_top4": [f"{row['task_id']}:{row['test_index']}" for row in all41_rows if row["shared_fixed_top4"]],
    }
    report = {
        "experiment": manifest["experiment_id"],
        "scope": "development transfer / mechanism validation only; not unseen, untouched, held-out, strict blind validation, or final generalization evidence",
        "provenance": {
            "raw_output": str(raw), "raw_manifest": manifest_check, "gpu_freeze": gpu_freeze,
            "source_commit": manifest["source_commit"], "frozen_artifact_git_commit": args.frozen_git_commit,
            "scoring_git_commit": args.scoring_git_commit, "archive_sha256": args.archive_sha256,
            "solutions_opened_after_verified_gpu_freeze": True,
            "solutions_path": str(solutions_path), "task_count": len(cohort["task_ids"]), "output_count": len(keys), "fold_count": gpu_freeze["folds"],
        },
        "score_reproduction": {"pass": True, "outputs": len(keys), "counts": counts, "expected": EXPECTED, "overlap_categories": categories},
        "sentinel6": {
            "outputs": len(sentinel_keys), "exact_cells": len(exact_rows), "oracle_outputs": len(oracle_output_records),
            "exact_rank_list": exact_rank_list, "score_association": association,
            "topk_exact_cell_coverage": topk_exact_cell_coverage,
            "topk_depth_family_coverage": topk_depth_coverage,
            "topk_view_family_coverage": topk_view_coverage,
            "basin_rule": "ISOLATED=1 exact cell; MEDIUM=2-4; BROAD>=5; descriptive only.",
            "oracle_output_records": oracle_output_records,
            "exact_cell_rows": exact_rows,
        },
        "hypotheses": hypotheses,
        "recommendation": {"choice": "B", "text": "STOP LOO AS SELECTOR, but retain adaptive depth/view as a candidate-generation axis."},
        "measured": ["Frozen score reproduction matched 3/41, 2/41, 2/41, 4/41.", f"Sentinel exact-cell ranks were {exact_rank_list}; Top4 selected none."],
        "inferred": ["Teacher-forced held-out-train competence can be decoupled from real-test extrapolation for this sampled Sentinel surface."],
        "not_established": ["A general population effect, a causal adapter mechanism, or a replacement selector."],
    }
    (output / "final_score_reproduction.json").write_text(json.dumps(report["score_reproduction"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "provenance.json").write_text(json.dumps(report["provenance"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "README.md").write_text("# Adaptive TTT Transfer30 post-mortem artifacts\n\nCPU-only analysis of byte-verified, frozen raw artifacts. `POSTMORTEM.md` is the human-readable conclusion; CSV files retain output- and fold-level measurements.\n", encoding="utf-8")
    (output / "POSTMORTEM.md").write_text(markdown(report), encoding="utf-8")
    (output / "REPORT.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    hashes = {path.name: sha256(path) for path in sorted(output.iterdir()) if path.is_file() and path.name != "SHA256SUMS.json"}
    (output / "SHA256SUMS.json").write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"event": "ADAPTIVE_TTT_TRANSFER30_POSTMORTEM_COMPLETE", "score_reproduction_pass": True, "recommendation": "B", "output": str(output)}, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-output", type=Path, required=True)
    parser.add_argument("--solutions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frozen-git-commit", default="e66da47a3403913c6d1e06a7cfb29a6d484e1faf")
    parser.add_argument("--scoring-git-commit", default="b94659258dbe55cbc66a820851cd90cd6ab95e41")
    parser.add_argument("--archive-sha256", default="84cdf9faced48ac98503f580c4355de234ced2b397c5d5155c72cf615e136ce4")
    main(parser.parse_args())
