"""Parallel CPU-only decoder-policy replay over frozen G1 Gold-prefix traces.

This script never imports a model runtime.  It replays only the stored native
ARC-token distributions observed along the Gold prefix.  Consequently, its
``LOCAL_BRANCHING_PROXY`` is a local retained-successor statistic, *not* a
DFS-node count, runtime prediction, or actual decoder evaluation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import multiprocessing as mp
import os
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable


# The fixed grid is declared here, before reading the frozen trace.
MAX_SCORE = -math.log(0.2)
AFFINE_TAU0 = (0.5, 1.0, MAX_SCORE, 2.0)
AFFINE_LAMBDA = (0.0025, 0.005, 0.01, 0.02, 0.04)
MEAN_NLL_THRESHOLDS = (0.01, 0.02, 0.03, 0.05, 0.08, 0.12)
REGRET_THRESHOLDS = (0.05, 0.10, 0.25, 0.50, 1.0, 2.0, 4.0)
WINDOW_SIZES = (8, 16, 32)
TOPK = (1, 2, 3, 4)
MARGIN_THRESHOLDS = (0.05, 0.10, 0.20, 0.40, 0.80)
EPSILON = 1e-12
# The finalist screen is fixed before reading results. It continues the prior
# counterfactual study's bounded-local-proxy convention, and deliberately
# excludes policies whose apparent Gold survival comes from obvious widening.
FINALIST_LOCAL_PROXY_CAP = 2.0


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in materialized for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(materialized)


def quantile(values: list[float], fraction: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    index = (len(values) - 1) * fraction
    lo, hi = math.floor(index), math.ceil(index)
    return values[lo] if lo == hi else values[lo] + (values[hi] - values[lo]) * (index - lo)


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def cell_id(row: dict[str, str]) -> tuple[str, int, int, str]:
    return row["task_id"], int(row["output_index"]), int(row["depth"]), row["view"]


def parse_dist(row: dict[str, str]) -> list[tuple[int, float]]:
    values = [(int(token), float(logprob)) for token, logprob in json.loads(row["legal_arc_token_logprobs_json"]).items()]
    if len(values) != 12 or not all(math.isfinite(logprob) for _token, logprob in values):
        raise RuntimeError(f"invalid legal-token distribution in {cell_id(row)}")
    return sorted(values, key=lambda item: (-item[1], item[0]))


def entropy(dist: list[tuple[int, float]]) -> float:
    probabilities = [math.exp(logprob) for _token, logprob in dist]
    total = sum(probabilities)
    if not total > 0:
        raise RuntimeError("non-positive probability mass")
    return -sum((p / total) * math.log(p / total) for p in probabilities if p > 0)


@dataclass(frozen=True)
class Policy:
    config_id: str
    family: str
    kind: str
    parameters: tuple[tuple[str, float | int | str], ...]

    def param(self, name: str) -> float | int | str:
        return dict(self.parameters)[name]


def policy(config_id: str, family: str, kind: str, **parameters: float | int | str) -> Policy:
    return Policy(config_id, family, kind, tuple(sorted(parameters.items())))


def base_policies() -> dict[str, list[Policy]]:
    tracks: dict[str, list[Policy]] = {
        "A": [policy("CURRENT_V5", "BASELINE", "v5")],
        "B": [], "C": [], "D": [], "E": [],
    }
    tracks["A"] += [policy(f"AFFINE_NLL_BUDGET_tau0={tau0:.3f}_lambda={lam:.4f}", "LENGTH_AWARE", "affine", tau0=tau0, lambda_=lam)
                    for tau0 in AFFINE_TAU0 for lam in AFFINE_LAMBDA]
    tracks["A"] += [policy(f"MEAN_NLL_tau={tau:.3f}", "LENGTH_AWARE", "mean_nll", tau=tau) for tau in MEAN_NLL_THRESHOLDS]
    tracks["B"] += [policy(f"CUMULATIVE_REGRET_r={r:.2f}", "RELATIVE_REGRET", "cumulative_regret", threshold=r) for r in REGRET_THRESHOLDS]
    tracks["B"] += [policy(f"WINDOWED_REGRET_w={window}_r={r:.2f}", "RELATIVE_REGRET", "windowed_regret", window=window, threshold=r)
                    for window in WINDOW_SIZES for r in REGRET_THRESHOLDS[:5]]
    tracks["C"] += [policy(f"TOPK_LOCAL_k={k}", "RANK_MARGIN", "topk", k=k) for k in TOPK]
    tracks["C"] += [policy(f"MARGIN_TRIGGERED_TOP2_delta={delta:.2f}", "RANK_MARGIN", "margin_top2", delta=delta) for delta in MARGIN_THRESHOLDS]
    tracks["D"] += [policy(f"V5_REGRET_RESERVE_r={r:.2f}", "HYBRID", "v5_regret_reserve", threshold=r) for r in (0.10, 0.25, 0.50, 1.0, 2.0)]
    tracks["D"] += [policy(f"V5_MARGIN_RESERVE_delta={delta:.2f}", "HYBRID", "v5_margin_reserve", delta=delta) for delta in MARGIN_THRESHOLDS]
    return tracks


def strict_set(dist: list[tuple[int, float]], prefix_nll: float) -> set[int]:
    return {token for token, logprob in dist if prefix_nll - logprob < MAX_SCORE}


def floor(kept: set[int], ranked: list[tuple[int, float]]) -> set[int]:
    return kept if kept else {ranked[0][0]}


def local_retained(policy_spec: Policy, ranked: list[tuple[int, float]], *, prefix_nll: float, prefix_regret: float,
                   regret_window: list[float], position: int) -> set[int]:
    by_token = dict(ranked)
    strict = strict_set(ranked, prefix_nll)
    top1 = ranked[0][1]
    second_gap = top1 - ranked[1][1]
    rank_by_token = {token: index + 1 for index, (token, _logprob) in enumerate(ranked)}
    kind = policy_spec.kind
    if kind == "v5":
        return floor(strict, ranked)
    if kind == "affine":
        budget = float(policy_spec.param("tau0")) + float(policy_spec.param("lambda_")) * (position + 1)
        return floor({token for token, logprob in ranked if prefix_nll - logprob <= budget + EPSILON}, ranked)
    if kind == "mean_nll":
        threshold = float(policy_spec.param("tau"))
        return floor({token for token, logprob in ranked if (prefix_nll - logprob) / (position + 1) <= threshold + EPSILON}, ranked)
    if kind == "cumulative_regret":
        threshold = float(policy_spec.param("threshold"))
        return floor({token for token, logprob in ranked if prefix_regret + top1 - logprob <= threshold + EPSILON}, ranked)
    if kind == "windowed_regret":
        threshold = float(policy_spec.param("threshold"))
        previous = sum(regret_window[-int(policy_spec.param("window")) + 1:])
        return floor({token for token, logprob in ranked if previous + top1 - logprob <= threshold + EPSILON}, ranked)
    if kind == "topk":
        return {token for token, _logprob in ranked[:int(policy_spec.param("k"))]}
    if kind == "margin_top2":
        return {token for token, _logprob in ranked[:2]} if second_gap <= float(policy_spec.param("delta")) + EPSILON else {ranked[0][0]}
    if kind == "v5_regret_reserve":
        threshold = float(policy_spec.param("threshold"))
        reserve = {token for token, logprob in ranked if rank_by_token[token] <= 2 and prefix_regret + top1 - logprob <= threshold + EPSILON}
        return floor(strict | reserve, ranked)
    if kind == "v5_margin_reserve":
        reserve = {token for token, _logprob in ranked[:2]} if second_gap <= float(policy_spec.param("delta")) + EPSILON else set()
        return floor(strict | reserve, ranked)
    if kind == "affine_rank2":
        budget = float(policy_spec.param("tau0")) + float(policy_spec.param("lambda_")) * (position + 1)
        primary = {token for token, logprob in ranked if prefix_nll - logprob <= budget + EPSILON}
        return floor(primary | {token for token, _logprob in ranked[:2]}, ranked)
    raise ValueError(f"unsupported policy kind {kind}")


def evaluate_cell(policy_spec: Policy, rows: list[dict[str, str]]) -> tuple[dict[str, Any], list[int]]:
    rows = sorted(rows, key=lambda row: int(row["token_position"]))
    if [int(row["token_position"]) for row in rows] != list(range(len(rows))):
        raise RuntimeError(f"non-contiguous positions for {cell_id(rows[0])}")
    if any(row["scoring_path"] != "INCREMENTAL_KV_REPLAY" for row in rows):
        raise RuntimeError(f"non-incremental scoring path for {cell_id(rows[0])}")
    prefix_nll = 0.0
    prefix_regret = 0.0
    regret_window: list[float] = []
    survives = True
    first_failure: int | None = None
    retained: list[int] = []
    backup_steps = 0
    entropies: list[float] = []
    for row in rows:
        ranked = parse_dist(row)
        gold = int(row["gold_token_id"])
        by_token = dict(ranked)
        if gold not in by_token:
            raise RuntimeError(f"Gold token missing from legal vocabulary for {cell_id(row)}")
        current = local_retained(policy("CURRENT_V5", "BASELINE", "v5"), ranked, prefix_nll=prefix_nll,
                                 prefix_regret=prefix_regret, regret_window=regret_window, position=int(row["token_position"]))
        kept = local_retained(policy_spec, ranked, prefix_nll=prefix_nll, prefix_regret=prefix_regret,
                              regret_window=regret_window, position=int(row["token_position"]))
        if policy_spec.kind == "v5" and (gold in kept) != as_bool(row["v5_local_gold_survives"]):
            raise RuntimeError(f"CURRENT_V5 mismatch {cell_id(row)} at {row['token_position']}")
        gold_here = gold in kept
        if survives and not gold_here:
            first_failure = int(row["token_position"])
        survives = survives and gold_here
        retained.append(len(kept))
        backup_steps += int(bool(kept - current))
        local_regret = ranked[0][1] - by_token[gold]
        prefix_regret += local_regret
        regret_window.append(local_regret)
        prefix_nll = float(row["cumulative_gold_nll"])
        entropies.append(entropy(ranked))
    anchor = rows[0]
    return ({
        "config_id": policy_spec.config_id, "family": policy_spec.family,
        "task_id": anchor["task_id"], "output_id": anchor["output_id"], "output_index": int(anchor["output_index"]),
        "depth": int(anchor["depth"]), "view": anchor["view"], "token_count": len(rows),
        "gold_path_survives": survives, "first_failure_position": "" if first_failure is None else first_failure,
        "first_failure_fraction": "" if first_failure is None else (first_failure + 1) / len(rows),
        "mean_entropy": mean(entropies), "backup_trigger_steps": backup_steps,
        "backup_trigger_fraction": backup_steps / len(rows),
    }, retained)


def group_trace(rows: list[dict[str, str]]) -> dict[tuple[str, int, int, str], list[dict[str, str]]]:
    grouped: dict[tuple[str, int, int, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[cell_id(row)].append(row)
    outputs = {row["output_id"] for row in rows}
    if len(grouped) != 336 or len(outputs) != 28:
        raise RuntimeError(f"expected frozen 336 cells / 28 outputs; got {len(grouped)} / {len(outputs)}")
    return grouped


def validate_summary(summary_rows: list[dict[str, str]], grouped: dict[tuple[str, int, int, str], list[dict[str, str]]]) -> None:
    """Bind the trace to the already-frozen 28-output G1-half inventory."""
    summary_ids = {row["output_id"] for row in summary_rows}
    trace_ids = {rows[0]["output_id"] for rows in grouped.values()}
    if len(summary_rows) != 28 or summary_ids != trace_ids:
        raise RuntimeError(
            f"G1-half output inventory mismatch: summary_rows={len(summary_rows)} "
            f"summary_outputs={len(summary_ids)} trace_outputs={len(trace_ids)}"
        )
    if any(int(row.get("cells", "12")) != 12 for row in summary_rows):
        raise RuntimeError("unexpected G1-half per-output cell inventory")


def eval_track(payload: tuple[list[Policy], dict[tuple[str, int, int, str], list[dict[str, str]]]]) -> list[tuple[dict[str, Any], list[int]]]:
    policies, grouped = payload
    result: list[tuple[dict[str, Any], list[int]]] = []
    for policy_spec in policies:
        for key in sorted(grouped):
            result.append(evaluate_cell(policy_spec, grouped[key]))
    return result


def config_summary(policy_spec: Policy, cells: list[dict[str, Any]], retained: list[int], baseline_mean: float) -> dict[str, Any]:
    by_output: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in cells:
        by_output[str(row["output_id"])].append(row)
    output_any = sum(any(row["gold_path_survives"] for row in output_cells) for output_cells in by_output.values())
    failed_fractions = [float(row["first_failure_fraction"]) for row in cells if row["first_failure_fraction"] != ""]
    survivors_by_depth_view = Counter(f"d{row['depth']}:{row['view']}" for row in cells if row["gold_path_survives"])
    return {
        "config_id": policy_spec.config_id, "family": policy_spec.family, "kind": policy_spec.kind,
        "parameters_json": json.dumps(dict(policy_spec.parameters), sort_keys=True, separators=(",", ":")),
        "CELL_GOLD_SURVIVAL": sum(bool(row["gold_path_survives"]) for row in cells),
        "OUTPUT_ANY_GOLD_SURVIVAL": output_any,
        "NEW_SEARCHABLE_OUTPUTS_VS_V5": output_any,  # baseline is filled after all summaries are known
        "median_first_failure_fraction": median(failed_fractions) if failed_fractions else 1.0,
        "LOCAL_BRANCHING_PROXY_MEAN": mean(retained), "LOCAL_BRANCHING_PROXY_MEDIAN": median(retained),
        "LOCAL_BRANCHING_PROXY_P90": quantile([float(v) for v in retained], 0.90),
        "LOCAL_BRANCHING_PROXY_P99": quantile([float(v) for v in retained], 0.99),
        "LOCAL_BRANCHING_PROXY_MAX": max(retained), "BACKUP_TRIGGER_RATE": mean(float(row["backup_trigger_fraction"]) for row in cells),
        "gold_survival_by_depth_view_json": json.dumps(dict(sorted(survivors_by_depth_view.items())), separators=(",", ":")),
        "COST_LIMITATION": "LOCAL_GOLD_PREFIX_CONDITIONAL_PROXY_NOT_TRUE_DFS_COST",
    }


def pareto_flags(rows: list[dict[str, Any]]) -> dict[str, bool]:
    flags: dict[str, bool] = {}
    for row in rows:
        ours = int(row["OUTPUT_ANY_GOLD_SURVIVAL"]), float(row["LOCAL_BRANCHING_PROXY_MEAN"])
        flags[str(row["config_id"])] = not any(
            int(other["OUTPUT_ANY_GOLD_SURVIVAL"]) >= ours[0]
            and float(other["LOCAL_BRANCHING_PROXY_MEAN"]) <= ours[1]
            and (int(other["OUTPUT_ANY_GOLD_SURVIVAL"]) > ours[0] or float(other["LOCAL_BRANCHING_PROXY_MEAN"]) < ours[1] - EPSILON)
            for other in rows
        )
    return flags


def fold_index(output_id: str) -> int:
    return int(hashlib.sha256(output_id.encode("utf-8")).hexdigest(), 16) % 4


def build_folds(policy_map: dict[str, Policy], cell_rows: list[dict[str, Any]], retained_by_config: dict[str, list[int]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    by_config_output: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in cell_rows:
        by_config_output[str(row["config_id"])][str(row["output_id"])].append(row)
    records: list[dict[str, Any]] = []
    robust_counts: Counter[str] = Counter()
    for fold in range(4):
        fold_rows: list[dict[str, Any]] = []
        for config_id, policy_spec in sorted(policy_map.items()):
            output_cells = by_config_output[config_id]
            selected_outputs = [output_id for output_id in output_cells if fold_index(output_id) == fold]
            selected_cells = [row for output_id in selected_outputs for row in output_cells[output_id]]
            # The local proxy is calculated from exactly the held-out fold's retained cells.
            proxy = mean([]) if False else mean([])  # never reached; retained is recomputed below
            # Reconstruct retained count is unavailable in persisted cells, so the deterministic
            # full policy is replayed separately in `run_study` for each fold.
            fold_rows.append({
                "config_id": config_id, "family": policy_spec.family, "fold": fold,
                "heldout_outputs": len(selected_outputs),
                "heldout_output_gold_survival": sum(any(cell["gold_path_survives"] for cell in output_cells[output_id]) for output_id in selected_outputs),
                "heldout_cell_gold_survival": sum(bool(cell["gold_path_survives"]) for cell in selected_cells),
            })
        records.extend(fold_rows)
    # Local proxy values are injected by run_study where original retained counts are available.
    return records, robust_counts


def taxonomy(rows: list[dict[str, str]], grouped: dict[tuple[str, int, int, str], list[dict[str, str]]]) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    v5 = policy("CURRENT_V5", "BASELINE", "v5")
    for key in sorted(grouped):
        trace = sorted(grouped[key], key=lambda row: int(row["token_position"]))
        result, _retained = evaluate_cell(v5, trace)
        if result["gold_path_survives"]:
            failure_type = "CURRENT_V5_SURVIVES"
            metrics: dict[str, Any] = {}
        else:
            position = int(result["first_failure_position"])
            first = trace[position]
            ranked = parse_dist(first)
            by_token = dict(ranked)
            gold = int(first["gold_token_id"])
            nlls = [-float(row["gold_token_logprob"]) for row in trace]
            regrets = [parse_dist(row)[0][1] - dict(parse_dist(row))[int(row["gold_token_id"])] for row in trace]
            total_nll = sum(nlls)
            concentration = {f"worst_{k}_nll_fraction": sum(sorted(nlls, reverse=True)[:k]) / total_nll if total_nll else 0.0 for k in (1, 3, 5)}
            first_rank = next(index + 1 for index, (token, _lp) in enumerate(ranked) if token == gold)
            # Ordered rules: rank > 2 demonstrates a genuine rank exclusion; otherwise
            # a token holding >=50% of total NLL is a spike; rank-1 cumulative failures
            # with no dominant token are gradual drift. All residual cases are mixed.
            if first_rank > 2:
                failure_type = "RANK_FAILURE"
            elif concentration["worst_1_nll_fraction"] >= 0.50:
                failure_type = "FEW_TOKEN_SPIKE"
            elif first_rank == 1 and concentration["worst_1_nll_fraction"] < 0.25:
                failure_type = "GRADUAL_DRIFT"
            else:
                failure_type = "MIXED"
            metrics = {
                "first_prune_rank": first_rank,
                "cumulative_nll_at_prune": float(first["cumulative_gold_nll"]),
                "cumulative_regret_at_prune": sum(regrets[:position + 1]),
                "max_single_gold_nll": max(nlls), "max_single_regret": max(regrets),
                "gold_rank_gt_1_count": sum(rank > 1 for rank in [int(row["gold_token_rank_legal_arc_vocab"]) for row in trace]),
                "gold_rank_gt_2_count": sum(rank > 2 for rank in [int(row["gold_token_rank_legal_arc_vocab"]) for row in trace]),
                **concentration,
            }
        cells.append({
            "task_id": key[0], "output_index": key[1], "depth": key[2], "view": key[3],
            "output_id": trace[0]["output_id"], "sequence_length": len(trace),
            "first_prune_position": result["first_failure_position"], "first_prune_fraction": result["first_failure_fraction"],
            "failure_taxonomy": failure_type, **metrics,
        })
    return cells


def rank_family(rows: list[dict[str, Any]], family: str) -> dict[str, Any] | None:
    candidates = [
        row for row in rows
        if row["family"] == family
        and int(row["NEW_SEARCHABLE_OUTPUTS_VS_V5"]) > 0
        and bool(row["ROBUST_PARETO"])
        and float(row["LOCAL_BRANCHING_PROXY_RATIO"]) <= FINALIST_LOCAL_PROXY_CAP + EPSILON
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda row: (-int(row["ROBUST_PARETO_FOLDS"]), -int(row["OUTPUT_ANY_GOLD_SURVIVAL"]), float(row["LOCAL_BRANCHING_PROXY_RATIO"]), str(row["config_id"])))
    return candidates[0]


def run_study(trace_rows: list[dict[str, str]], summary_rows: list[dict[str, str]]) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    grouped = group_trace(trace_rows)
    validate_summary(summary_rows, grouped)
    tracks = base_policies()
    # Tracks A, B and C have no data dependency and replay in separate processes.
    parallel = [(tracks[name], grouped) for name in ("A", "B", "C")]
    workers = min(10, max(1, (os.cpu_count() or 1) - 2), len(parallel))
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as executor:
        chunks = list(executor.map(eval_track, parallel))
    raw_results = [pair for chunk in chunks for pair in chunk]
    policy_map = {policy_spec.config_id: policy_spec for values in tracks.values() for policy_spec in values}
    by_config_cells: dict[str, list[dict[str, Any]]] = defaultdict(list)
    retained_by_config: dict[str, list[int]] = defaultdict(list)
    for cell, retained in raw_results:
        by_config_cells[str(cell["config_id"])].append(cell)
        retained_by_config[str(cell["config_id"])].extend(retained)
    summaries = [config_summary(policy_map[config_id], cells, retained_by_config[config_id], 1.0) for config_id, cells in sorted(by_config_cells.items())]
    baseline = next(row for row in summaries if row["config_id"] == "CURRENT_V5")
    baseline_outputs = int(baseline["OUTPUT_ANY_GOLD_SURVIVAL"])
    baseline_proxy = float(baseline["LOCAL_BRANCHING_PROXY_MEAN"])
    for row in summaries:
        row["NEW_SEARCHABLE_OUTPUTS_VS_V5"] = int(row["OUTPUT_ANY_GOLD_SURVIVAL"]) - baseline_outputs
        row["LOCAL_BRANCHING_PROXY_RATIO"] = float(row["LOCAL_BRANCHING_PROXY_MEAN"]) / baseline_proxy

    # Pick at most three Pareto-relevant affine primary budgets deterministically,
    # then evaluate the specified affine+rank2 hybrids after Track A is frozen.
    affine = [row for row in summaries if row["kind"] == "affine"]
    affine_front = pareto_flags(affine)
    selected_affine = [row for row in affine if affine_front[str(row["config_id"])]]
    selected_affine.sort(key=lambda row: (-int(row["OUTPUT_ANY_GOLD_SURVIVAL"]), float(row["LOCAL_BRANCHING_PROXY_MEAN"]), str(row["config_id"])))
    for row in selected_affine[:3]:
        params = json.loads(str(row["parameters_json"]))
        spec = policy(f"AFFINE_RANK2_tau0={float(params['tau0']):.3f}_lambda={float(params['lambda_']):.4f}", "HYBRID", "affine_rank2", tau0=float(params["tau0"]), lambda_=float(params["lambda_"]))
        tracks["D"].append(spec)
        policy_map[spec.config_id] = spec
    # D's dependency is now resolved; its own policies replay in parallel.
    d_results = eval_track((tracks["D"], grouped))
    for cell, retained in d_results:
        by_config_cells[str(cell["config_id"])].append(cell)
        retained_by_config[str(cell["config_id"])].extend(retained)
    d_summaries = [config_summary(policy_map[config_id], cells, retained_by_config[config_id], baseline_proxy)
                   for config_id, cells in sorted(by_config_cells.items()) if config_id not in {row["config_id"] for row in summaries}]
    summaries += d_summaries
    for row in summaries:
        row["NEW_SEARCHABLE_OUTPUTS_VS_V5"] = int(row["OUTPUT_ANY_GOLD_SURVIVAL"]) - baseline_outputs
        row["LOCAL_BRANCHING_PROXY_RATIO"] = float(row["LOCAL_BRANCHING_PROXY_MEAN"]) / baseline_proxy

    # Fold metrics are computed from the same frozen cells and local retained counts.
    fold_rows: list[dict[str, Any]] = []
    robust_counts: Counter[str] = Counter()
    summary_by_id = {str(row["config_id"]): row for row in summaries}
    for fold in range(4):
        fold_comp: list[dict[str, Any]] = []
        for config_id, cells in sorted(by_config_cells.items()):
            selected_cells = [row for row in cells if fold_index(str(row["output_id"])) == fold]
            outputs = {str(row["output_id"]) for row in selected_cells}
            records_by_output: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for cell in selected_cells:
                records_by_output[str(cell["output_id"])].append(cell)
            # The exact local retained stream can be identified by replaying cells in deterministic order.
            retained_values: list[int] = []
            for key in sorted(grouped):
                if fold_index(grouped[key][0]["output_id"]) == fold:
                    _cell, values = evaluate_cell(policy_map[config_id], grouped[key])
                    retained_values.extend(values)
            candidate = {
                "config_id": config_id, "family": summary_by_id[config_id]["family"], "fold": fold,
                "heldout_outputs": len(outputs),
                "heldout_output_gold_survival": sum(any(cell["gold_path_survives"] for cell in values) for values in records_by_output.values()),
                "heldout_cell_gold_survival": sum(bool(cell["gold_path_survives"]) for cell in selected_cells),
                "heldout_LOCAL_BRANCHING_PROXY": mean(retained_values),
            }
            fold_comp.append(candidate)
        flags = pareto_flags([{**row, "OUTPUT_ANY_GOLD_SURVIVAL": row["heldout_output_gold_survival"], "LOCAL_BRANCHING_PROXY_MEAN": row["heldout_LOCAL_BRANCHING_PROXY"]} for row in fold_comp])
        for row in fold_comp:
            row["pareto_optimal"] = flags[row["config_id"]]
            # Fixed near-Pareto rule: within one surviving output of a frontier
            # point whose proxy is no more than 5% lower.
            row["near_pareto"] = any(
                int(other["heldout_output_gold_survival"]) >= int(row["heldout_output_gold_survival"]) - 1
                and float(other["heldout_LOCAL_BRANCHING_PROXY"]) <= float(row["heldout_LOCAL_BRANCHING_PROXY"]) * 1.05 + EPSILON
                for other in fold_comp if flags[other["config_id"]]
            )
            robust_counts[row["config_id"]] += int(bool(row["pareto_optimal"]) or bool(row["near_pareto"]))
        fold_rows.extend(fold_comp)
    for row in summaries:
        row["ROBUST_PARETO_FOLDS"] = robust_counts[str(row["config_id"])]
        row["ROBUST_PARETO"] = robust_counts[str(row["config_id"])] >= 3

    global_pareto = pareto_flags(summaries)
    for row in summaries:
        row["GLOBAL_PARETO"] = global_pareto[str(row["config_id"])]

    taxonomy_rows = taxonomy(trace_rows, grouped)
    failure_counts = Counter(row["failure_taxonomy"] for row in taxonomy_rows)
    family_best = {family: rank_family(summaries, family) for family in ("LENGTH_AWARE", "RELATIVE_REGRET", "RANK_MARGIN", "HYBRID")}
    finalists = []
    # Different mechanisms first.  The rank policy is retained as the third
    # finalist because it is the bounded direct-insurance control; the best
    # hybrid reserve is reported but not promoted if it is dominated by it.
    for family, label in (("RELATIVE_REGRET", "FINALIST_A"), ("LENGTH_AWARE", "FINALIST_B"), ("RANK_MARGIN", "FINALIST_C")):
        best = family_best[family]
        finalists.append({"slot": label, "config_id": best["config_id"] if best and best["ROBUST_PARETO"] else "NONE",
                          "family": family, "reason": "highest robust fixed-grid survival with bounded local proxy" if best and best["ROBUST_PARETO"] else "no positive robust fixed-grid candidate"})
    promoted = [row for row in finalists if row["config_id"] != "NONE"]
    max_gain = max((int(row["NEW_SEARCHABLE_OUTPUTS_VS_V5"]) for row in summaries), default=0)
    hypothesis = "STRONGLY_SUPPORTED" if max_gain >= 14 and promoted else "SUPPORTED" if max_gain >= 7 and promoted else "PARTIALLY_SUPPORTED" if max_gain else "NOT_SUPPORTED"
    decision = {
        "CPU_AUDIT_STATUS": "PASS", "TRACE_CELLS": "336/336", "OUTPUTS": 28,
        "CURRENT_V5_SEARCHABLE": f"{baseline_outputs}/28", "CURRENT_UNION": "33/89",
        "failure_taxonomy_cell_counts": dict(sorted(failure_counts.items())),
        "MOST_SUPPORTED_FAILURE_MECHANISM": max(failure_counts, key=failure_counts.get) if failure_counts else "UNKNOWN",
        "FINALISTS": finalists, "DECODER_PRUNING_HYPOTHESIS": hypothesis,
        "NEXT": "D1_SMALL_REAL_GPU_AB" if promoted else "CAPACITY_CONTROL",
        "GPU_USED": "NO", "MODEL_LOADED": "NO", "DFS_RUN": "NO",
        "LOCAL_BRANCHING_PROXY_LIMITATION": "Gold-prefix conditional retained-successor counts only; not actual DFS nodes, runtime multiplier, or true search complexity.",
        "FOLD_RULE": "sha256(output_id) mod 4; robust Pareto means Pareto-optimal or fixed near-Pareto in >=3/4 folds.",
        "FINALIST_LOCAL_PROXY_CAP": FINALIST_LOCAL_PROXY_CAP,
        "FAMILY_BEST_BOUNDED": {family: (family_best[family]["config_id"] if family_best[family] else "NONE") for family in family_best},
    }
    # Track rows are a strict deterministic split of all summaries.
    output = {
        "A": [row for row in summaries if row["config_id"] == "CURRENT_V5" or row["family"] == "LENGTH_AWARE"],
        "B": [row for row in summaries if row["family"] == "RELATIVE_REGRET"],
        "C": [row for row in summaries if row["family"] == "RANK_MARGIN"],
        "D": [row for row in summaries if row["family"] == "HYBRID"],
        "E": taxonomy_rows, "all": sorted(summaries, key=lambda row: str(row["config_id"])),
        "folds": fold_rows, "finalists": finalists,
    }
    return decision, output


def schema_audit(rows: list[dict[str, str]]) -> dict[str, Any]:
    columns = set(rows[0]) if rows else set()
    required = {"gold_token_logprob", "cumulative_gold_nll", "gold_token_rank_legal_arc_vocab", "legal_arc_token_logprobs_json", "depth", "view", "token_position"}
    missing = sorted(required - columns)
    if missing:
        raise RuntimeError(f"missing required frozen trace fields: {missing}")
    return {
        "trace_rows": len(rows), "trace_cells": len(group_trace(rows)), "trace_outputs": len({row["output_id"] for row in rows}),
        "fields": sorted(columns),
        "gold_token_logprob": "AVAILABLE", "cumulative_gold_nll": "AVAILABLE", "legal_token_rank": "AVAILABLE",
        "legal_token_logprob_vector": "AVAILABLE", "top1_top2_columns": "AVAILABLE" if {"top1_logprob", "top2_logprob"} <= columns else "NOT_AVAILABLE",
        "entropy": "DERIVABLE_FROM_FROZEN_LEGAL_TOKEN_VECTOR", "sequence_length": "DERIVABLE_FROM_TOKEN_POSITIONS",
        "first_prune_position": "DERIVABLE_FROM_FROZEN_CURRENT_V5_LOCAL_SURVIVAL", "depth": "AVAILABLE", "view": "AVAILABLE",
        "scoring_path": sorted(set(row["scoring_path"] for row in rows)),
    }


def markdown(decision: dict[str, Any], outputs: dict[str, list[dict[str, Any]]]) -> str:
    lines = ["# D0 parallel decoder-policy study", "", "CPU-only replay over frozen G1 incremental-KV Gold-prefix traces.", "",
             "**Limit:** `LOCAL_BRANCHING_PROXY` is a Gold-prefix conditional statistic. It is not actual DFS nodes, runtime, or true search complexity.", "",
             "## Family bests", "", "| Family | Config | Outputs | Local proxy ratio | Robust folds |", "|---|---|---:|---:|---:|"]
    for family in ("LENGTH_AWARE", "RELATIVE_REGRET", "RANK_MARGIN", "HYBRID"):
        bounded_id = decision["FAMILY_BEST_BOUNDED"].get(family, "NONE")
        row = next((entry for entry in outputs["all"] if entry["config_id"] == bounded_id), None)
        if row:
            lines.append(f"| {family} | {row['config_id']} | {row['OUTPUT_ANY_GOLD_SURVIVAL']} | {row['LOCAL_BRANCHING_PROXY_RATIO']:.3f} | {row['ROBUST_PARETO_FOLDS']}/4 |")
        else:
            lines.append(f"| {family} | NONE within {decision['FINALIST_LOCAL_PROXY_CAP']:.1f}x cap | — | — | — |")
    lines += ["", "## Finalists", ""]
    for finalist in outputs["finalists"]:
        lines.append(f"- {finalist['slot']}: {finalist['config_id']} ({finalist['reason']})")
    lines += ["", "## Interpretation", "", f"- Hypothesis: {decision['DECODER_PRUNING_HYPOTHESIS']}", f"- Next: {decision['NEXT']}", "- No candidate generation, TTT, DFS, model loading, or GPU work was run."]
    return "\n".join(lines) + "\n"


def pilot_plan(decision: dict[str, Any], outputs: dict[str, list[dict[str, Any]]]) -> str:
    active = [row["config_id"] for row in outputs["finalists"] if row["config_id"] != "NONE"]
    return "\n".join([
        "# D1 GPU pilot plan (not executed)", "", "Use a fixed 8-12 output cohort selected before GPU work; do not select from these Gold-survival results.",
        "Compare V5 baseline with: " + (", ".join(active) if active else "no promoted decoder policy"), "",
        "Measure actual exact outputs, candidate recall, expanded nodes, runtime, seconds/output, GPU-hours, peak VRAM, and new exact solves/GPU-hour.",
        "The present CPU replay is insufficient to estimate any of those actual decoder quantities.", "",
        "No GPU test has been launched by this plan.",
    ]) + "\n"


def main() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-csv", type=Path, required=True)
    parser.add_argument("--g1-summary-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    trace_rows = read_csv(args.trace_csv)
    audit = schema_audit(trace_rows)
    summary_rows = read_csv(args.g1_summary_csv)
    decision, output = run_study(trace_rows, summary_rows)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "TRACE_SCHEMA_AUDIT.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(out / "track_a_length_aware.csv", output["A"])
    write_csv(out / "track_b_relative_regret.csv", output["B"])
    write_csv(out / "track_c_rank_margin.csv", output["C"])
    write_csv(out / "track_d_hybrids.csv", output["D"])
    write_csv(out / "track_e_failure_taxonomy.csv", output["E"])
    write_csv(out / "all_decoder_configs.csv", output["all"])
    write_csv(out / "fold_robustness.csv", output["folds"])
    frontier = [row for row in output["all"] if bool(row["GLOBAL_PARETO"])]
    write_csv(out / "pareto_frontier.csv", frontier)
    write_csv(out / "finalists.csv", output["finalists"])
    (out / "DECODER_PARALLEL_STUDY.md").write_text(markdown(decision, output), encoding="utf-8")
    (out / "D1_GPU_PILOT_PLAN.md").write_text(pilot_plan(decision, output), encoding="utf-8")
    (out / "DECISION.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    provenance = {"experiment": "D0_PARALLEL_DECODER_POLICY_SWEEP_V1", "source_trace": str(args.trace_csv), "source_g1_summary": str(args.g1_summary_csv),
                  "trace_schema": audit, "frozen_evidence": {"G1_HALF": "5e3f6ca253631b446971da9ef0e6fddc313b345a", "G2": "5253dd868ad7c536236184f92e175fd5c570e103", "G3A": "a26a94d4327cd8a00e2110ef6c85fb178560bcc7"},
                  "current_development_union": "33/89", "gpu_used": False, "model_loaded": False, "dfs_run": False}
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
