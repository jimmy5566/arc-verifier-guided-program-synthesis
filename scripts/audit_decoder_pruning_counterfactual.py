"""CPU-only counterfactual pruning audit over frozen G1 incremental-KV traces.

This intentionally evaluates only whether the *frozen Gold path* would remain
reachable under fixed decoder policies.  It never invokes a model or decoder,
and its retained-successor counts are local, Gold-prefix conditional cost
proxies rather than reconstructed DFS-tree costs.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable


MAX_SCORE = -math.log(0.2)
TOPK_POLICIES = ("CURRENT_V5", "TOPK_INSURANCE_1", "TOPK_INSURANCE_2")
RELATIVE_THRESHOLDS = (0.5, 1.0, 2.0, 4.0)
LOCAL_COST_CAP = 2.0


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


def percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("empty percentile input")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def bool_value(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def cell_key(row: dict[str, str]) -> tuple[str, int, int, str]:
    return (str(row["task_id"]), int(row["output_index"]), int(row["depth"]), str(row["view"]))


def parse_distribution(row: dict[str, str]) -> list[tuple[int, float]]:
    raw = json.loads(row["legal_arc_token_logprobs_json"])
    values = [(int(token), float(logprob)) for token, logprob in raw.items()]
    if len(values) != 12 or not all(math.isfinite(logprob) for _token, logprob in values):
        raise RuntimeError(f"invalid native-token distribution for {cell_key(row)}")
    return sorted(values, key=lambda item: (-item[1], item[0]))


def strict_tokens(distribution: list[tuple[int, float]], prefix_nll: float, max_score: float) -> list[int]:
    return [token for token, logprob in distribution if prefix_nll - logprob < max_score]


def policy_ids() -> tuple[str, ...]:
    return (*TOPK_POLICIES, *(f"RELREGRET_{threshold:.1f}" for threshold in RELATIVE_THRESHOLDS))


def evaluate_cell(rows: list[dict[str, str]], policy: str, max_score: float) -> dict[str, Any]:
    """Evaluate one frozen Gold continuation under a fixed, global policy."""
    rows = sorted(rows, key=lambda row: int(row["token_position"]))
    expected_positions = list(range(len(rows)))
    if [int(row["token_position"]) for row in rows] != expected_positions:
        raise RuntimeError(f"noncontiguous token positions for {cell_key(rows[0])}")
    if any(str(row["scoring_path"]) != "INCREMENTAL_KV_REPLAY" for row in rows):
        raise RuntimeError(f"non-incremental trace for {cell_key(rows[0])}")

    survives = True
    first_block: int | None = None
    retained_counts: list[int] = []
    prefix_nll = 0.0
    prefix_regret = 0.0
    for row in rows:
        dist = parse_distribution(row)
        gold = int(row["gold_token_id"])
        by_token = dict(dist)
        if gold not in by_token:
            raise RuntimeError(f"Gold token absent from native vocabulary for {cell_key(row)}")
        ranked = [token for token, _logprob in dist]
        best_logprob = dist[0][1]
        strict = strict_tokens(dist, prefix_nll, max_score)
        strict_gold = gold in strict

        if policy == "CURRENT_V5":
            kept = strict if strict else [ranked[0]]
            gold_here = gold in kept
            if gold_here != bool_value(row["v5_local_gold_survives"]):
                raise RuntimeError(f"CURRENT_V5 replay mismatch for {cell_key(row)} at {row['token_position']}")
        elif policy.startswith("TOPK_INSURANCE_"):
            top_k = int(policy.rsplit("_", 1)[1])
            kept = sorted(set(strict).union(ranked[:top_k]), key=lambda token: (-(by_token[token]), token))
            gold_here = gold in kept
        elif policy.startswith("RELREGRET_"):
            threshold = float(policy.removeprefix("RELREGRET_"))
            candidate_regret = prefix_regret + best_logprob - by_token[gold]
            kept = [
                token for token, logprob in dist
                if prefix_regret + best_logprob - logprob < threshold
            ]
            if not kept:
                kept = [ranked[0]]  # retain the V5 floor convention locally
            gold_here = gold in kept
            prefix_regret = candidate_regret
        else:
            raise ValueError(f"unknown policy: {policy}")

        retained_counts.append(len(kept))
        if survives and not gold_here:
            first_block = int(row["token_position"])
        survives = survives and gold_here
        prefix_nll = float(row["cumulative_gold_nll"])

    anchor = rows[0]
    return {
        "policy": policy,
        "task_id": anchor["task_id"],
        "output_id": anchor["output_id"],
        "output_index": int(anchor["output_index"]),
        "depth": int(anchor["depth"]),
        "view": anchor["view"],
        "token_count": len(rows),
        "gold_path_survives": survives,
        "first_block_position": "" if first_block is None else first_block,
        "estimated_retained_branches_per_step_mean": mean(retained_counts),
        "estimated_retained_branches_per_step_median": median(retained_counts),
        "estimated_retained_branches_per_step_p90": percentile([float(value) for value in retained_counts], 0.9),
        "estimated_retained_branches_per_step_max": max(retained_counts),
    }


def _baseline_membership(summary_rows: list[dict[str, str]]) -> set[str]:
    if len(summary_rows) != 28:
        raise RuntimeError(f"expected 28 frozen G1 output summaries, got {len(summary_rows)}")
    return {str(row["output_id"]) for row in summary_rows}


def audit(trace_rows: list[dict[str, str]], summary_rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[tuple[str, int, int, str], list[dict[str, str]]] = defaultdict(list)
    for row in trace_rows:
        grouped[cell_key(row)].append(row)
    expected_outputs = _baseline_membership(summary_rows)
    trace_outputs = {row["output_id"] for row in trace_rows}
    if trace_outputs != expected_outputs or len(grouped) != 336:
        raise RuntimeError(f"frozen G1 trace inventory mismatch cells={len(grouped)} outputs={len(trace_outputs)}")

    cell_rows: list[dict[str, Any]] = []
    for policy in policy_ids():
        for key in sorted(grouped):
            cell_rows.append(evaluate_cell(grouped[key], policy, MAX_SCORE))

    current = [row for row in cell_rows if row["policy"] == "CURRENT_V5"]
    current_cost = mean(float(row["estimated_retained_branches_per_step_mean"]) for row in current)
    output_rows: list[dict[str, Any]] = []
    cost_rows: list[dict[str, Any]] = []
    policy_output_counts: dict[str, int] = {}
    for policy in policy_ids():
        selected = [row for row in cell_rows if row["policy"] == policy]
        by_output: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in selected:
            by_output[str(row["output_id"])].append(row)
        searchable = 0
        for output_id in sorted(expected_outputs):
            cells = by_output[output_id]
            surviving = [row for row in cells if bool(row["gold_path_survives"])]
            if surviving:
                searchable += 1
            output_rows.append({
                "policy": policy, "output_id": output_id,
                "gold_surviving_cells": len(surviving), "gold_surviving_depth_view": ";".join(
                    f"d{row['depth']}:{row['view']}" for row in surviving
                ), "output_searchable": bool(surviving),
            })
        policy_output_counts[policy] = searchable
        retained = [float(row["estimated_retained_branches_per_step_mean"]) for row in selected]
        step_values: list[float] = []
        for row in selected:
            step_values.extend([float(row["estimated_retained_branches_per_step_median"]), float(row["estimated_retained_branches_per_step_p90"])])
        cost_rows.append({
            "policy": policy,
            "gold_surviving_cells": sum(bool(row["gold_path_survives"]) for row in selected),
            "searchable_outputs": searchable,
            "new_searchable_outputs_vs_current": searchable - policy_output_counts.get("CURRENT_V5", 0),
            "estimated_mean_retained_branches_per_step": mean(retained),
            "median_cell_retained_branches_per_step": median(retained),
            "p90_cell_retained_branches_per_step": percentile(retained, 0.9),
            "estimated_local_expansion_multiplier_vs_current": mean(retained) / current_cost,
            "cost_within_predeclared_2x_local_cap": mean(retained) / current_cost <= LOCAL_COST_CAP,
            "cost_proxy_scope": "LOCAL_GOLD_PREFIX_CONDITIONAL_NOT_RECONSTRUCTED_DFS_TREE",
        })

    candidates = [
        row for row in cost_rows
        if row["policy"] != "CURRENT_V5"
        and bool(row["cost_within_predeclared_2x_local_cap"])
        and int(row["new_searchable_outputs_vs_current"]) > 0
    ]
    candidates.sort(key=lambda row: (-int(row["new_searchable_outputs_vs_current"]), float(row["estimated_local_expansion_multiplier_vs_current"]), str(row["policy"])))
    selected = candidates[:2]
    best = [str(row["policy"]) for row in selected]
    best_outputs = int(selected[0]["new_searchable_outputs_vs_current"]) if selected else 0
    hypothesis = "SUPPORTED" if best_outputs >= 7 else "PARTIALLY_SUPPORTED" if best_outputs else "NOT_SUPPORTED"
    decision = {
        "CPU_AUDIT_STATUS": "PASS",
        "TRACE_CELLS": len(grouped), "TRACE_OUTPUTS": len(trace_outputs),
        "CURRENT_V5_SEARCHABLE_OUTPUTS": policy_output_counts["CURRENT_V5"],
        **{f"{policy}_SEARCHABLE_OUTPUTS": policy_output_counts[policy] for policy in policy_ids() if policy != "CURRENT_V5"},
        "BEST_POLICY_1": best[0] if best else "NONE",
        "BEST_POLICY_2": best[1] if len(best) > 1 else "NONE",
        "ESTIMATED_COST_MULTIPLIER": selected[0]["estimated_local_expansion_multiplier_vs_current"] if selected else None,
        "DECODER_PRUNING_HYPOTHESIS": hypothesis,
        "NEXT": "SMALL_REAL_DFS_AB_TEST" if best else "CAPACITY_CONTROL",
        "GPU_USED": "NO",
        "CURRENT_UNION": "33/89",
        "COST_PROXY_LIMITATION": "Local retained-successor counts on frozen Gold-prefix states; actual DFS-tree expansion is not established.",
        "SELECTION_RULE": f"fixed policy grid; select up to two policies with positive new searchable outputs and <= {LOCAL_COST_CAP}x local cost proxy",
    }
    return cell_rows, output_rows, cost_rows, decision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-csv", type=Path, required=True)
    parser.add_argument("--g1-summary-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    cells, outputs, costs, decision = audit(read_csv(args.trace_csv), read_csv(args.g1_summary_csv))
    write_csv(args.output_dir / "pruning_policy_cells.csv", cells)
    write_csv(args.output_dir / "pruning_policy_outputs.csv", outputs)
    write_csv(args.output_dir / "pruning_cost_summary.csv", costs)
    (args.output_dir / "DECISION.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# Decoder pruning counterfactual audit", "", "CPU-only diagnostic over frozen G1 incremental-KV Gold paths.", "",
             "Gold survival is not an actual solve. Cost is a local Gold-prefix conditional proxy, not a reconstructed DFS tree.", "",
             "## Policy summary", "", "| Policy | Searchable outputs | Gold-surviving cells | Local multiplier | Under 2x cap |", "|---|---:|---:|---:|---|"]
    for row in costs:
        lines.append(f"| {row['policy']} | {row['searchable_outputs']} | {row['gold_surviving_cells']} | {row['estimated_local_expansion_multiplier_vs_current']:.3f} | {row['cost_within_predeclared_2x_local_cap']} |")
    lines += ["", "## Decision", "", f"- Hypothesis: {decision['DECODER_PRUNING_HYPOTHESIS']}", f"- Best policy 1: {decision['BEST_POLICY_1']}", f"- Best policy 2: {decision['BEST_POLICY_2']}", f"- Next: {decision['NEXT']}", "", f"- Limitation: {decision['COST_PROXY_LIMITATION']}"]
    (args.output_dir / "DECODER_PRUNING_COUNTERFACTUAL.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
