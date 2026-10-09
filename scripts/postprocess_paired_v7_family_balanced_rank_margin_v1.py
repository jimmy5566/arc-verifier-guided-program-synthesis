#!/usr/bin/env python3
"""Target-blind CPU postprocessor for paired V7/Family-Balanced rank evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.paired_v7_family_balanced_rank_margin_contract import adjusted_interval, family_stratified_bootstrap_deltas, paired_margin_sensitivity_u, sensitivity_gate

CONDITIONS = ("RECONSTRUCTED_FOUNDATION_V2_V7", "FAMILY_BALANCED")
COMPONENTS = ("GRID_CONTENT", "ROW_SEPARATOR", "EOS_END")


def quantile(values: list[float], probability: float) -> float:
    if not values or not 0 <= probability <= 1:
        raise RuntimeError("BOOTSTRAP_QUANTILE_INVALID")
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * probability)]


def component_summary(row: dict, component_name: str) -> dict:
    component = row.get("components", {}).get(component_name)
    if not isinstance(component, dict) or not isinstance(component.get("token_count"), int) or component["token_count"] < 1:
        raise RuntimeError("GRID_COMPONENT_INVALID")
    count = component["token_count"]
    first = row.get("first_free_running_error")
    return {
        "correct_top1_rate": component.get("correct_top1_count") / count,
        "correct_top2_rate": component.get("correct_top2_count") / count,
        "mean_margin": component.get("margin_sum") / count,
        "first_error_rank": None if first is None else first.get("rank"),
        "first_error_parser_status": "NO_ERROR" if first is None else "ERROR",
        "token_count": count,
    }


def grid_summary(row: dict) -> dict:
    """Compatibility shape for the exact discrete B1/B32 gate."""
    value = component_summary(row, "GRID_CONTENT")
    return {"grid_content_correct_top1_rate": value["correct_top1_rate"], "grid_content_correct_top2_rate": value["correct_top2_rate"], "grid_content_mean_margin": value["mean_margin"], "first_error_rank": value["first_error_rank"], "first_error_parser_status": value["first_error_parser_status"], "grid_token_count": value["token_count"]}


def pair_modes(rows: list[dict], mode_id: str, expected_per_condition: int) -> dict[str, dict[str, dict]]:
    selected = [row for row in rows if row.get("mode_id") == mode_id]
    if len(selected) != expected_per_condition * 2:
        raise RuntimeError("MODE_ROW_COUNT_INVALID")
    grouped: dict[str, dict[str, dict]] = {}
    for row in selected:
        episode, condition = row.get("episode_id"), row.get("checkpoint_condition")
        if not isinstance(episode, str) or condition not in CONDITIONS or condition in grouped.setdefault(episode, {}):
            raise RuntimeError("MODE_PAIRING_INVALID")
        grouped[episode][condition] = row
    if len(grouped) != expected_per_condition or any(set(pair) != set(CONDITIONS) for pair in grouped.values()):
        raise RuntimeError("MODE_PAIRING_INVALID")
    return grouped


def paired_effect_rows(primary: dict[str, dict[str, dict]], component_name: str) -> dict[str, list[dict]]:
    metrics = {"top1": [], "top2": [], "margin": []}
    for pair in primary.values():
        baseline, candidate = pair[CONDITIONS[0]], pair[CONDITIONS[1]]
        if baseline.get("family") != candidate.get("family"):
            raise RuntimeError("PAIRED_FAMILY_MISMATCH")
        left, right = component_summary(baseline, component_name), component_summary(candidate, component_name)
        for metric, field in (("top1", "correct_top1_rate"), ("top2", "correct_top2_rate"), ("margin", "mean_margin")):
            metrics[metric].append({"family": baseline["family"], "family_balanced_minus_v7": right[field] - left[field]})
    return metrics


def bootstrap_interval(rows: list[dict]) -> tuple[float, float]:
    values = family_stratified_bootstrap_deltas(rows, seed=20261010, replicates=10_000)
    return quantile(values, 0.025), quantile(values, 0.975)


def equal_family_point(rows: list[dict]) -> float:
    families: dict[str, list[float]] = {}
    for row in rows:
        families.setdefault(row["family"], []).append(float(row["family_balanced_minus_v7"]))
    if len(families) != 5 or any(len(values) != 12 for values in families.values()):
        raise RuntimeError("EQUAL_FAMILY_POINT_INVALID")
    return sum(sum(values) / len(values) for values in families.values()) / len(families)


def family_cells(metric_rows: list[dict], *, margin_u: float = 0.0) -> dict:
    output = {}
    for family in sorted({row["family"] for row in metric_rows}):
        values = [float(row["family_balanced_minus_v7"]) for row in metric_rows if row["family"] == family]
        if len(values) != 12:
            raise RuntimeError("FAMILY_CELL_DENOMINATOR_INVALID")
        # Deterministic within-family paired bootstrap, kept separate from the
        # equal-family primary estimand.
        import random
        rng = random.Random(20261010)
        boot = [sum(rng.choice(values) for _ in values) / len(values) for _ in range(10_000)]
        raw = (quantile(boot, 0.025), quantile(boot, 0.975))
        output[family] = {"point_estimate": sum(values) / len(values), "raw_bootstrap_95_ci": raw, "numerical_sensitivity_adjusted_ci": adjusted_interval(raw, margin_u)}
    return output


def pooled_token_micro(primary: dict[str, dict[str, dict]], component_name: str) -> dict:
    totals = {}
    for condition in CONDITIONS:
        aggregate = {"token_count": 0, "correct_top1_count": 0, "correct_top2_count": 0, "margin_sum": 0.0}
        for pair in primary.values():
            value = pair[condition]["components"].get(component_name)
            if not isinstance(value, dict):
                raise RuntimeError("POOLED_COMPONENT_INVALID")
            for key in aggregate:
                aggregate[key] += value[key]
        totals[condition] = aggregate
    return {"token_count": totals[CONDITIONS[0]]["token_count"], "family_balanced_minus_v7_top1": totals[CONDITIONS[1]]["correct_top1_count"] / totals[CONDITIONS[1]]["token_count"] - totals[CONDITIONS[0]]["correct_top1_count"] / totals[CONDITIONS[0]]["token_count"], "family_balanced_minus_v7_top2": totals[CONDITIONS[1]]["correct_top2_count"] / totals[CONDITIONS[1]]["token_count"] - totals[CONDITIONS[0]]["correct_top2_count"] / totals[CONDITIONS[0]]["token_count"], "family_balanced_minus_v7_margin": totals[CONDITIONS[1]]["margin_sum"] / totals[CONDITIONS[1]]["token_count"] - totals[CONDITIONS[0]]["margin_sum"] / totals[CONDITIONS[0]]["token_count"]}


def first_error_summary(primary: dict[str, dict[str, dict]]) -> dict:
    output = {}
    for condition in CONDITIONS:
        by_family = {}
        for family in sorted({pair[condition]["family"] for pair in primary.values()}):
            errors = [pair[condition].get("first_free_running_error") for pair in primary.values() if pair[condition]["family"] == family]
            observed = [item for item in errors if item is not None]
            by_family[family] = {"episode_count": len(errors), "first_error_count": len(observed), "mean_rank_when_present": None if not observed else sum(item["rank"] for item in observed) / len(observed), "mean_margin_when_present": None if not observed else sum(item["margin"] for item in observed) / len(observed)}
        output[condition] = by_family
    return output


def descriptive_exact_groups(primary: dict[str, dict[str, dict]]) -> dict:
    groups = {"V7_EXACT_FAMILY_BALANCED_EXACT": 0, "V7_EXACT_FAMILY_BALANCED_NONEXACT": 0, "V7_NONEXACT_FAMILY_BALANCED_EXACT": 0, "V7_NONEXACT_FAMILY_BALANCED_NONEXACT": 0}
    for pair in primary.values():
        left = pair[CONDITIONS[0]].get("first_free_running_error") is None
        right = pair[CONDITIONS[1]].get("first_free_running_error") is None
        groups[("V7_EXACT" if left else "V7_NONEXACT") + ("_FAMILY_BALANCED_EXACT" if right else "_FAMILY_BALANCED_NONEXACT")] += 1
    return {"descriptive_only": True, "episode_counts": groups}


def classify(*, discrete_status: str, top1_adjusted: tuple[float, float], top2_adjusted: tuple[float, float], margin_adjusted: tuple[float, float]) -> str:
    if discrete_status != "PASS":
        return "INCONCLUSIVE_INVALID"
    if top1_adjusted[0] > 0 and top2_adjusted[0] >= 0 and margin_adjusted[0] >= 0:
        return "LOCAL_RANKING_IMPROVED_BUT_EXACT_NOT_IMPROVED"
    if top1_adjusted[1] <= 0 and margin_adjusted[1] <= 0:
        return "RANKING_NOT_IMPROVED"
    return "INCONCLUSIVE_MIXED_EVIDENCE"


def postprocess(rows: list[dict], *, protected_families: set[str] | None = None) -> dict:
    primary, batch1 = pair_modes(rows, "PRIMARY_B32", 60), pair_modes(rows, "SENSITIVITY_B1", 12)
    if not set(batch1).issubset(primary):
        raise RuntimeError("BATCH1_NOT_PRIMARY_SUBSET")
    primary_discrete, batch1_discrete = {}, {}
    primary_margin, batch1_margin = {}, {}
    for episode in batch1:
        for condition in CONDITIONS:
            key = f"{condition}:{episode}"
            primary_discrete[key] = grid_summary(primary[episode][condition])
            batch1_discrete[key] = grid_summary(batch1[episode][condition])
        primary_margin[episode] = {"v7_grid_margin": primary_discrete[f"{CONDITIONS[0]}:{episode}"]["grid_content_mean_margin"], "family_balanced_grid_margin": primary_discrete[f"{CONDITIONS[1]}:{episode}"]["grid_content_mean_margin"]}
        batch1_margin[episode] = {"v7_grid_margin": batch1_discrete[f"{CONDITIONS[0]}:{episode}"]["grid_content_mean_margin"], "family_balanced_grid_margin": batch1_discrete[f"{CONDITIONS[1]}:{episode}"]["grid_content_mean_margin"]}
    discrete = sensitivity_gate(primary_discrete, batch1_discrete)
    u = paired_margin_sensitivity_u(primary_margin, batch1_margin)
    effects = paired_effect_rows(primary, "GRID_CONTENT")
    points = {metric: equal_family_point(metric_rows) for metric, metric_rows in effects.items()}
    intervals = {metric: bootstrap_interval(metric_rows) for metric, metric_rows in effects.items()}
    adjusted = {metric: adjusted_interval(interval, u if metric == "margin" else 0.0) for metric, interval in intervals.items()}
    all_components = {}
    for component in COMPONENTS:
        component_effects = paired_effect_rows(primary, component)
        all_components[component] = {"equal_family_macro": {metric: {"point_estimate": equal_family_point(metric_rows), "raw_bootstrap_95_ci": bootstrap_interval(metric_rows), "numerical_sensitivity_adjusted_ci": adjusted_interval(bootstrap_interval(metric_rows), u if metric == "margin" else 0.0)} for metric, metric_rows in component_effects.items()}, "pooled_token_micro": pooled_token_micro(primary, component)}
    protected = protected_families or {"PROTECTED_SAME_COLOR"}
    protected_cells = family_cells(effects["top1"])
    harms = sorted(family for family in protected & set(protected_cells) if protected_cells[family]["numerical_sensitivity_adjusted_ci"][1] < 0)
    classification = "FAMILY_SPECIFIC_RANKING_HARM" if harms else classify(discrete_status=discrete["status"], top1_adjusted=adjusted["top1"], top2_adjusted=adjusted["top2"], margin_adjusted=adjusted["margin"])
    return {
        "protocol_id": "PAIRED_V7_FAMILY_BALANCED_CORRECT_TOKEN_RANK_MARGIN_V1",
        "status": "COMPLETE_TARGET_BLIND_REUSED_DEV_EVIDENCE_ONLY",
        "primary_estimand": "FAMILY_BALANCED_minus_V7_equal_family_macro_GRID_CONTENT",
        "equal_family_macro_points": points,
        "raw_bootstrap_95_ci": intervals,
        "numerical_sensitivity_u": u,
        "numerical_sensitivity_adjusted_ci": adjusted,
        "batch1_discrete_gate": discrete,
        "classification": classification,
        "family_specific_ranking_harm": harms,
        "grid_content_family_cells": {metric: family_cells(metric_rows, margin_u=u if metric == "margin" else 0.0) for metric, metric_rows in effects.items()},
        "component_results": all_components,
        "first_error_summary": first_error_summary(primary),
        "outcome_defined_exact_grid_groups": descriptive_exact_groups(primary),
        "raw_rows": len(rows),
        "target_token_ids_persisted": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--raw", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("POSTPROCESS_OUTPUT_NON_OVERWRITE_REQUIRED")
    rows = [json.loads(line) for line in args.raw.read_text(encoding="utf-8").splitlines() if line.strip()]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(postprocess(rows), sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
