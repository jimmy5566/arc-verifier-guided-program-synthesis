#!/usr/bin/env python3
"""Target-blind CPU postprocessor for paired V7/Family-Balanced rank evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.paired_v7_family_balanced_rank_margin_contract import adjusted_interval, family_stratified_bootstrap_deltas, paired_margin_sensitivity_u, sensitivity_gate

CONDITIONS = ("RECONSTRUCTED_FOUNDATION_V2_V7", "FAMILY_BALANCED")


def quantile(values: list[float], probability: float) -> float:
    if not values or not 0 <= probability <= 1:
        raise RuntimeError("BOOTSTRAP_QUANTILE_INVALID")
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * probability)]


def grid_summary(row: dict) -> dict:
    component = row.get("components", {}).get("GRID_CONTENT")
    if not isinstance(component, dict) or not isinstance(component.get("token_count"), int) or component["token_count"] < 1:
        raise RuntimeError("GRID_COMPONENT_INVALID")
    count = component["token_count"]
    first = row.get("first_free_running_error")
    return {
        "grid_content_correct_top1_rate": component.get("correct_top1_count") / count,
        "grid_content_correct_top2_rate": component.get("correct_top2_count") / count,
        "grid_content_mean_margin": component.get("margin_sum") / count,
        "first_error_rank": None if first is None else first.get("rank"),
        "first_error_parser_status": "NO_ERROR" if first is None else "ERROR",
        "grid_token_count": count,
    }


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


def paired_effect_rows(primary: dict[str, dict[str, dict]]) -> dict[str, list[dict]]:
    metrics = {"top1": [], "top2": [], "margin": []}
    for pair in primary.values():
        baseline, candidate = pair[CONDITIONS[0]], pair[CONDITIONS[1]]
        if baseline.get("family") != candidate.get("family"):
            raise RuntimeError("PAIRED_FAMILY_MISMATCH")
        left, right = grid_summary(baseline), grid_summary(candidate)
        for metric, field in (("top1", "grid_content_correct_top1_rate"), ("top2", "grid_content_correct_top2_rate"), ("margin", "grid_content_mean_margin")):
            metrics[metric].append({"family": baseline["family"], "family_balanced_minus_v7": right[field] - left[field]})
    return metrics


def bootstrap_interval(rows: list[dict]) -> tuple[float, float]:
    values = family_stratified_bootstrap_deltas(rows, seed=20261010, replicates=10_000)
    return quantile(values, 0.025), quantile(values, 0.975)


def classify(*, discrete_status: str, top1_adjusted: tuple[float, float], top2_adjusted: tuple[float, float], margin_adjusted: tuple[float, float]) -> str:
    if discrete_status != "PASS":
        return "INCONCLUSIVE_INVALID"
    if top1_adjusted[0] > 0 and top2_adjusted[0] >= 0 and margin_adjusted[0] >= 0:
        return "LOCAL_RANKING_IMPROVED_BUT_EXACT_NOT_IMPROVED"
    if top1_adjusted[1] <= 0 and margin_adjusted[1] <= 0:
        return "RANKING_NOT_IMPROVED"
    return "INCONCLUSIVE_MIXED_EVIDENCE"


def postprocess(rows: list[dict]) -> dict:
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
    effects = paired_effect_rows(primary)
    intervals = {metric: bootstrap_interval(metric_rows) for metric, metric_rows in effects.items()}
    adjusted = {metric: adjusted_interval(interval, u if metric == "margin" else 0.0) for metric, interval in intervals.items()}
    return {
        "protocol_id": "PAIRED_V7_FAMILY_BALANCED_CORRECT_TOKEN_RANK_MARGIN_V1",
        "status": "COMPLETE_TARGET_BLIND_REUSED_DEV_EVIDENCE_ONLY",
        "primary_estimand": "FAMILY_BALANCED_minus_V7_equal_family_macro_GRID_CONTENT",
        "raw_bootstrap_95_ci": intervals,
        "numerical_sensitivity_u": u,
        "numerical_sensitivity_adjusted_ci": adjusted,
        "batch1_discrete_gate": discrete,
        "classification": classify(discrete_status=discrete["status"], top1_adjusted=adjusted["top1"], top2_adjusted=adjusted["top2"], margin_adjusted=adjusted["margin"]),
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
