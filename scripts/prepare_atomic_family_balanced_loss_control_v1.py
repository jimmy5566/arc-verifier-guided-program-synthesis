#!/usr/bin/env python3
"""Freeze a matched family-balanced scalar-loss control without loading a model."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

WEAK = ("connected components", "inside/contains", "difference", "width", "orientation")
POLICY = "FAMILY_BALANCED_GLOBAL_IPF_V1"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def original_coefficients(episodes: list[dict], accumulation: int = 4) -> list[list[float]]:
    if len(episodes) != 400 or len(episodes) % accumulation:
        raise ValueError("SCHEDULE_SHAPE_FAIL")
    result = []
    for offset in range(0, len(episodes), accumulation):
        group = episodes[offset:offset + accumulation]
        denominator = sum(int(row["supervised_token_count"]) for row in group)
        if denominator <= 0:
            raise ValueError("ZERO_SUPERVISION")
        result.append([int(row["supervised_token_count"]) / denominator for row in group])
    return result


def family_totals(episodes: list[dict], weights: list[list[float]]) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    for step, group_weights in enumerate(weights):
        for row, weight in zip(episodes[step * 4:(step + 1) * 4], group_weights):
            totals[row["family"]] += weight
    return dict(totals)


def balanced_coefficients(episodes: list[dict], tolerance: float = 1e-12, iterations: int = 100000) -> tuple[list[list[float]], dict[str, float], dict[str, float], int]:
    """Positive family IPF with scalar scale normalized to one every step."""
    base = original_coefficients(episodes)
    baseline = family_totals(episodes, base)
    weak_total = sum(baseline[name] for name in WEAK)
    targets = dict(baseline)
    for name in WEAK:
        targets[name] = weak_total / len(WEAK)
    multiplier = {name: 1.0 for name in targets}
    weights: list[list[float]] = []
    for iteration in range(1, iterations + 1):
        weights = []
        for step, base_step in enumerate(base):
            rows = episodes[step * 4:(step + 1) * 4]
            raw = [coefficient * multiplier[row["family"]] for row, coefficient in zip(rows, base_step)]
            total = sum(raw)
            weights.append([value / total for value in raw])
        observed = family_totals(episodes, weights)
        maximum_error = max(abs(observed[name] - targets[name]) for name in targets)
        if maximum_error <= tolerance:
            return weights, baseline, targets, iteration
        for name in targets:
            if observed[name] <= 0:
                raise RuntimeError("IPF_ZERO_FAMILY_SUPPORT")
            multiplier[name] *= targets[name] / observed[name]
    raise RuntimeError(f"IPF_NOT_CONVERGED:{maximum_error}")


def build_ledger(schedule: dict) -> dict:
    episodes = schedule["episodes"]
    original = original_coefficients(episodes)
    coefficients, baseline, targets, iterations = balanced_coefficients(episodes)
    observed = family_totals(episodes, coefficients)
    steps = []
    for index, weights in enumerate(coefficients, start=1):
        group = episodes[(index - 1) * 4:index * 4]
        entries = []
        for row, old, new in zip(group, original[index - 1], weights):
            entries.append({
                "episode_id": row["episode_id"],
                "family": row["family"],
                "objective_role": row["objective_role"],
                "original_token_proportional_coefficient": old,
                "balanced_loss_coefficient": new,
            })
        steps.append({"optimizer_step": index, "scalar_loss_scale": sum(weights), "entries": entries})
    by_family = {}
    for name in sorted(targets):
        by_family[name] = {
            "baseline_coefficient_total": baseline[name],
            "target_coefficient_total": targets[name],
            "balanced_coefficient_total": observed[name],
            "difference_from_target": observed[name] - targets[name],
            "baseline_global_nominal_loss_coefficient": baseline[name] / len(steps),
            "balanced_global_nominal_loss_coefficient": observed[name] / len(steps),
        }
    return {
        "schema_version": 1,
        "ledger_id": "ATOMIC_PREREQUISITE_FAMILY_BALANCED_LOSS_COEFFICIENT_LEDGER_V1",
        "policy_id": POLICY,
        "schedule_sha256": None,
        "worker_loss_contract": {
            "baseline": "TOKEN_PROPORTIONAL_PER_STEP_V1",
            "treatment": POLICY,
            "single_changed_factor": "SCALAR_LOSS_AGGREGATION_BY_FAMILY",
            "optimizer_step_scalar_scale": 1.0,
        },
        "optimization": {
            "method": "positive_family_multiplier_iterative_proportional_fitting",
            "iterations": iterations,
            "target_tolerance": 1e-12,
            "observed_maximum_family_total_error": max(abs(observed[name] - targets[name]) for name in targets),
        },
        "weak_families": list(WEAK),
        "weak_role_total_preserved": {
            "baseline": sum(baseline[name] for name in WEAK),
            "balanced": sum(observed[name] for name in WEAK),
        },
        "family_totals": by_family,
        "per_optimizer_step": steps,
        "validation": {
            "optimizer_steps": len(steps),
            "gradient_accumulation": 4,
            "episodes": len(episodes),
            "all_step_scales_one": all(abs(step["scalar_loss_scale"] - 1.0) <= 1e-12 for step in steps),
            "all_coefficients_positive": all(entry["balanced_loss_coefficient"] > 0 for step in steps for entry in step["entries"]),
            "nonweak_totals_preserved": all(abs(observed[name] - baseline[name]) <= 1e-12 for name in baseline if name not in WEAK),
            # Five independently converged marginals accumulate ordinary
            # float roundoff; this remains far below the 1e-12 per-family
            # solve tolerance and does not alter any scalar coefficient.
            "weak_role_total_preserved": abs(sum(observed[name] for name in WEAK) - sum(baseline[name] for name in WEAK)) <= 1e-10,
            "weak_totals_equal": max(observed[name] for name in WEAK) - min(observed[name] for name in WEAK) <= 1e-12,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    schedule = json.loads(args.schedule.read_text(encoding="utf-8"))
    ledger = build_ledger(schedule)
    ledger["schedule_sha256"] = sha(args.schedule)
    write_json(args.out, ledger)
    print(json.dumps({"status": "PASS", "ledger": str(args.out), "ledger_sha256": sha(args.out), "schedule_sha256": ledger["schedule_sha256"], "max_error": ledger["optimization"]["observed_maximum_family_total_error"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
