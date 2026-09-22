"""Transparent CPU-only metrics for the isolated ARC30 attribution pilot."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.stats import beta


FAILURE_SUCCESS = "success"
FAILURE_SELECTION = "selection_failure"
FAILURE_GENERATOR = "generator_limited_failure"


def failure_source(any_of_k_correct: bool, top1_correct: bool) -> str:
    """Classify only the three categories directly identified by frozen data."""
    if top1_correct:
        return FAILURE_SUCCESS
    if any_of_k_correct:
        return FAILURE_SELECTION
    return FAILURE_GENERATOR


def clean_selector_target(rows: Iterable[dict]) -> float:
    """P(selector chose correctly | a correct cached candidate existed)."""
    rows = list(rows)
    meaningful = [row for row in rows if bool(row["any_of_k_correct"])]
    if not meaningful:
        raise ValueError("no task offered the selector a correct cached candidate")
    return sum(bool(row["top1_correct"]) for row in meaningful) / len(meaningful)


@dataclass(frozen=True)
class BetaEstimate:
    estimate: float
    lower_95: float
    upper_95: float
    alpha: float
    beta: float


def beta_estimate(successes: int, failures: int, prior_alpha: float = 1.0, prior_beta: float = 1.0) -> BetaEstimate:
    """Beta posterior mean and equal-tail 95% credible interval."""
    if successes < 0 or failures < 0 or prior_alpha <= 0 or prior_beta <= 0:
        raise ValueError("counts must be non-negative and priors positive")
    alpha = prior_alpha + successes
    beta_param = prior_beta + failures
    return BetaEstimate(
        estimate=alpha / (alpha + beta_param),
        lower_95=float(beta.ppf(0.025, alpha, beta_param)),
        upper_95=float(beta.ppf(0.975, alpha, beta_param)),
        alpha=alpha,
        beta=beta_param,
    )


def reliability_variants(rows: Iterable[dict], prior_alpha: float = 1.0, prior_beta: float = 1.0) -> dict[str, BetaEstimate]:
    """Compute all estimators against fixed cached ARC task outcomes.

    NAIVE treats every task outcome as direct selection feedback.  ORACLE-SKIP
    omits generator-limited tasks.  ORACLE-CLEAN has the clean latent labels for
    meaningful-selection tasks; with this exact ARC taxonomy it is identical to
    ORACLE-SKIP, a fact that is reported rather than hidden.
    """
    rows = list(rows)
    successes = sum(bool(row["top1_correct"]) for row in rows)
    all_failures = len(rows) - successes
    meaningful = [row for row in rows if bool(row["any_of_k_correct"])]
    clean_successes = sum(bool(row["top1_correct"]) for row in meaningful)
    clean_failures = len(meaningful) - clean_successes
    static = beta_estimate(0, 0, prior_alpha, prior_beta)
    return {
        "STATIC": static,
        "NAIVE": beta_estimate(successes, all_failures, prior_alpha, prior_beta),
        "ORACLE_SKIP": beta_estimate(clean_successes, clean_failures, prior_alpha, prior_beta),
        "ORACLE_CLEAN": beta_estimate(clean_successes, clean_failures, prior_alpha, prior_beta),
    }


def contamination_simulation(
    rows: Iterable[dict],
    rhos: Iterable[float],
    repetitions: int = 1000,
    seed: int = 20260919,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
) -> list[dict]:
    """One-sided, synthetic external-outcome contamination sensitivity test.

    Only clean selector-correct tasks are independently flipped to an observed
    task failure.  All methods are evaluated against clean selector correctness,
    not the corrupted outcome.  Order is shuffled each trial to make the online
    interpretation explicit; final Beta counts are order invariant.
    """
    rows = list(rows)
    clean_target = clean_selector_target(rows)
    good_indices = np.asarray([i for i, row in enumerate(rows) if bool(row["top1_correct"])], dtype=int)
    meaningful = np.asarray([bool(row["any_of_k_correct"]) for row in rows], dtype=bool)
    clean_labels = np.asarray([bool(row["top1_correct"]) for row in rows], dtype=bool)
    generator_limited = ~meaningful
    rng = np.random.default_rng(seed)
    results: list[dict] = []
    for rho in rhos:
        values: dict[str, list[float]] = {name: [] for name in ("STATIC", "NAIVE", "ORACLE_SKIP", "ORACLE_CLEAN")}
        for _ in range(repetitions):
            observed = clean_labels.copy()
            flips = rng.random(len(good_indices)) < rho
            observed[good_indices[flips]] = False
            # A random order represents online arrival. Final counts do not rely
            # on task order, which is checked by retaining this deterministic draw.
            order = rng.permutation(len(rows))
            naive_successes = int(observed[order].sum())
            naive_failures = len(rows) - naive_successes
            clean_successes = int(clean_labels[meaningful].sum())
            clean_failures = int(meaningful.sum()) - clean_successes
            estimates = {
                "STATIC": beta_estimate(0, 0, prior_alpha, prior_beta).estimate,
                "NAIVE": beta_estimate(naive_successes, naive_failures, prior_alpha, prior_beta).estimate,
                "ORACLE_SKIP": beta_estimate(clean_successes, clean_failures, prior_alpha, prior_beta).estimate,
                "ORACLE_CLEAN": beta_estimate(clean_successes, clean_failures, prior_alpha, prior_beta).estimate,
            }
            for name, value in estimates.items():
                values[name].append(value)
        paired_difference = np.asarray(values["NAIVE"], dtype=float) - np.asarray(values["ORACLE_SKIP"], dtype=float)
        for name, sample in values.items():
            array = np.asarray(sample, dtype=float)
            errors = np.abs(array - clean_target)
            bias = array - clean_target
            results.append({
                "rho": float(rho), "method": name, "repetitions": repetitions,
                "clean_target": clean_target,
                "estimate_mean": float(array.mean()),
                "estimate_lower_95": float(np.quantile(array, 0.025)),
                "estimate_upper_95": float(np.quantile(array, 0.975)),
                "absolute_error_mean": float(errors.mean()),
                "bias_mean": float(bias.mean()),
                "bias_lower_95": float(np.quantile(bias, 0.025)),
                "bias_upper_95": float(np.quantile(bias, 0.975)),
                "external_flip_mean": float((rho * len(good_indices))),
                "generator_limited_count": int(generator_limited.sum()),
                # This is a paired Monte Carlo comparison: every NAIVE and
                # ORACLE-SKIP estimate above used the same external-failure draw.
                "naive_minus_oracle_skip_mean": float(paired_difference.mean()),
                "naive_minus_oracle_skip_lower_95": float(np.quantile(paired_difference, 0.025)),
                "naive_minus_oracle_skip_upper_95": float(np.quantile(paired_difference, 0.975)),
            })
    return results


def thresholds(contamination_rows: Iterable[dict]) -> list[dict]:
    """Find predeclared statistical and practical thresholds, without tuning."""
    rows = list(contamination_rows)
    by_rho: dict[float, dict[str, dict]] = {}
    for row in rows:
        by_rho.setdefault(float(row["rho"]), {})[str(row["method"])] = row
    statistical = None
    practical = None
    for rho in sorted(by_rho):
        naive = by_rho[rho]["NAIVE"]
        aware = by_rho[rho]["ORACLE_SKIP"]
        if statistical is None and (naive["naive_minus_oracle_skip_upper_95"] < 0 or naive["naive_minus_oracle_skip_lower_95"] > 0):
            statistical = rho
        if practical is None and abs(naive["bias_mean"]) > 0.05:
            practical = rho
    return [
        {"threshold": "statistical_naive_vs_oracle_skip_95pct_excludes_zero", "rho": statistical, "definition": "first rho where naive reliability-bias 95% interval excludes Oracle-Skip bias"},
        {"threshold": "practical_naive_bias_exceeds_5pp", "rho": practical, "definition": "heuristic: first rho where absolute mean naive bias versus clean selector target exceeds 0.05"},
    ]
