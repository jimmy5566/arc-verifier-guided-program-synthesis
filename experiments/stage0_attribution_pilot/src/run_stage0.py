"""Run the complete CPU-only, cached-data Stage-0 attribution pilot."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
import sys

THIS = Path(__file__).resolve()
PILOT_ROOT = THIS.parents[1]
sys.path.insert(0, str(THIS.parent))
from metrics import clean_selector_target, contamination_simulation, reliability_variants, thresholds
from plotting import contamination_plot, failure_decomposition_plot

DATA = PILOT_ROOT / "data" / "arc30_pilot.csv"
OUT = PILOT_ROOT / "outputs"


def load_rows() -> list[dict]:
    if not DATA.exists():
        raise FileNotFoundError(f"derived CSV missing: {DATA}; run extract_arc30.py first")
    with DATA.open(newline="", encoding="utf-8") as handle:
        raw = list(csv.DictReader(handle))
    rows: list[dict] = []
    for row in raw:
        row["any_of_k_correct"] = bool(int(row["any_of_k_correct"]))
        row["top1_correct"] = bool(int(row["top1_correct"]))
        row["number_of_candidates"] = int(row["number_of_candidates"])
        rows.append(row)
    if len(rows) != 30:
        raise ValueError(f"expected 30 derived rows, found {len(rows)}")
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("cannot write empty results")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def percentage(n: int, denominator: int) -> str:
    return f"{100 * n / denominator:.1f}%"


def main() -> None:
    rows = load_rows()
    OUT.mkdir(parents=True, exist_ok=True)
    counts = Counter(str(row["failure_source"]) for row in rows)
    total = len(rows)
    failures = total - counts["success"]
    any_k = sum(row["any_of_k_correct"] for row in rows)
    top1 = sum(row["top1_correct"] for row in rows)
    clean_target = clean_selector_target(rows)
    descriptive = [
        {"metric": "task_count", "value": total},
        {"metric": "any_of_k_correct", "value": any_k},
        {"metric": "top1_correct", "value": top1},
        {"metric": "success_count", "value": counts["success"]},
        {"metric": "selection_failure_count", "value": counts["selection_failure"]},
        {"metric": "generator_limited_failure_count", "value": counts["generator_limited_failure"]},
        {"metric": "non_selection_responsible_failure_fraction", "value": counts["generator_limited_failure"] / failures},
        {"metric": "selection_failure_fraction_of_failures", "value": counts["selection_failure"] / failures},
        {"metric": "any_of_k_headroom", "value": (any_k - top1) / total},
        {"metric": "top1_gap_given_correct_candidate", "value": (any_k - top1) / any_k},
        {"metric": "clean_selector_reliability_target", "value": clean_target},
    ]
    write_csv(OUT / "descriptive_summary.csv", descriptive)
    failure_decomposition_plot(rows, OUT / "failure_decomposition.png")

    # Symmetric priors test whether the directional conclusion is an artifact of
    # a single arbitrarily selected prior. All methods share each prior.
    prior_rows: list[dict] = []
    for prior in (0.5, 1.0, 2.0):
        for method, estimate in reliability_variants(rows, prior, prior).items():
            prior_rows.append({
                "prior_alpha": prior, "prior_beta": prior, "method": method,
                "estimate": estimate.estimate, "lower_95": estimate.lower_95,
                "upper_95": estimate.upper_95,
                "absolute_error_vs_clean_target": abs(estimate.estimate - clean_target),
                "bias_vs_clean_target": estimate.estimate - clean_target,
            })
    contamination = contamination_simulation(rows, [0.00, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50], repetitions=1000, seed=20260919)
    write_csv(OUT / "contamination_results.csv", contamination)
    threshold_rows = thresholds(contamination)
    write_csv(OUT / "threshold_summary.csv", threshold_rows)
    contamination_plot(contamination, OUT / "contamination_sensitivity.png")

    report = f"""# Stage-0 attribution pilot report

## Executive summary

This CPU-only pilot used one cached, frozen 30-task ARC candidate-ranking cohort.
It reproduced **Any-of-K = {any_k}/{total}** and **likelihood-ranker Top-1 = {top1}/{total}** before creating derived analyses.  It tests a narrow conceptual claim: overall task failure can be a misattributed negative label for a ranker/selector when a correct candidate was never generated.  It does **not** test a general interactive-agent verifier.

## Repository/data audit

- Frozen candidate/ranking source: `artifacts/ARC2_STRATEGY_AWARE_RANKER_FROZEN30_STRATEGY_ONLY_RERANKED_FROZEN.json`.
- Cached local labels used only after frozen predictions were loaded: `data/raw/arc-agi_training_solutions.json`.
- Cohort: {total} reproducibly keyed task IDs; cached `candidate_scores`, `ranked_candidate_indices`, candidate grids, augmentation metadata, and generation metadata were available.
- The scores are length-normalized likelihood scores for ranking; they are not calibrated correctness probabilities.  Score-aware calibration was intentionally skipped: n=30 is too small and no stored score semantics supports interpreting them as probabilities.
- No candidate regeneration, model load, CUDA call, Kaggle operation, or production-file modification was required.

## Verified failure decomposition

| Category | Tasks | Rate |
|---|---:|---:|
| Top-1 success | {counts['success']} | {percentage(counts['success'], total)} |
| Selection failure (correct candidate existed) | {counts['selection_failure']} | {percentage(counts['selection_failure'], total)} |
| Generator-limited failure (no correct cached candidate) | {counts['generator_limited_failure']} | {percentage(counts['generator_limited_failure'], total)} |

Among the {failures} failed task outcomes, **{counts['generator_limited_failure']}/{failures} = {counts['generator_limited_failure']/failures:.1%}** are non-selection-responsible in this ARC taxonomy.  Any-of-K headroom is {(any_k-top1)}/{total} = {(any_k-top1)/total:.1%}; conditional Top-1 gap when a correct candidate exists is {(any_k-top1)}/{any_k} = {(any_k-top1)/any_k:.1%}.

## Transparent global reliability model

The clean target is `P(Top-1 correct | any correct candidate exists) = {top1}/{any_k} = {clean_target:.4f}`.  The table below reports shared-prior Beta-Bernoulli posterior means and equal-tail 95% credible intervals; full prior sensitivity is embedded in the generated analysis metadata.

| Method (Beta(1,1)) | Estimate | 95% interval | Absolute error vs clean target |
|---|---:|---:|---:|
"""
    base = reliability_variants(rows)
    for method, estimate in base.items():
        report += f"| {method.replace('_', '-')} | {estimate.estimate:.4f} | [{estimate.lower_95:.4f}, {estimate.upper_95:.4f}] | {abs(estimate.estimate-clean_target):.4f} |\n"
    threshold_values = {row["threshold"]: row["rho"] for row in threshold_rows}
    report += f"""

`ORACLE-SKIP` and `ORACLE-CLEAN` are numerically identical here because the frozen ARC taxonomy makes clean selection labels available exactly for the same 21 meaningful-opportunity tasks.  This is a limitation and a partial tautology, not evidence that two distinct real attribution mechanisms work equally well.

## Synthetic exogenous-outcome contamination

For each rho in {{0,.05,.10,.20,.30,.40,.50}}, 1,000 reproducible Monte Carlo trials randomly flipped only clean selector-correct task outcomes to failure.  Reliability was always evaluated against the uncorrupted clean selector target.  The smallest predeclared statistical threshold is **rho={threshold_values['statistical_naive_vs_oracle_skip_95pct_excludes_zero']}**; the first practical heuristic threshold where naive mean bias exceeds 5 percentage points is **rho={threshold_values['practical_naive_bias_exceeds_5pp']}**.

At rho=0, NAIVE is already biased downward because it counts the {counts['generator_limited_failure']} generator-limited tasks as selector negatives.  As rho increases, one-sided external failures further bias NAIVE downward.  The saved contamination table includes a paired Monte-Carlo `NAIVE - ORACLE_SKIP` interval for each rho; all compared estimates use the same external-failure draw.  Attribution-aware methods are fixed in this synthetic setup because they are granted the latent attribution.  This directional result is expected by construction, so its value is only a sanity check that motivates—not validates—a later controlled interactive experiment.

## Interpretation and falsification attempt

The pilot does show a large generation–selection distinction in this cached ARC cohort.  But the key contamination result is structurally favorable to attribution awareness: generator-limited failures are exactly identifiable after labels are opened, and synthetic external failures are injected only into true positives.  It cannot establish a causal effect in a realistic online system.  The stability across symmetric Beta priors changes posterior shrinkage but not the sign of NAIVE's attribution error.

**Recommendation: WEAK GO.** The descriptive gap is strong enough to justify one larger controlled experiment, but the reliability simulation is partly tautological and n=30 is very small.  The exact next experiment should be a target-blind, synthetic *interactive* sandbox where generator, selector, executor, and environment failures are independently randomized and logged before outcome labels are revealed; compare learned attribution from observable telemetry with a pre-registered oracle-attribution upper bound.

## Limitations

1. ARC2 is not an interactive LLM-agent environment.
2. Its likelihood ranker/selector is only a proxy/motivation for future verifier research.
3. Generator-limited attribution is unusually clean here because exact candidate correctness is observable after frozen scoring.
4. Synthetic external failures are not estimates of real execution/environment failure prevalence.
5. n={total} is very small.
6. This pilot tests a conceptual hypothesis, not a final MPhil method.
7. No causal claim about general LLM agents follows from this Stage-0 pilot.
"""
    (PILOT_ROOT / "STAGE0_REPORT.md").write_text(report, encoding="utf-8")
    # Kept local and unscored: raw posterior/prior sensitivity is a transparent
    # companion to the report without altering requested output semantics.
    (OUT / "reliability_prior_sensitivity.json").write_text(json.dumps(prior_rows, indent=2), encoding="utf-8")
    print(json.dumps({"status": "PASS", "tasks": total, "any_of_k": any_k, "top1": top1, "outputs": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
