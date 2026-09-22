"""Run the CPU-only Frozen60 primary Stage-0 analysis."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
import sys

THIS = Path(__file__).resolve()
FROZEN60_ROOT = THIS.parents[1]
PILOT_ROOT = THIS.parents[2]
sys.path.insert(0, str(PILOT_ROOT / "src"))
sys.path.insert(0, str(THIS.parent))
from metrics import clean_selector_target, contamination_simulation, reliability_variants, thresholds
from plotting import cohort_comparison, contamination, failure_decomposition, top2_hierarchy

DATA60 = FROZEN60_ROOT / "data" / "arc60_pilot.csv"
DATA30 = PILOT_ROOT / "data" / "arc30_pilot.csv"
OUT60 = FROZEN60_ROOT / "outputs"
COMPARISON = PILOT_ROOT / "comparison"


def load(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"missing derived frozen dataset: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["any_of_k_correct"] = bool(int(row["any_of_k_correct"]))
        row["top1_correct"] = bool(int(row["top1_correct"]))
        if "top2_correct" in row:
            row["top2_correct"] = bool(int(row["top2_correct"]))
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def summary(name: str, rows: list[dict], reliability: dict) -> dict:
    counts = Counter(str(row["failure_source"]) for row in rows)
    total = len(rows); any_k = sum(row["any_of_k_correct"] for row in rows); top1 = sum(row["top1_correct"] for row in rows)
    top2 = sum(row.get("top2_correct", False) for row in rows) if any("top2_correct" in row for row in rows) else None
    failures = total - counts["success"]
    return {
        "cohort": name, "task_count": total, "any_of_k": any_k, "any_of_k_rate": any_k / total,
        "top1": top1, "top1_rate": top1 / total, "top2": "" if top2 is None else top2,
        # Frozen30's cached artifact has no frozen second attempt, so missing
        # Top-2 is represented as N/A rather than a zero score.
        "top2_rate": "" if top2 is None else top2 / total,
        "conditional_selector_accuracy": top1 / any_k,
        "generator_limited_failures": counts["generator_limited_failure"],
        "selection_failures": counts["selection_failure"], "all_failures": failures,
        "non_selection_responsible_failure_fraction": counts["generator_limited_failure"] / failures,
        "naive_reliability_estimate": reliability["NAIVE"].estimate,
        "oracle_skip_reliability_estimate": reliability["ORACLE_SKIP"].estimate,
        "naive_absolute_error": abs(reliability["NAIVE"].estimate - top1 / any_k),
        "oracle_skip_absolute_error": abs(reliability["ORACLE_SKIP"].estimate - top1 / any_k),
    }


def main() -> None:
    rows60, rows30 = load(DATA60), load(DATA30)
    if len(rows60) != 60 or len(rows30) != 30:
        raise ValueError("expected exact 60-task primary and 30-task historical cohorts")
    ids60, ids30 = {row["task_id"] for row in rows60}, {row["task_id"] for row in rows30}
    overlap = sorted(ids60 & ids30)
    OUT60.mkdir(parents=True, exist_ok=True); COMPARISON.mkdir(parents=True, exist_ok=True)
    counts = Counter(str(row["failure_source"]) for row in rows60)
    total = len(rows60); any_k = sum(row["any_of_k_correct"] for row in rows60); top1 = sum(row["top1_correct"] for row in rows60); top2 = sum(row["top2_correct"] for row in rows60)
    clean = clean_selector_target(rows60)
    descriptor = [
        {"metric": "task_count", "value": total}, {"metric": "any_of_k_correct", "value": any_k},
        {"metric": "top1_correct", "value": top1}, {"metric": "top2_correct", "value": top2},
        {"metric": "success_count", "value": counts["success"]}, {"metric": "selection_failure_count", "value": counts["selection_failure"]},
        {"metric": "generator_limited_failure_count", "value": counts["generator_limited_failure"]},
        {"metric": "non_selection_responsible_failure_fraction", "value": counts["generator_limited_failure"] / (total - top1)},
        {"metric": "conditional_selector_accuracy", "value": clean}, {"metric": "top2_conditional_retention", "value": top2 / any_k},
        {"metric": "any_of_k_to_top2_headroom", "value": (any_k - top2) / total}, {"metric": "top2_to_top1_headroom", "value": (top2 - top1) / total},
        {"metric": "top1_wrong_top2_correct", "value": top2 - top1}, {"metric": "correct_candidate_below_top2", "value": any_k - top2},
    ]
    write_csv(OUT60 / "descriptive_summary.csv", descriptor)
    failure_decomposition(rows60, OUT60 / "frozen60_failure_decomposition.png")
    top2_hierarchy(rows60, OUT60 / "frozen60_top2_hierarchy.png")

    reliability60 = reliability_variants(rows60)
    prior = []
    for value in (0.5, 1.0, 2.0):
        for method, estimate in reliability_variants(rows60, value, value).items():
            prior.append({"prior_alpha": value, "prior_beta": value, "method": method, "estimate": estimate.estimate, "lower_95": estimate.lower_95, "upper_95": estimate.upper_95, "absolute_error_vs_clean_target": abs(estimate.estimate - clean)})
    (OUT60 / "reliability_prior_sensitivity.json").write_text(json.dumps(prior, indent=2), encoding="utf-8")
    contamination_rows = contamination_simulation(rows60, [0.00, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50], repetitions=1000, seed=20260920)
    write_csv(OUT60 / "contamination_results.csv", contamination_rows)
    threshold_rows = thresholds(contamination_rows)
    write_csv(OUT60 / "threshold_summary.csv", threshold_rows)
    contamination(contamination_rows, OUT60 / "frozen60_contamination_sensitivity.png")

    reliability30 = reliability_variants(rows30)
    comparison_rows = [summary("Frozen30 historical", rows30, reliability30), summary("Frozen60 primary", rows60, reliability60)]
    write_csv(COMPARISON / "frozen30_vs_frozen60.csv", comparison_rows)
    cohort_comparison(comparison_rows, COMPARISON / "frozen30_vs_frozen60.png")
    threshold = {row["threshold"]: row["rho"] for row in threshold_rows}
    report = f"""# Frozen60 primary Stage-0 report

## Artifact audit

Frozen60 uses the immutable local manifest, its cached 32-view candidate pool,
and frozen B-SUPPORT two-attempt predictions. The extractor validates all task
membership, freeze statuses, source SHA linkage, and 32-augmentation metadata
before opening the already-local training labels. No inference, GPU, model, or
Kaggle operation is used.

Frozen30 has {len(ids30)} tasks and Frozen60 has {len(ids60)}; task overlap is
**{len(overlap)}**. They are task-disjoint, but share project/model context and
are not claimed as fully independent statistical replications.

## Reproduced Frozen60 metrics

- Any-of-K: **{any_k}/60**
- Top-1: **{top1}/60**
- Top-2: **{top2}/60**

## Failure decomposition and Top-2 structure

| Category | Tasks | Rate |
|---|---:|---:|
| Success / Top-1 correct | {counts['success']} | {counts['success']/total:.1%} |
| Selection failure | {counts['selection_failure']} | {counts['selection_failure']/total:.1%} |
| Generator-limited failure | {counts['generator_limited_failure']} | {counts['generator_limited_failure']/total:.1%} |

Among {total-top1} failures, **{counts['generator_limited_failure']}/{total-top1} = {counts['generator_limited_failure']/(total-top1):.1%}** are non-selection-responsible under this ARC taxonomy. Conditional selector accuracy is {top1}/{any_k} = **{clean:.1%}**. Top-2 conditional retention is {top2}/{any_k} = **{top2/any_k:.1%}**.

Of the {any_k} candidate-covered tasks: {top1} are Top-1 correct, {top2-top1} are additional fixed Top-2 recoveries, and {any_k-top2} retain a correct candidate below Top-2. Top-2 therefore adds limited but real portfolio evidence beyond the Any-of-K/Top-1 split; it does not resolve the remaining ranking or generator limitations.

## Beta(1,1) selector reliability

Clean target: `{top1}/{any_k} = {clean:.4f}`. This is conditional on a correct cached candidate existing.

| Method | Posterior mean | 95% credible interval | Absolute error |
|---|---:|---:|---:|
"""
    for method, estimate in reliability60.items():
        report += f"| {method.replace('_', '-')} | {estimate.estimate:.4f} | [{estimate.lower_95:.4f}, {estimate.upper_95:.4f}] | {abs(estimate.estimate-clean):.4f} |\n"
    report += f"""

## Frozen30 comparison

Frozen30 Top-1 is {sum(row['top1_correct'] for row in rows30)}/30, higher than Frozen60's {top1}/60, while its conditional selector accuracy is {clean_selector_target(rows30):.1%}, lower than Frozen60's {clean:.1%}. This supports the narrow claim that end-to-end task success is not the same quantity as selection-layer reliability: Frozen60 has more generator-limited failures while its selector does better when a correct candidate exists. The comparison is descriptive only.

## Frozen60 contamination sensitivity

At every rho, 1,000 fixed-seed trials evaluate reliability against clean latent selection correctness after flipping only genuine correct selections to observed external failures. The statistical threshold is rho={threshold['statistical_naive_vs_oracle_skip_95pct_excludes_zero']}; the 5pp practical heuristic is rho={threshold['practical_naive_bias_exceeds_5pp']}. Both are zero because the unmodified Frozen60 already has 30 generator-limited failures that Naive misattributes as selector errors. This is not an empirical estimate of real external failure rates.

## Critical interpretation and updated recommendation

Oracle-Skip necessarily beats Naive once generator-limited labels are known. Frozen60 therefore demonstrates a responsibility-label mismatch, not a solved online attribution mechanism. Oracle-Clean equals Oracle-Skip here because exact ARC candidate correctness makes the clean selection label available on exactly the same candidate-covered tasks: this is partly tautological. Frozen60 is larger and Top-2 adds five target-blind recoveries, but it strengthens motivation more than it proves the future interactive-agent hypothesis.

**WEAK GO.** Next, run a pre-registered CPU synthetic interactive sandbox with independently hidden generator, selector, executor, and environment failures. Compare telemetry-only attribution to an oracle upper bound and a naive outcome updater. Do not derive labels from test targets during that experiment.

## Limitations

ARC2 is not an interactive LLM-agent environment. Its ranker/selector is only a proxy for future verifier research. Generator limitation is unusually observable due to exact ARC scoring. Synthetic rhos are not real failure-rate estimates. n=60 is small. No causal claim about general LLM agents follows.
"""
    (FROZEN60_ROOT / "FROZEN60_REPORT.md").write_text(report, encoding="utf-8")
    (COMPARISON / "STAGE0_COMPARISON_REPORT.md").write_text(f"# Frozen30 vs Frozen60\n\nTask-ID overlap: **{len(overlap)}**. The cohorts are task-disjoint but share project/model context. See `frozen30_vs_frozen60.csv` for reproducible descriptive metrics; do not interpret it as independent-replication significance testing.\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "any_of_k": any_k, "top1": top1, "top2": top2, "overlap_with_frozen30": len(overlap)}, indent=2))


if __name__ == "__main__":
    main()
