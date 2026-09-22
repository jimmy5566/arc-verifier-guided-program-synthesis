# Stage-0 attribution pilot report

## Executive summary

This CPU-only pilot used one cached, frozen 30-task ARC candidate-ranking cohort.
It reproduced **Any-of-K = 21/30** and **likelihood-ranker Top-1 = 9/30** before creating derived analyses.  It tests a narrow conceptual claim: overall task failure can be a misattributed negative label for a ranker/selector when a correct candidate was never generated.  It does **not** test a general interactive-agent verifier.

## Repository/data audit

- Frozen candidate/ranking source: `artifacts/ARC2_STRATEGY_AWARE_RANKER_FROZEN30_STRATEGY_ONLY_RERANKED_FROZEN.json`.
- Cached local labels used only after frozen predictions were loaded: `data/raw/arc-agi_training_solutions.json`.
- Cohort: 30 reproducibly keyed task IDs; cached `candidate_scores`, `ranked_candidate_indices`, candidate grids, augmentation metadata, and generation metadata were available.
- The scores are length-normalized likelihood scores for ranking; they are not calibrated correctness probabilities.  Score-aware calibration was intentionally skipped: n=30 is too small and no stored score semantics supports interpreting them as probabilities.
- No candidate regeneration, model load, CUDA call, Kaggle operation, or production-file modification was required.

## Verified failure decomposition

| Category | Tasks | Rate |
|---|---:|---:|
| Top-1 success | 9 | 30.0% |
| Selection failure (correct candidate existed) | 12 | 40.0% |
| Generator-limited failure (no correct cached candidate) | 9 | 30.0% |

Among the 21 failed task outcomes, **9/21 = 42.9%** are non-selection-responsible in this ARC taxonomy.  Any-of-K headroom is 12/30 = 40.0%; conditional Top-1 gap when a correct candidate exists is 12/21 = 57.1%.

## Transparent global reliability model

The clean target is `P(Top-1 correct | any correct candidate exists) = 9/21 = 0.4286`.  The table below reports shared-prior Beta-Bernoulli posterior means and equal-tail 95% credible intervals; full prior sensitivity is embedded in the generated analysis metadata.

| Method (Beta(1,1)) | Estimate | 95% interval | Absolute error vs clean target |
|---|---:|---:|---:|
| STATIC | 0.5000 | [0.0250, 0.9750] | 0.0714 |
| NAIVE | 0.3125 | [0.1668, 0.4804] | 0.1161 |
| ORACLE-SKIP | 0.4348 | [0.2439, 0.6365] | 0.0062 |
| ORACLE-CLEAN | 0.4348 | [0.2439, 0.6365] | 0.0062 |


`ORACLE-SKIP` and `ORACLE-CLEAN` are numerically identical here because the frozen ARC taxonomy makes clean selection labels available exactly for the same 21 meaningful-opportunity tasks.  This is a limitation and a partial tautology, not evidence that two distinct real attribution mechanisms work equally well.

## Synthetic exogenous-outcome contamination

For each rho in {0,.05,.10,.20,.30,.40,.50}, 1,000 reproducible Monte Carlo trials randomly flipped only clean selector-correct task outcomes to failure.  Reliability was always evaluated against the uncorrupted clean selector target.  The smallest predeclared statistical threshold is **rho=0.0**; the first practical heuristic threshold where naive mean bias exceeds 5 percentage points is **rho=0.0**.

At rho=0, NAIVE is already biased downward because it counts the 9 generator-limited tasks as selector negatives.  As rho increases, one-sided external failures further bias NAIVE downward.  The saved contamination table includes a paired Monte-Carlo `NAIVE - ORACLE_SKIP` interval for each rho; all compared estimates use the same external-failure draw.  Attribution-aware methods are fixed in this synthetic setup because they are granted the latent attribution.  This directional result is expected by construction, so its value is only a sanity check that motivates—not validates—a later controlled interactive experiment.

## Interpretation and falsification attempt

The pilot does show a large generation–selection distinction in this cached ARC cohort.  But the key contamination result is structurally favorable to attribution awareness: generator-limited failures are exactly identifiable after labels are opened, and synthetic external failures are injected only into true positives.  It cannot establish a causal effect in a realistic online system.  The stability across symmetric Beta priors changes posterior shrinkage but not the sign of NAIVE's attribution error.

**Recommendation: WEAK GO.** The descriptive gap is strong enough to justify one larger controlled experiment, but the reliability simulation is partly tautological and n=30 is very small.  The exact next experiment should be a target-blind, synthetic *interactive* sandbox where generator, selector, executor, and environment failures are independently randomized and logged before outcome labels are revealed; compare learned attribution from observable telemetry with a pre-registered oracle-attribution upper bound.

## Limitations

1. ARC2 is not an interactive LLM-agent environment.
2. Its likelihood ranker/selector is only a proxy/motivation for future verifier research.
3. Generator-limited attribution is unusually clean here because exact candidate correctness is observable after frozen scoring.
4. Synthetic external failures are not estimates of real execution/environment failure prevalence.
5. n=30 is very small.
6. This pilot tests a conceptual hypothesis, not a final MPhil method.
7. No causal claim about general LLM agents follows from this Stage-0 pilot.
