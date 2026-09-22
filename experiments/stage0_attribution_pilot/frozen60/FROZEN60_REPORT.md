# Frozen60 primary Stage-0 report

## Artifact audit

Frozen60 uses the immutable local manifest, its cached 32-view candidate pool,
and frozen B-SUPPORT two-attempt predictions. The extractor validates all task
membership, freeze statuses, source SHA linkage, and 32-augmentation metadata
before opening the already-local training labels. No inference, GPU, model, or
Kaggle operation is used.

Frozen30 has 30 tasks and Frozen60 has 60; task overlap is
**0**. They are task-disjoint, but share project/model context and
are not claimed as fully independent statistical replications.

## Reproduced Frozen60 metrics

- Any-of-K: **30/60**
- Top-1: **16/60**
- Top-2: **21/60**

## Failure decomposition and Top-2 structure

| Category | Tasks | Rate |
|---|---:|---:|
| Success / Top-1 correct | 16 | 26.7% |
| Selection failure | 14 | 23.3% |
| Generator-limited failure | 30 | 50.0% |

Among 44 failures, **30/44 = 68.2%** are non-selection-responsible under this ARC taxonomy. Conditional selector accuracy is 16/30 = **53.3%**. Top-2 conditional retention is 21/30 = **70.0%**.

Of the 30 candidate-covered tasks: 16 are Top-1 correct, 5 are additional fixed Top-2 recoveries, and 9 retain a correct candidate below Top-2. Top-2 therefore adds limited but real portfolio evidence beyond the Any-of-K/Top-1 split; it does not resolve the remaining ranking or generator limitations.

## Beta(1,1) selector reliability

Clean target: `16/30 = 0.5333`. This is conditional on a correct cached candidate existing.

| Method | Posterior mean | 95% credible interval | Absolute error |
|---|---:|---:|---:|
| STATIC | 0.5000 | [0.0250, 0.9750] | 0.0333 |
| NAIVE | 0.2742 | [0.1715, 0.3907] | 0.2591 |
| ORACLE-SKIP | 0.5312 | [0.3603, 0.6985] | 0.0021 |
| ORACLE-CLEAN | 0.5312 | [0.3603, 0.6985] | 0.0021 |


## Frozen30 comparison

Frozen30 Top-1 is 9/30, higher than Frozen60's 16/60, while its conditional selector accuracy is 42.9%, lower than Frozen60's 53.3%. This supports the narrow claim that end-to-end task success is not the same quantity as selection-layer reliability: Frozen60 has more generator-limited failures while its selector does better when a correct candidate exists. The comparison is descriptive only.

## Frozen60 contamination sensitivity

At every rho, 1,000 fixed-seed trials evaluate reliability against clean latent selection correctness after flipping only genuine correct selections to observed external failures. The statistical threshold is rho=0.0; the 5pp practical heuristic is rho=0.0. Both are zero because the unmodified Frozen60 already has 30 generator-limited failures that Naive misattributes as selector errors. This is not an empirical estimate of real external failure rates.

## Critical interpretation and updated recommendation

Oracle-Skip necessarily beats Naive once generator-limited labels are known. Frozen60 therefore demonstrates a responsibility-label mismatch, not a solved online attribution mechanism. Oracle-Clean equals Oracle-Skip here because exact ARC candidate correctness makes the clean selection label available on exactly the same candidate-covered tasks: this is partly tautological. Frozen60 is larger and Top-2 adds five target-blind recoveries, but it strengthens motivation more than it proves the future interactive-agent hypothesis.

**WEAK GO.** Next, run a pre-registered CPU synthetic interactive sandbox with independently hidden generator, selector, executor, and environment failures. Compare telemetry-only attribution to an oracle upper bound and a naive outcome updater. Do not derive labels from test targets during that experiment.

## Limitations

ARC2 is not an interactive LLM-agent environment. Its ranker/selector is only a proxy for future verifier research. Generator limitation is unusually observable due to exact ARC scoring. Synthetic rhos are not real failure-rate estimates. n=60 is small. No causal claim about general LLM agents follows.
