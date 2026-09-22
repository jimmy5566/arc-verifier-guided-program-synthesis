# Stage-1 interactive attribution report

## Design

This preregistered CPU-only simulator uses 3--8 sequential decisions per
episode and one hidden primary source: success, generator, selector, executor,
or environment. Generator omits the useful candidate; selector commits a bad
candidate despite availability; executor corrupts the appropriate commit; and
environment defeats an otherwise correct transition. Runtime attribution sees
only noisy telemetry. No GPU, LLM, Kaggle resource, or ARC2 production path was
used.

Four frozen prevalence regimes are in `PREREGISTRATION.md`: balanced A,
low-external B, high-non-selector C, and selector-heavy D. Each has 6,000
calibration-train and 5,000 held-out test episodes. OOD trains on A and tests
noisy C telemetry. Delay conditions are 0/1/3/5 decisions.

## Attribution quality

RandomForest all-telemetry results on held-out ID tests:

| Regime | Accuracy | Macro-F1 | Selector F1 | Selector AUROC |
|---|---:|---:|---:|---:|
| A balanced | .749 | .690 | .810 | .977 |
| B low-external | .839 | .676 | .878 | .983 |
| C high-non-selector | .730 | .692 | .766 | .976 |
| D selector-heavy | .801 | .676 | .885 | .978 |
| OOD A -> noisy C | .629 | .580 | .564 | .921 |

Balanced per-class F1 is success .930, generator .655, selector .810,
executor .534, environment .521. The balanced confusion matrix is stored in
`outputs/attribution_metrics.csv` and rendered in the output figure. Important
confusions remain executor/environment and generator/executor rather than a
perfect class decoder.

## Reliability adaptation (mean online absolute error, delay=0)

| Regime | Static | Naive | Oracle | Predicted | Confidence-gated |
|---|---:|---:|---:|---:|---:|
| A | .3193 | .4222 | .0050 | .0086 | .0176 |
| B | .2101 | .2068 | .0072 | .0341 | .0405 |
| C | .3801 | .5291 | .0080 | .0298 | .0176 |
| D | .1685 | .2616 | .0088 | .0434 | .0162 |
| OOD A -> noisy C | .3798 | .5336 | .0047 | .0751 | .0548 |

Predicted attribution is better than Naive in all four regimes and OOD. The
exception that matters is B: low external contamination leaves Naive slightly
better than Static, so the result does not support a claim that updating is
always harmful; the benefit depends on heterogeneous non-selector failures.

## Oracle-benefit recovery

Using `(error_naive-error_predicted)/(error_naive-error_oracle)`, ID recovery
is A .991, B .865, C .958, D .863; OOD recovery is .867. This is a strong
within-simulator result, but it should be interpreted together with the
triviality warning below. The ablations show macro-F1/recovery pairs:
selector scores .436/.968; execution .461/.978; environment .349/.928; all
.696/.993; all except executor status .664/.995. There is no clean empirical
threshold in this simulator at which macro-F1 suddenly becomes useful:
aggregate reliability benefit remains high even when multi-class attribution is
only modest. That weakens the intended claim that high per-episode attribution
accuracy is necessary.

## Delay and shift

In balanced episodes, Predicted mean online error is .0086 at delays 0 and 1,
.0087 at 3, and .0088 at 5. Delay has only a small effect because episodes are
long and the global estimator aggregates 5,000 decisions; it is not a strong
stress test. OOD reduces macro-F1 from .690 to .580 and increases Predicted
error from .0086 to .0751, while retaining a positive advantage over Naive.

## Interpretation and decision

**Decision: WEAK GO.** The strongest supporting observation is that noisy,
non-single-feature telemetry predicts enough responsibility to materially
improve reliability adaptation over Naive across multiple heterogeneous
regimes, including the preregistered OOD condition. The strongest threat is
that recovery is near-oracle even when multi-class attribution is weak, and
delay barely matters. This indicates that the global Beta objective may reward
aggregate class calibration rather than genuine per-episode responsibility
inference; the simulator is still too easy/structured for a general agent
claim.

## Recommended Stage-2

Construct a partially observed sequential simulator with multi-fault episodes,
state-dependent nonstationary transition dynamics, and sparse/ambiguous outcome
telemetry. Train attribution only from delayed, noisy responsibility audits
(not simulator labels for every calibration sample). Pre-register a per-episode
decision criterion and evaluate whether predicted attribution remains useful
when aggregate source prevalence is matched but individual responsibility is
harder to infer.

## Limits

This is not ARC2 production or a general LLM-agent environment. Latent labels
are simulator-defined; calibration supervision and telemetry distributions are
synthetic. Oracle remains an upper bound, not a deployable method. See
`LEAKAGE_AND_TRIVIALITY_AUDIT.md` for the required adverse evidence.
