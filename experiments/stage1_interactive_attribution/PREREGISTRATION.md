# Stage-1 preregistration

**Frozen before implementation/results:** 2026-09-19.  This is a CPU-only
synthetic experiment and does not use ARC2 production execution, Kaggle, an LLM,
or a GPU.

## Hypotheses

- **H1:** Under heterogeneous hidden failure causes, telemetry-based predicted
  responsibility improves online selector-reliability estimation over treating
  every failed task as a selector error.
- **H2:** recovery of the oracle-attribution benefit varies with attribution
  quality, non-selector failure prevalence, feedback delay, and OOD shift.

## Environment and single-fault latent mechanism

Each sequential episode has 3--8 decisions. One latent episode label is drawn:
`success`, `generator`, `selector`, `executor`, or `environment`. At its hidden
failure step: generator omits the useful action; selector chooses a bad action
despite its presence; executor corrupts an appropriate committed action; or an
external transition defeats an otherwise appropriate action. At most one primary
failure is injected. Episode task outcome is revealed after the configured
delay. The latent label is never a telemetry feature.

## Frozen regimes and data

| Regime | success | generator | selector | executor | environment |
|---|---:|---:|---:|---:|---:|
| A balanced | .40 | .15 | .15 | .15 | .15 |
| B low-external | .50 | .15 | .25 | .05 | .05 |
| C high-non-selector | .35 | .20 | .10 | .15 | .20 |
| D selector-heavy | .40 | .10 | .30 | .10 | .10 |

For every in-distribution regime: 6,000 calibration-train, 2,000 validation,
and 5,000 held-out test episodes. Fixed seeds: A=1101, B=1201, C=1301, D=1401.
OOD trains on A and tests 5,000 C-regime episodes with telemetry-noise multiplier
1.45, seed 1501. No threshold or prevalence will be tuned after results.

## Observable telemetry and hidden variables

Observed, noisy features: step count, failure-step position, candidate count,
score margin, selector entropy, candidate disagreement, committed-action code,
executor status (OK/WARN/ERROR), execution latency, state-delta ratio,
expected-transition match, environment-warning indicator, and prior episode
failure rate. Source label, intended action, true transition, and injection
flags are hidden. Status/error features are deliberately overlapping: executor
errors can appear OK and environmental failures can emit WARN/ERROR.

## Models, baselines, and metrics

Attribution baselines: prevalence/majority; multinomial logistic regression;
RandomForest (160 trees, min leaf 8, fixed seed). RandomForest with all features
is the preregistered predicted-attribution model used for adaptation. Ablations:
selector-score only; execution only; environment only; all telemetry. A uniform
random attribution baseline is included in the leakage audit.

Reliability methods share Beta(1,1): STATIC, NAIVE outcome update, ORACLE
attribution, PREDICTED attribution (fractional class-probability evidence), and
CONFIDENCE-GATED predicted attribution (max class probability >= .70). Clean
target is `P(selector correct | non-generator episode)`. Primary reliability
metric is mean online absolute estimation error against that clean target;
secondary metrics are final error, Brier score of P(selector failure), and ECE.

Attribution: accuracy, macro-F1, per-class F1, selector-vs-other AUROC, selector
probability Brier/ECE, confusion matrix. Recovery ratio is
`(error_naive-error_predicted)/(error_naive-error_oracle)`.

## Delay, OOD, decision criteria

Delays are 0, 1, 3, 5 decision ticks and change when updates become available.
OOD is as defined above. We report the prescribed metrics, not post-hoc-selected
subgroups. `GO` requires predicted attribution to beat NAIVE across >1 regime,
retain a positive recovery ratio under OOD, and survive leakage ablations.
`WEAK GO` permits gains restricted to high-quality/narrow telemetry. `PIVOT`
applies if oracle has no benefit or gains require trivial leakage; otherwise
`INCONCLUSIVE`.
