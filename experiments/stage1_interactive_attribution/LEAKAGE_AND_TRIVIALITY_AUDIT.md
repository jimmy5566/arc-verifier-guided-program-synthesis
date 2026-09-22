# Leakage and triviality audit

## What the attribution model cannot see

The telemetry matrix has 13 observed fields: decision count/position, candidate
count, score margin, selector entropy/disagreement, committed-action code,
executor status, latency, state change, expected-transition match, environment
warning, and lagged history. It contains no source label, injection flag,
intended action, true transition, or direct `executor_failed` feature.

Executor status is intentionally non-deterministic: executor failures can emit
OK, and environment failures can emit WARN/ERROR. The class label is only used
as an offline supervised target on the separate simulator calibration split and
as a held-out evaluation label; it is never included among runtime features.

## Empirical checks

- Best single-feature multinomial-logistic macro-F1: **0.273** (state-delta
  ratio), far below all-telemetry RandomForest macro-F1 **0.696** on balanced
  held-out data.
- Executor-telemetry-only RandomForest macro-F1: **0.461**.
- Removing the apparent executor-status feature from all telemetry leaves
  macro-F1 **0.664** (versus **0.696**) and selector F1 **0.805** (versus
  **0.809**). Thus no single raw status code is a near-perfect label decoder,
  although multiple simulator correlations remain informative.
- Uniform attribution is not competitive: macro-F1 **0.112**, selector AUROC
  **0.500**, selector F1 **0**.
- OOD A-to-noisy-C macro-F1 falls to **0.580** and selector F1 to **0.564**;
  attribution is not invariant to changed prevalence/telemetry reliability.

## Falsification outcome

The test rejects the narrow claim that one obvious telemetry feature perfectly
leaks the latent source. It does **not** establish realism. Predicted
adaptation recovers a strikingly large share of Oracle gain even under mediocre
multi-class macro-F1 in some ablations. This is a warning: aggregate
reliability can be helped by coarse class-probability calibration without
solving per-episode responsibility, and the simulator may still be too
structured. The experiment demonstrates an internally valid possibility, not
that real LLM-agent telemetry has this signal.
