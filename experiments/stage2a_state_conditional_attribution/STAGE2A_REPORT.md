# Stage-2A: state-conditional attribution and downstream decisions

## Scope and design

This is an isolated CPU-only synthetic extension of Stage-1. Stage-0 and
Stage-1 files were not modified. Each of 12,000 held-out episodes has one hidden
primary source: selector/verifier (`V`), generation (`G`), executor (`X`),
environment (`E`), or success. The source is unavailable to the online
updater. The updater sees only noisy telemetry plus one observable context
state. There are no ARC targets, GPUs, LLMs, or Kaggle calls.

| State | True selector reliability P(not V | not G) | P(G) | P(V) | P(X) | P(E) |
|---|---:|---:|---:|---:|---:|
| A high | 0.92 | 0.100 | 0.0720 | 0.140 | 0.140 |
| B medium | 0.75 | 0.250 | 0.1875 | 0.200 | 0.150 |
| C low | 0.55 | 0.200 | 0.3600 | 0.080 | 0.160 |
| D volatile | 0.85 | 0.050 | 0.1425 | 0.300 | 0.150 |

`G`, `X`, and `E` unevenly contaminate observed outcomes, so treating every
episode failure as a local verifier failure is intentionally wrong.

The frozen policy commits at estimated reliability >= 0.70; otherwise it calls
a stronger verifier. The strong verifier corrects `V` with probability 0.96,
cannot repair other primary sources, and costs 0.08 per call. This fixed policy
is intentionally separate from attribution training.

## Attribution quality

The held-out full telemetry classifier reached accuracy **0.755**, macro-F1
**0.705**, selector-failure F1 **0.840**, selector AUROC **0.979**, Brier
**0.048**, and ECE **0.055**. Its state-only audit reached macro-F1 **0.256**
and selector-F1 **0.415**, so the result is not produced by state prevalence
alone. See `LEAKAGE_AND_TRIVIALITY_AUDIT.md` for the data-flow audit.

## Per-state reliability calibration and policy outcome

| State | Method | Final estimate | Abs. reliability error | Task success | Strong-call cost | Wrong commit |
|---|---|---:|---:|---:|---:|---:|
| A | State-naive | 0.554 | 0.366 | 0.622 | 0.0800 | 0.000 |
| A | Oracle attribution | 0.920 | 0.000 | 0.554 | 0.0000 | 0.072 |
| A | Predicted attribution | 0.900 | 0.020 | 0.554 | 0.0001 | 0.072 |
| A | Degraded attribution | 0.789 | 0.131 | 0.554 | 0.0001 | 0.072 |
| B | State-naive | 0.220 | 0.530 | 0.392 | 0.0800 | 0.000 |
| B | Oracle attribution | 0.762 | 0.012 | 0.224 | 0.0018 | 0.175 |
| B | Predicted attribution | 0.772 | 0.022 | 0.224 | 0.0014 | 0.175 |
| B | Degraded attribution | 0.716 | 0.034 | 0.227 | 0.0027 | 0.172 |
| C | State-naive | 0.192 | 0.358 | 0.535 | 0.0800 | 0.000 |
| C | Oracle attribution | 0.550 | 0.000 | 0.535 | 0.0800 | 0.000 |
| C | Predicted attribution | 0.595 | 0.045 | 0.535 | 0.0800 | 0.000 |
| C | Degraded attribution | 0.627 | 0.077 | 0.535 | 0.0800 | 0.000 |
| D | State-naive | 0.349 | 0.501 | 0.488 | 0.0800 | 0.000 |
| D | Oracle attribution | 0.848 | 0.002 | 0.348 | 0.0001 | 0.144 |
| D | Predicted attribution | 0.844 | 0.006 | 0.348 | 0.0001 | 0.144 |
| D | Degraded attribution | 0.761 | 0.089 | 0.348 | 0.0002 | 0.144 |

Selector Brier/ECE for predicted attribution by state A/B/C/D are respectively
0.028/0.034, 0.052/0.057, 0.076/0.076, and 0.039/0.056. Oracle attribution has
zero attribution-probability error by construction and is only an upper bound.

Predicted attribution recovers **87.5%--99.2%** of the calibration-error gap
between State-naive and Oracle in every state. Deliberately degrading
attribution worsens the final estimate in A (0.131 error), C (0.077), and D
(0.089), demonstrating an attribution-quality-to-calibration relation rather
than a state-only effect.

## Aggregate downstream result

| Method | Weighted abs. reliability error | Task success | Strong verifier cost | Cost-adjusted success |
|---|---:|---:|---:|---:|
| Global naive | 0.431 | 0.516 | 0.0800 | 0.436 |
| State naive | 0.431 | 0.516 | 0.0800 | 0.436 |
| Oracle state attribution | 0.003 | 0.427 | 0.0211 | 0.406 |
| Predicted state attribution | 0.024 | 0.427 | 0.0210 | 0.406 |
| Predicted degraded attribution | 0.085 | 0.428 | 0.0214 | 0.407 |

The important negative result is deliberate and informative. Under this fixed
policy, the naive methods become very pessimistic and call the strong verifier
on every episode. Because that verifier repairs most `V` failures and cost is
only 0.08, the conservative methods achieve higher task success and higher
cost-adjusted success even though their reliability estimates are badly wrong.
Better episode-level attribution therefore improves **calibration** but does
not improve this particular downstream policy's task-success objective. It
instead trades correctness for a large verifier-cost reduction.

## Decision gate: PIVOT

**PIVOT.** Predicted attribution is clearly valuable for state-conditional
calibration, but the required critical test fails: it does not improve future
decision quality under the preregistered commit-or-strong-verifier policy. A
simple conservative update is currently downstream-superior because it calls a
cheap, effective stronger verifier everywhere. This is not a leakage claim or
a failure of the attribution metric; it is a policy/utility mismatch exposed by
the experiment.

- **Strongest supporting result:** predicted attribution reduces weighted
  reliability error from 0.431 to 0.024 and recovers at least 87.5% of the
  State-naive-to-Oracle gap in each state without hidden-source inputs.
- **Strongest threatening result:** State-naive task success is 0.516 versus
  0.427 for predicted attribution; cost-adjusted success is also higher
  (0.436 versus 0.406).

## Recommended next experiment

Run a small Stage-2B policy-only study on a **separate, preregistered**
validation/test split: choose a realistic strong-verifier correction rate and
cost before testing, then compare fixed expected-utility thresholds. Keep the
same single-fault simulator and attribution model. Do not add multi-faults,
LLMs, or deep models until attribution has an objective-aligned downstream
policy benefit.

## Reproducibility

Outputs are `outputs/attribution_metrics.csv`,
`outputs/state_calibration_and_decisions.csv`,
`outputs/aggregate_decision_results.csv`, `outputs/recovery_summary.json`, and
`outputs/state_calibration_and_decisions.png`. Seeds, distributions, model, and
threshold are frozen in `PREREGISTRATION.md`.
