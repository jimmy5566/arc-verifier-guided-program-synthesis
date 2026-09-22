# Stage-2B: stronger-verifier utility sensitivity

## Scope and frozen design

Stage-2B is a new CPU-only study. It imports the unchanged Stage-2A synthetic
single-fault generator and telemetry attribution model; it does not modify
Stage-0, Stage-1, or Stage-2A. The held-out test contains 12,000 episodes. The
Stage-2A calibration and validation splits are reproduced, but the validation
split is not used: the commit threshold remains fixed at 0.70 for every policy
and every cell.

The predeclared grid crosses repair probability `{0.50, 0.70, 0.90, 1.00}`
with verifier-call cost `{0.00, 0.02, 0.05, 0.10, 0.20, 0.30}`. Compared methods
are State-Naive, Predicted-Attribution, Oracle-Attribution (ideal-label
comparator only), Always-Strong, and Never-Strong. All policies use the same
latent repair draw within each grid cell.

## Utility definition

`utility = task_success - (strong_verifier_called * verifier_cost)`

Reported utility is the held-out mean. The strong verifier repairs only a
selector/verifier failure (`V`) with the configured probability; it cannot
repair generation (`G`), executor (`X`), or environment (`E`) failures. A wrong
commit is a committed `V` episode.

## 2D regime map: best deployable policy

`N/A` means State-Naive and Always-Strong tie; `P` is Predicted-Attribution and
`Never` is Never-Strong.

| Repair probability / Cost | 0.00 | 0.02 | 0.05 | 0.10 | 0.20 | 0.30 |
|---:|---|---|---|---|---|---|
| 0.50 | N/A | N/A | N/A | **P** | Never | Never |
| 0.70 | N/A | N/A | N/A | **P** | **P** | Never |
| 0.90 | N/A | N/A | N/A | N/A | **P** | **P** |
| 1.00 | N/A | N/A | N/A | N/A | **P** | **P** |

Predicted attribution is strictly best in **7/24** frozen regimes. It is not
best when verification is free/cheap, nor when verification is so costly that
never verifying wins. The rendered map is `outputs/utility_regime_map.png`;
all per-cell metrics are in `outputs/policy_sensitivity.csv`.

## Representative exact results

### Strongest attribution region: repair 1.00, cost 0.20

| Policy | Task success | Wrong commit | Strong-call rate | Cost | Utility |
|---|---:|---:|---:|---:|---:|
| State-Naive | 0.524 | 0.000 | 1.000 | 0.200 | 0.324 |
| Predicted-Attribution | 0.431 | 0.093 | 0.263 | 0.053 | **0.378** |
| Oracle-Attribution | 0.431 | 0.093 | 0.264 | 0.053 | 0.378 |
| Always-Strong | 0.524 | 0.000 | 1.000 | 0.200 | 0.324 |
| Never-Strong | 0.338 | 0.186 | 0.000 | 0.000 | 0.338 |

Predicted attribution exceeds the strongest simple baseline by **0.0406**
utility in this cell, mostly by reducing verification calls.

### Weakest attribution region: repair 0.50, cost 0.00

| Policy | Task success | Wrong commit | Strong-call rate | Cost | Utility |
|---|---:|---:|---:|---:|---:|
| State-Naive / Always-Strong | 0.431 | 0.000 | 1.000 | 0.000 | **0.431** |
| Predicted-Attribution | 0.385 | 0.093 | 0.263 | 0.000 | 0.385 |
| Oracle-Attribution | 0.385 | 0.093 | 0.264 | 0.000 | 0.385 |
| Never-Strong | 0.338 | 0.186 | 0.000 | 0.000 | 0.338 |

When stronger verification is free, conservative verification dominates; there
is no economic reason to commit based on a calibrated local estimate.

## Oracle-benefit recovery

In the 10 cells where Oracle-Attribution has a positive utility increment over
State-Naive, Predicted-Attribution recovers **100.2%--100.6%** of that
increment. Values slightly above 100% do not mean predicted attribution is more
accurate than ideal source labels: soft predicted updates cross a fixed online
threshold on slightly different episodes than hard oracle updates. The oracle
is an ideal-label comparator, not a guaranteed utility upper bound.

Where Oracle's utility is not positive versus State-Naive, recovery is reported
as undefined rather than assigning a misleading ratio.

## Decision: GO

**GO.** Predicted attribution has a meaningful, predeclared 7/24 region where
it improves the same cost-adjusted utility over all deployable baselines:
State-Naive, Always-Strong, and Never-Strong. The Stage-2A PIVOT was therefore
not evidence that attribution is intrinsically useless. It was the consequence
of a cheap/effective verifier, where calling it on every episode was rational.

The benefit remains economic-policy dependent. The proposal should position
attribution as a method for allocating expensive verification, not as a
universal replacement for verification. The next experiment should use a
separate preregistered split and realistic fixed repair/cost estimates to assess
one expected-utility policy; do not yet add multi-faults, LLMs, or deep models.

## Reproducibility

No target information was used for selection or threshold tuning. Tests:
**2 passed**. Outputs include `policy_sensitivity.csv`, `oracle_recovery.csv`,
`regime_map.csv`, `utility_regime_map.png`, and `run_metadata.json`.
