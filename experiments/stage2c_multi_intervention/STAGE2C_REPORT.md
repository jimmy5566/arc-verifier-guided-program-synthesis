# Stage-2C: multi-intervention responsibility routing

## Scope

This is an isolated CPU-only synthetic experiment. It imports Stage-2A's
unchanged single-latent-source state/telemetry environment and writes new output
only in this directory. Stage-0, Stage-1, Stage-2A, and Stage-2B are unchanged.
No GPU, LLM, Kaggle resource, ARC data, or production code is accessed.

## Exact effectiveness matrix

Each entry is the noisy probability that the final task succeeds after the
chosen action, conditional on the hidden source.

| Action | success | G | V | X | E |
|---|---:|---:|---:|---:|---:|
| COMMIT | .99 | .03 | .03 | .03 | .03 |
| REGENERATE | .96 | .82 | .12 | .10 | .12 |
| STRONG_VERIFY | .98 | .18 | .84 | .08 | .10 |
| TOOL_RETRY | .97 | .12 | .10 | .78 | .40 |
| REOBSERVE_REPLAN | .97 | .10 | .10 | .38 | .82 |

The `V_STRONG` sensitivity changes only `STRONG_VERIFY -> V` from `.84` to
`.97`. All values and the three cost regimes were preregistered before running:

| Cost regime | COMMIT | REGENERATE | STRONG_VERIFY | TOOL_RETRY | REOBSERVE_REPLAN |
|---|---:|---:|---:|---:|---:|
| BASE | .00 | .05 | .15 | .08 | .10 |
| CHEAP_STRONG | .00 | .05 | .04 | .08 | .10 |
| HIGH_INTERVENTION | .00 | .12 | .24 | .18 | .22 |

The fixed utility is `task_success - intervention_cost` per episode. The 0.70
Stage-2A reliability threshold is not used by the responsibility router, and
the uncertainty/failure-risk thresholds were frozen in preregistration.

## Attribution quality

| Scenario | Full macro-F1 | Full selector F1 | Full selector AUROC | State-only macro-F1 | State-only selector F1 |
|---|---:|---:|---:|---:|---:|
| ID | .705 | .840 | .979 | .256 | .415 |
| OOD prevalence, no retraining | .703 | .869 | .975 | .225 | .485 |

The binary generic failure-risk model has AUROC .989 on ID and .989 on OOD for
any-failure detection. Thus generic risk is strong, but does not identify which
intervention is appropriate.

## ID BASE policy results

| Policy | Success | Mean cost | Utility | Wrong intervention | Unnecessary intervention | Regret vs Oracle |
|---|---:|---:|---:|---:|---:|---:|
| Commit only | .354 | .000 | .354 | .662 | .000 | .455 |
| Always strong verify | .542 | .150 | .392 | .476 | .338 | .418 |
| Uncertainty only | .505 | .042 | .463 | .485 | .015 | .346 |
| Failure risk only | .544 | .100 | .444 | .477 | .026 | .365 |
| State only | .604 | .091 | .513 | .407 | .338 | .297 |
| Attribution aware | .747 | .072 | **.675** | **.214** | .131 | **.135** |
| Attribution degraded | .737 | .086 | .651 | .222 | .338 | .159 |
| Shuffled attribution | .498 | .072 | .426 | .532 | .268 | .384 |
| Oracle attribution | .874 | .064 | .810 | .000 | .000 | .000 |

Attribution-aware utility exceeds Uncertainty-Only by **+0.212** and the
strongest non-attribution baseline (State-Only) by **+0.162**. It recovers
**54.7%** of the Oracle-versus-best-simple utility gap.

The base action-by-cause matrix gives the decision mechanism directly. For true
G/V/X/E, Attribution-Aware chooses its matching action with rates
**.77 / .81 / .66 / .43**, respectively. Uncertainty-Only can only choose
COMMIT or STRONG_VERIFY: it commits on .82 of G, .87 of X, and .76 of E cases.
See `outputs/intervention_confusion_matrix.png` and
`outputs/intervention_confusion.csv`.

## Sensitivity and OOD

Attribution-Aware is best among the simple/full deployable policies in every
one of the six preregistered ID cost/effectiveness combinations:

| Effects | BASE cost | CHEAP_STRONG | HIGH_INTERVENTION |
|---|---:|---:|---:|
| BASE effects | .675 | .698 | .613 |
| V_STRONG effects | .698 | .720 | .636 |

These are Attribution-Aware utilities. Its advantage over Uncertainty-Only is
between **+0.172** and **+0.212** across these six cells; its Oracle recovery is
between **54.2%** and **56.5%**. Thus the benefit survives both cheaper strong
verification and a substantially stronger V-specific intervention.

In the preregistered OOD prevalence shift, with no retraining:

| Policy | Utility | Wrong intervention | Regret vs Oracle |
|---|---:|---:|---:|
| Uncertainty only | .397 | .518 | .366 |
| Failure risk only | .384 | .502 | .380 |
| State only | .430 | .486 | .334 |
| Attribution aware | **.610** | **.249** | **.153** |
| Oracle attribution | .764 | .000 | .000 |

The OOD attribution-aware gain over Uncertainty-Only is **+0.213**, with
**54.0%** Oracle-gap recovery.

## Decision: GO

**GO.** Explicit responsibility attribution beats uncertainty-only, generic
failure-risk, state-only, always-strong, and commit-only routing in multiple
cost/effectiveness conditions and under the held-out OOD prevalence shift. The
shuffled-ablation collapse confirms that the gain needs episode-specific
responsibility alignment rather than merely a favorable action distribution.

**Strongest supporting evidence:** ID BASE utility .675 versus .463 for
Uncertainty-Only, coupled with a wrong-intervention reduction from .485 to
.214; OOD preserves a +.213 utility advantage.

**Strongest evidence against overclaiming:** this is a single-primary-fault
environment with an intentionally component-specialized effect matrix and
fairly informative telemetry. Degrading attribution only lowers ID utility by
.024, while the full-to-Oracle gap remains .135. The numerical magnitude should
not be projected to multi-fault or real-agent systems.

## Meaning for the proposal and next experiment

The proposal is now justified as **cost-aware intervention selection from
component responsibility**, not merely verifier reliability estimation. The
next experiment should be a small preregistered Stage-3 realism check with two
co-occurring but independently observable failures, capped intervention budget,
and unchanged no-LLM/CPU-only boundaries. It should test whether the Stage-2C
gain survives responsibility ambiguity rather than expanding model complexity.

## Reproducibility

Full per-policy/per-regime results are in `outputs/policy_results.csv`.
Rendered figures are `utility_by_policy.png`, `cost_regime_map.png`,
`intervention_confusion_matrix.png`, `oracle_recovery.png`, and
`ood_results.png`. The leakage and triviality audit is in
`LEAKAGE_AND_TRIVIALITY_AUDIT.md`.
