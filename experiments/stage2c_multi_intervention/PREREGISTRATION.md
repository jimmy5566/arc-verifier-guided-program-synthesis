# Stage-2C preregistration: multi-intervention responsibility routing

Frozen before execution. Stage-2C is a new CPU-only synthetic experiment. It
imports, but does not alter, the Stage-2A environment functions. It does not
modify Stage-0, Stage-1, Stage-2A, or Stage-2B.

## Environment, labels, and splits

One hidden primary source is sampled per episode: success, generator/planner
failure (`G`), verifier/selector failure (`V`), executor/tool failure (`X`), or
environment failure (`E`). The observable state design and ID source
prevalences are exactly Stage-2A:

| State | P(state) | P(success) | P(G) | P(V) | P(X) | P(E) |
|---|---:|---:|---:|---:|---:|---:|
| A high | .30 | .5480 | .10 | .0720 | .14 | .14 |
| B medium | .25 | .2125 | .25 | .1875 | .20 | .15 |
| C low | .25 | .2000 | .20 | .3600 | .08 | .16 |
| D volatile | .20 | .3575 | .05 | .1425 | .30 | .15 |

Telemetry is the unchanged Stage-2A noisy observable telemetry and context.
The latent source is never a runtime feature. The model is the unchanged
Stage-2A RandomForest (160 trees, minimum leaf 8, balanced subsample): 8,000
calibration episodes (seed 2201), 2,000 validation episodes (seed 2202), and
12,000 held-out ID test episodes (seed 2203). Validation is materialized only;
no policy threshold is tuned on validation or test.

## Interventions and effectiveness

Rows are actions and columns are the hidden source. Each entry is the noisy
Bernoulli probability that the final task succeeds after choosing that action.
All entries, including success retention, are intentionally non-deterministic.

| Action | success | G | V | X | E |
|---|---:|---:|---:|---:|---:|
| COMMIT | .99 | .03 | .03 | .03 | .03 |
| REGENERATE | .96 | .82 | .12 | .10 | .12 |
| STRONG_VERIFY | .98 | .18 | .84 | .08 | .10 |
| TOOL_RETRY | .97 | .12 | .10 | .78 | .40 |
| REOBSERVE_REPLAN | .97 | .10 | .10 | .38 | .82 |

The effectiveness sensitivity condition `V_STRONG` changes only
`STRONG_VERIFY -> V` from `.84` to `.97`.

## Costs and utility

Rows use action order COMMIT, REGENERATE, STRONG_VERIFY, TOOL_RETRY,
REOBSERVE_REPLAN.

| Cost regime | COMMIT | REGENERATE | STRONG_VERIFY | TOOL_RETRY | REOBSERVE_REPLAN |
|---|---:|---:|---:|---:|---:|
| BASE | .00 | .05 | .15 | .08 | .10 |
| CHEAP_STRONG | .00 | .05 | .04 | .08 | .10 |
| HIGH_INTERVENTION | .00 | .12 | .24 | .18 | .22 |

For each episode, fixed utility is `task_success - intervention_cost`. There
is no post-hoc rescaling.

## Policies and fixed rules

1. **COMMIT_ONLY:** always COMMIT.
2. **ALWAYS_STRONG_VERIFY:** always STRONG_VERIFY.
3. **UNCERTAINTY_ONLY:** from raw observable margin, entropy, and disagreement,
   calculate `((1-margin) + entropy/1.2 + disagreement)/3`; choose
   STRONG_VERIFY at score >= .60, otherwise COMMIT.
4. **FAILURE_RISK_ONLY:** an RF trained only on `P(any failure)` from telemetry;
   choose STRONG_VERIFY at risk >= .45, otherwise COMMIT.
5. **STATE_ONLY:** RF using only observable state one-hot and the same
   responsibility expected-utility rule; an ablation, not a generic baseline.
6. **ATTRIBUTION_AWARE:** full `P(success,G,V,X,E | telemetry)` and choose the
   action with highest expected success probability minus action cost.
7. **ATTRIBUTION_DEGRADED:** `.5 * full_probability + .5 * calibration_prior`;
   otherwise identical to Attribution-Aware.
8. **SHUFFLED_ATTRIBUTION:** a fixed seed-2206 permutation of full probability
   vectors; otherwise identical to Attribution-Aware.
9. **ORACLE_ATTRIBUTION:** hidden one-hot source at decision time; a
   non-deployable comparator only.

The expected-utility rule is
`sum_z P(z | telemetry) * effectiveness[action,z] - cost[action]`.
No threshold is separately optimized by method or regime.

## OOD condition and metrics

`OOD_PREVALENCE` changes only state/source prevalence, not per-cause telemetry
or effectiveness. Its state distribution is `(.15,.20,.35,.30)` and source
distributions for A/B/C/D are respectively `(.45,.10,.13,.16,.16)`,
`(.15,.20,.28,.22,.15)`, `(.12,.16,.45,.10,.17)`, and
`(.20,.06,.23,.30,.21)`. OOD is sampled from a large Stage-2A conditional
telemetry pool; no model is retrained on OOD labels.

Metrics: task success, mean utility, mean intervention cost, action frequency,
wrong-intervention rate, unnecessary-intervention rate, action-by-cause
confusion, regret versus Oracle, attribution-aware gain over Uncertainty-Only,
and oracle-recovery ratio relative to the strongest non-attribution baseline.

## Decision gate

**GO:** Attribution-Aware beats Uncertainty-Only and all simple deployable
baselines in at least two ID cost/effectiveness conditions, has positive utility
gain on OOD BASE, and no leakage/triviality check fails.

**WEAK GO:** benefit exists only in one condition, vanishes on OOD, or requires
the full attribution model to be near oracle.

**PIVOT:** Uncertainty-Only or another simple baseline is within .01 utility of
Attribution-Aware in almost every reasonable condition, or Attribution-Aware
does not improve intervention choice.
