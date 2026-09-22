# Stage-3 preregistration: ambiguous multi-fault responsibility under budget

Frozen before execution. Stage-3 is a CPU-only synthetic experiment and does
not modify Stage-0, Stage-1, Stage-2A, Stage-2B, or Stage-2C.

## Latent multi-fault environment

Each episode has either no fault, exactly one fault, or exactly two faults from
`G,V,X,E`. No triple faults occur. State is observable but the fault set is not.
State probabilities are A/B/C/D = `(.25,.25,.30,.20)`. Each state uses this
category distribution, in order `none,G,V,X,E,GV,GX,GE,VX,VE,XE`:

| State | none | G | V | X | E | GV | GX | GE | VX | VE | XE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | .50 | .08 | .06 | .07 | .07 | .04 | .03 | .03 | .03 | .03 | .06 |
| B | .32 | .10 | .10 | .10 | .10 | .05 | .05 | .05 | .04 | .04 | .05 |
| C | .25 | .12 | .15 | .07 | .10 | .08 | .04 | .04 | .04 | .05 | .06 |
| D | .30 | .06 | .08 | .16 | .10 | .03 | .05 | .04 | .06 | .05 | .07 |

This yields prevalence: **none .338, single .382, dual .280**. Telemetry is a
noisy function of the active fault vector and observable state: candidate count,
margin, entropy, disagreement, executor status, latency, transition change,
environment warning, and history. It contains no raw fault bit or category.

Data split: calibration 10,000 seed 3101; unused validation 2,500 seed 3102;
held-out ID test 15,000 seed 3103. Full and state-only component classifiers are
four independent RandomForest classifiers (160 trees, min leaf 10, seed
3101+component), trained only on calibration labels. The generic risk classifier
is binary any-fault only. No threshold or rule is tuned on validation/test.

## Actions, partial repair probabilities, and costs

Rows are actions and columns `G,V,X,E`; numbers are per-active-component repair
probabilities. If two components are active, both must be repaired for the task
to succeed. For a sequence, unrepaired probabilities compound independently.

| Action | G | V | X | E | clean retention |
|---|---:|---:|---:|---:|---:|
| COMMIT | .00 | .00 | .00 | .00 | .990 |
| REGENERATE | .70 | .20 | .15 | .15 | .980 |
| STRONG_VERIFY | .18 | .75 | .10 | .12 | .990 |
| TOOL_RETRY | .12 | .12 | .70 | .42 | .985 |
| REOBSERVE_REPLAN | .10 | .12 | .40 | .75 | .985 |

The `V_STRONG` effectiveness sensitivity changes only STRONG_VERIFY's V repair
probability `.75 -> .92`. Cost regimes (COMMIT, REGENERATE, STRONG_VERIFY,
TOOL_RETRY, REOBSERVE_REPLAN) are:

| Regime | Costs |
|---|---|
| LOW | `(.00,.03,.08,.05,.07)` |
| MEDIUM | `(.00,.06,.16,.10,.12)` |
| HIGH | `(.00,.12,.28,.18,.22)` |

Utility per episode is exactly `task_success - sum(action_cost)`. A budget is
the maximum number of distinct non-commit actions: **1** or **2**. Actions may
not be repeated; additional actions can partially repair a remaining fault.

## Fixed policies

- **COMMIT_ONLY:** empty action sequence.
- **ALWAYS_STRONG_VERIFY:** STRONG_VERIFY once, even under budget 2.
- **UNCERTAINTY_ONLY:** raw `(1-margin + entropy/1.3 + disagreement)/3`; at
  >=.58 use STRONG_VERIFY; under budget 2 and >=.78 add TOOL_RETRY.
- **FAILURE_RISK_ONLY:** RF P(any fault); at >=.45 use STRONG_VERIFY; under
  budget 2 and >=.70 add TOOL_RETRY.
- **STATE_ONLY:** state-only four marginal probabilities and the same expected
  utility sequence maximisation as attribution-aware.
- **ATTRIBUTION_AWARE:** full predicted marginal `P(G),P(V),P(X),P(E)` and
  expected utility sequence maximisation.
- **SHUFFLED_ATTRIBUTION:** fixed seed-3106 permutation of full probability
  vectors, same expected utility rule.
- **ORACLE_MULTI_LABEL:** true multi-hot fault vector with the same rule; this
  is a non-deployable comparator only.

For any marginal vector p and action sequence with component repair q, expected
success is `clean_retention(sequence) * product_z((1-p_z)+p_z*q_z)`, then costs
are subtracted. The attribution policies may choose no action, one action, or
two actions subject to budget. This does not expose the joint latent category.

## OOD condition and metrics

`OOD_DUAL_SHIFT` uses state distribution `(.15,.20,.35,.30)` and category rows:

| State | none | G | V | X | E | GV | GX | GE | VX | VE | XE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | .30 | .07 | .07 | .07 | .07 | .08 | .06 | .06 | .06 | .06 | .10 |
| B | .20 | .08 | .10 | .09 | .08 | .08 | .08 | .08 | .07 | .06 | .08 |
| C | .15 | .07 | .13 | .06 | .07 | .15 | .06 | .06 | .07 | .08 | .10 |
| D | .20 | .04 | .06 | .14 | .08 | .05 | .08 | .06 | .10 | .08 | .11 |

OOD therefore has more dual faults and shifted combinations. It uses the same
conditional telemetry and effects; no model is retrained on OOD labels.

Metrics: per-component AUROC/F1/Brier/ECE, macro/micro F1, multi-label exact
match, success, utility, cost, action frequency, wrong/unnecessary intervention,
action-by-component usage, regret versus Oracle, attribution-minus-uncertainty,
and oracle-gap recovery relative to the strongest simple baseline.

## Decision gate

**GO:** Attribution-Aware beats Uncertainty-Only and the strongest simple
baseline in at least two ID cost/budget/effectiveness regimes and has positive
gain on OOD budget-2 MEDIUM without leakage.

**WEAK GO:** benefit survives but is limited to budget 2, narrow costs, or falls
below .02 utility on OOD.

**PIVOT:** simple routing is within .01 utility in almost all conditions or
attribution collapses under dual-fault ambiguity.
