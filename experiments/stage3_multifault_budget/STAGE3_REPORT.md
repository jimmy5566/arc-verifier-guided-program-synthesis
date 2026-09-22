# Stage-3: multi-fault responsibility under intervention budgets

## Scope and prevalence

Stage-3 is isolated and CPU-only. No prior stage was modified. It tests the
Stage-2C one-cause simplification directly: one episode can have no fault, one
fault, or two simultaneous faults from `G,V,X,E`; no triple fault is allowed.

The exact ID preregistered state/category distributions are in
`PREREGISTRATION.md`. The realized 15,000-episode ID prevalence was **none
.337, single .387, dual .276**. The un-retrained OOD dual-shift prevalence was
**none .197, single .318, dual .485**.

## Exact intervention effects and costs

Repair probabilities per active component are intentionally overlapping:

| Action | G | V | X | E | clean retention |
|---|---:|---:|---:|---:|---:|
| COMMIT | .00 | .00 | .00 | .00 | .990 |
| REGENERATE | .70 | .20 | .15 | .15 | .980 |
| STRONG_VERIFY | .18 | .75 | .10 | .12 | .990 |
| TOOL_RETRY | .12 | .12 | .70 | .42 | .985 |
| REOBSERVE_REPLAN | .10 | .12 | .40 | .75 | .985 |

For an active dual fault, all active components must be repaired; sequence
repair probabilities compound. `V_STRONG` changes only V repair by
STRONG_VERIFY from .75 to .92. Costs are:

| Regime | COMMIT | REGENERATE | STRONG_VERIFY | TOOL_RETRY | REOBSERVE_REPLAN |
|---|---:|---:|---:|---:|---:|
| LOW | .00 | .03 | .08 | .05 | .07 |
| MEDIUM | .00 | .06 | .16 | .10 | .12 |
| HIGH | .00 | .12 | .28 | .18 | .22 |

Budget is at most one or two distinct non-commit actions. Utility is exactly
`task_success - sum(action_cost)`.

## Multi-label attribution metrics

| Scenario | Component | AUROC | F1 | Brier | ECE |
|---|---|---:|---:|---:|---:|
| ID | G | .928 | .723 | .102 | .095 |
| ID | V | .937 | .761 | .093 | .075 |
| ID | X | .965 | .808 | .072 | .062 |
| ID | E | .933 | .730 | .100 | .090 |
| ID aggregate | macro/micro F1 | .756 / .756 | exact match .621 | mean Brier .092 | mean ECE .080 |
| OOD aggregate | macro/micro F1 | .777 / .778 | exact match .541 | mean Brier .112 | mean ECE .090 |

The OOD exact-match reduction is expected because dual faults are much more
common; component-level discrimination remains high without OOD retraining.

## Budget-1 and budget-2 results: ID BASE effects, MEDIUM costs

| Policy | B1 success | B1 utility | B1 wrong action | B2 success | B2 utility | B2 wrong action |
|---|---:|---:|---:|---:|---:|---:|
| Commit-only | .333 | .333 | .000 | .333 | .333 | .000 |
| Always strong verify | .460 | .300 | .422 | .460 | .300 | .422 |
| Uncertainty-only | .369 | .336 | .063 | .372 | .337 | .062 |
| Failure-risk-only | .455 | .349 | .410 | .611 | .446 | .107 |
| State-only | .474 | .406 | .366 | .606 | .446 | .108 |
| Attribution-aware | .585 | **.519** | .096 | .730 | **.589** | .022 |
| Shuffled attribution | .429 | .362 | .250 | .515 | .374 | .166 |
| Oracle multi-label | .644 | .586 | .000 | .772 | .668 | .000 |

At budget 1, attribution-aware beats Uncertainty-Only by **+.182** utility and
the best simple baseline by **+.113**; Oracle-gap recovery is **62.6%**. At
budget 2 those figures rise to **+.252**, **+.143**, and **64.6%**.

Responsibility becomes behaviorally useful rather than only predictive: under
budget 1, attribution-aware uses matching actions for active G/V/X/E at rates
`.70/.29/.72/.26`, respectively. V and E are deliberately less clean because
the action effects overlap and dual-fault context makes a G/X action sometimes
have higher expected overall value. The full action-by-component matrix is in
`outputs/component_action_confusion.csv`.

## Cost, effect, and budget sensitivity

Attribution-aware is the best deployable policy in **all 12** frozen ID
combinations (BASE or V_STRONG effects × LOW/MEDIUM/HIGH cost × budget 1/2).

- Its utility range is `.466–.567` at budget 1 and `.484–.681` at budget 2.
- Its advantage over Uncertainty-Only is `.154–.208` at budget 1 and
  `.173–.319` at budget 2.
- Its advantage over the strongest simple baseline is `.113–.143` at budget 1
  and `.130–.160` at budget 2.
- Oracle-gap recovery is `59.1%–66.7%` across all 12 cells.

The strong V intervention ablation does not collapse attribution value: with
V_STRONG and CHEAP_STRONG, utility is .531 (B1) and .612 (B2), still ahead of
the frozen simple policies.

## OOD dual-fault shift

The preregistered OOD condition increases dual faults to .485 and changes fault
combinations, without retraining. At BASE effects, MEDIUM costs, budget 2:

| Policy | Utility | Wrong intervention | Mean cost |
|---|---:|---:|---:|
| Commit-only | .195 | .000 | .000 |
| Always strong verify | .158 | .465 | .160 |
| Uncertainty-only | .184 | .094 | .056 |
| Failure-risk-only | .305 | .077 | .202 |
| State-only | .338 | .092 | .160 |
| Attribution-aware | **.471** | **.017** | .164 |
| Shuffled attribution | .270 | .167 | .164 |
| Oracle multi-label | .561 | .000 | .143 |

Attribution-aware is **+.287** over Uncertainty-Only and **+.133** over the
strongest simple baseline; it recovers **59.6%** of the Oracle gap.

## Decision: GO

**GO.** Explicit marginal responsibility remains decision-useful with ambiguous
single/dual faults, partial overlapping interventions, imperfect attribution,
and finite budgets. It beats both uncertainty-only and every strongest simple
baseline in multiple cost/effectiveness/budget regimes, including the held-out
dual-fault OOD shift.

**Most important support:** budget-2 ID MEDIUM utility .589 versus .337 for
Uncertainty-Only and .446 for the best simple policy; OOD remains .471 versus
.184 and .338, respectively. Shuffling the same probability/action-rate
marginals collapses B2 utility to .374, establishing that episode-specific
component responsibility—not intervention volume—drives the gain.

**Most important threat:** this is still a bounded, independent-repair,
at-most-dual-fault simulator with informative telemetry. The full-to-Oracle gap
remains substantial (.067 B1 and .079 B2 under ID MEDIUM); responsibility
uncertainty and correlated/feedback failures may be materially harder.

## Research implication and next experiment

The proposal can now claim a limited but concrete result: component-level
responsibility estimates are useful for **budgeted, cost-aware partial repair**,
not solely for one-cause routing. The next experiment should not increase model
complexity. It should add a capped adaptive observation budget or correlated
dual-fault repair effects, preregister a robustness sweep, and test whether
sequential re-observation reduces the remaining Oracle gap without label
leakage.

## Reproducibility

The full specification is in `PREREGISTRATION.md`. Outputs are
`policy_results.csv`, `multilabel_attribution_metrics.csv`,
`component_action_confusion.csv`, and figures `utility_by_policy.png`,
`cost_regime_map.png`, `intervention_confusion_matrix.png`,
`oracle_recovery.png`, and `ood_results.png`. The data-flow audit is
`LEAKAGE_AND_TRIVIALITY_AUDIT.md`.
