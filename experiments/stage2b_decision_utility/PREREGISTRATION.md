# Stage-2B preregistration: decision-utility sensitivity

Frozen before execution. This is a new CPU-only sensitivity study that imports
the unchanged Stage-2A simulator functions. It does not modify or rerun Stage-0,
Stage-1, or Stage-2A.

## Fixed environment

The environment, single hidden primary-source assumption, observable states,
seeds, telemetry attribution model, and reliability update semantics are exactly
Stage-2A: calibration 8,000 (seed 2201), unused validation 2,000 (2202), and
held-out test 12,000 (2203). The RandomForest parameters are unchanged. Sources
remain unavailable to the predicted online updater.

The commit threshold is fixed at **0.70**, inherited from Stage-2A. It is not
tuned on validation or test in this sensitivity study.

## Frozen grid and policies

The full Cartesian grid is evaluated once:

- stronger-verifier repair probability: `[0.50, 0.70, 0.90, 1.00]`;
- stronger-verifier call cost: `[0.00, 0.02, 0.05, 0.10, 0.20, 0.30]`.

The compared policies are State-Naive, Predicted-Attribution,
Oracle-Attribution (an explicit non-deployable ideal-label comparator), Always-Strong, and
Never-Strong. The first three retain the unchanged Stage-2A update definitions.
Always/Never are simple verification baselines.

All policies use the same held-out repair random draw for a given episode and
repair probability. This controls Monte-Carlo noise in pairwise comparisons.

## Utility and decision rule

For an episode, `utility = task_success - (strong_call * verifier_cost)`.
Reported utility is the test-set mean. The strong verifier repairs only `V`
with the configured probability; it cannot repair `G`, `X`, or `E`. A wrong
commit is a committed episode with source `V`.

Primary gate: Predicted-Attribution must strictly exceed **all three**
deployable baselines (State-Naive, Always-Strong, Never-Strong) in at least one
non-tied grid cell to qualify for GO. WEAK GO applies only if this occurs in a
narrow or implausible regime. Otherwise PIVOT.
