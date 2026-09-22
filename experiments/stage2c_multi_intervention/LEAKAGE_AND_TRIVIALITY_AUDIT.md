# Stage-2C leakage and triviality audit

## Runtime separation

The hidden source is generated separately from the Stage-2A observable
telemetry. The deployed `ATTRIBUTION_AWARE` branch receives only the full
classifier probability matrix estimated from telemetry. It does not receive the
latent source. The source is accessed only to simulate the selected action's
environmental outcome, construct offline metrics, and form the explicitly
non-deployable `ORACLE_ATTRIBUTION` comparator.

The full attribution model is trained on labelled **calibration** episodes only.
No validation or held-out labels choose thresholds, intervention effects, costs,
or policy rules. The OOD model is not retrained.

## Non-triviality ablations

The ID BASE policy comparison includes all of the following:

- raw uncertainty routing using only margin, entropy, and disagreement;
- a generic binary failure-risk router;
- state/context-only responsibilities;
- shuffled full responsibility probabilities, with identical action-frequency
  marginal distribution but no episode-level responsibility alignment;
- degraded probabilities (`50%` full telemetry probability plus `50%`
  calibration prior);
- ideal labels only as the non-deployable oracle comparator.

The strongest ordinary ablation is State-Only at utility 0.513. Full
Attribution-Aware reaches 0.675; shuffled attribution falls to 0.426. Thus the
gain depends on per-episode responsibility alignment, not merely action-mix,
observable state, or generic uncertainty.

## Scope limitation

The conditional effect matrix intentionally favors matched component actions,
as a multi-intervention sandbox must, but it retains partial cross-effects and
non-deterministic outcomes. The experiment has one primary cause per episode
and relatively informative synthetic telemetry. It therefore establishes a
decision-value mechanism, not a real-world performance claim or a multi-fault
solution.
