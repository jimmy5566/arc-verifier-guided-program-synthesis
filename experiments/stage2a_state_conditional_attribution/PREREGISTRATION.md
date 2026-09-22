# Stage-2A preregistration

Frozen before results. CPU-only extension of Stage-1; it does not modify or rerun Stage-0/1.

## Observable states and true selector reliability

| State | P(state) | P(G) | P(V) | P(X) | P(E) | P(success) | Clean selector reliability P(not V | not G) |
|---|---:|---:|---:|---:|---:|---:|---:|
| A high | .30 | .10 | .072 | .14 | .14 | .548 | .92 |
| B medium | .25 | .25 | .1875 | .20 | .15 | .2125 | .75 |
| C low | .25 | .20 | .36 | .08 | .16 | .20 | .55 |
| D volatile | .20 | .05 | .1425 | .30 | .15 | .3575 | .85 |

One hidden primary source is sampled per episode. State is observable; source is
never a telemetry feature. A=high reliability, B=medium, C=low, D=high but with
executor/environment contamination. This deliberately makes naive outcome
updates locally distorted in different ways.

## Data, telemetry, models, and policy

8,000 labeled synthetic calibration episodes (seed 2201), 2,000 validation
episodes (2202, not used for tuning), and 12,000 held-out online test episodes
(2203). Attribution sees the Stage-1 noisy telemetry plus observable state
one-hot; it uses preregistered RandomForest (160 trees/min leaf 8). A degraded
attribution ablation mixes 50% learned probabilities with 50% calibration class
prevalence. No source/injection flag, intended action, or true transition is
observed.

Methods: Global-Naive; State-Naive; Oracle-State; Predicted-State; and
Predicted-State-Degraded. All use Beta(1,1). The policy commits when its current
state reliability estimate is >= .70; otherwise it calls a simulated strong
verifier (selector-correction probability .96, fixed cost .08). Strong verifier
cannot repair G, X, or E sources. Metrics: per-state calibration error, Brier/ECE
of selector-failure probability, wrong-commit rate, unnecessary-call rate,
episode success, strong-verifier cost, and attribution metrics.

Decision gate: GO if predicted state attribution beats state-naive calibration
and downstream quality in multiple states without one-feature leakage; WEAK GO
if gain is fragile; PIVOT if global/simple prevalence correction is comparable.
