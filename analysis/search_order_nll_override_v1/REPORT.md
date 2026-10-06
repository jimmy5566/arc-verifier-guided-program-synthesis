# Conservative cumulative-NLL override audit v1

Frozen source: `2da4ef598cc8e792ddab5b9f9d85036e1b0792ce`  
Preregistered analysis: `16c460d61c5130248e64060b44f91aa8e9f728df`

## Result

Across 64,005 true Gold-opportunity frontier states, cumulative NLL remains the strongest robust anchor:

- Top-1 = 32.91%
- Top-2 = 48.43%
- Top-5 = 80.72%
- MRR = 0.5204

The preregistered task-LOOTO trust classifier is weak: ROC-AUC for NLL Top-1 correctness is 0.5791. Under the frozen conservative policy, an override is allowed only when P(NLL Top1 correct) < 0.20, then a relative pairwise model selects among NLL ranks 2..5.

The hybrid produces:

- Top-1 = 28.75% (delta -4.16pp)
- Top-2 = 53.47% (delta +5.04pp)
- Top-5 = 80.72% (unchanged)
- MRR = 0.5116 (delta -0.0088)
- overrides = 22,793
- FIX = 3,651
- HARM = 6,311
- catastrophic tasks = 1

Task-bootstrap 95% CI:
- Delta Top-1: [-0.1316, +0.0341]
- Delta MRR: [-0.0574, +0.0331]

The policy improves Top-2 coverage but damages the primary Top-1 decision because the trust model cannot reliably identify when NLL should be overridden. HARM substantially exceeds FIX.

## Decision

**DO NOT RUN GPU NLL OVERRIDE.**

The next causal experiment should preserve the strongest observed signal and test **global cumulative-NLL best-first traversal** as the live retained-frontier policy. That experiment changes only retained-node pop order and must be preregistered separately.
