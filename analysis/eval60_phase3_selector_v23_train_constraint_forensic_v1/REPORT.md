# Selector V2.3 train-derived deterministic constraint forensic

Classification: `TRAIN_CONSTRAINT_SIGNAL_NOT_SUPPORTED`.

## MEASURED

- CPU-only. No GPU, model call, DFS, TTT, candidate regeneration, likelihood recomputation, or S2 modification.
- Historical challenge SHA256: `e7c62a4bd211867c6b538f66b8013b81f299663c82ca062f49a52bf439d6e4e8`.
- Historical solutions SHA256: `84be4f4f39b79e82c36d565fc878830988b094917f052ee7069aef30b33ca8f1`.
- S2 reproduced exactly: Gold-rank parity 35/35 and Top2 27/35.
- Final-rank/grid uniqueness 35/35; train-pair order invariance 35/35.
- S2 miss outputs: 8. Informative outputs: 1/8.
- WIN / LOSS / TIE / UNINFORMATIVE = 1 / 0 / 0 / 7.
- 242 Gold-vs-ahead comparisons: 1 Gold-better, 0 Gold-worse, 241 ties.
- Only palette produced one discriminatory comparison. Shape, symmetry, bbox, and component families produced none.

## INFERRED

These coarse deterministic train-derived constraints are useful as sanity checks but have essentially no selector resolution inside the current dense S2 candidate pools.

## NOT ESTABLISHED

This does not rule out semantic transformation verification, executable-rule verification, learned verification, or other genuinely more discriminative evidence. It does not establish hidden-set generalization.

## Decision

Close this simple structural-constraint family unless genuinely new evidence appears. Do not modify S2 from this experiment.

The next planned CPU experiment is Consensus Repair Forensic. If that fails, perform Meta-Ranker Data Readiness before considering any new GPU work.
