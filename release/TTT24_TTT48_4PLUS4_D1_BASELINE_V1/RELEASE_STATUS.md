# D1 baseline release review

This is a CPU-only review gate. It is not a Kaggle run, a hidden-coverage result, or a competition submission.

- Baseline: `TTT24_TTT48_4PLUS4_D1_BASELINE_V1`
- D1 replay: Top-1 `20/89`; Top-2 `28/89`; pool oracle `30/89`.
- Remaining Kaggle GPU quota observed: `None`.
- CPU test suite: `14 passed`.
- Release status: **RELEASE_BLOCKED**.

## Blockers
- MODEL_STATE_PARITY_UNVERIFIED: failed TTT48 cross-score run had numerical anchor mismatches

## Required before any release re-evaluation

1. Establish the TTT48 numerical anchor parity using the same verified adaptation path; do not relax its tolerance.
2. Bind the verified live CUDA worker bootstrap into the D1 release image, then run a separate no-submission GPU parity gate.
3. Re-evaluate GPU quota only when a cost estimate and recovery reserve are supplied.
