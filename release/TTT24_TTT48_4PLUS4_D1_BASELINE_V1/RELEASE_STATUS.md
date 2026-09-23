# D1 baseline release review

This is a CPU-only review gate. It is not a Kaggle run, a hidden-coverage result, or a competition submission.

- Baseline: `TTT24_TTT48_4PLUS4_D1_BASELINE_V1`
- D1 replay: Top-1 `20/89`; Top-2 `28/89`; pool oracle `30/89`.
- Remaining Kaggle GPU quota observed: `GPU 1.19h remaining as observed 2026-09-24`.
- CPU test suite: `429 passed, 3 xfailed, 53.38s (CPU only)`.
- Release status: **RELEASE_BLOCKED**.

## Blockers
- RERUN_PATH_NOT_EQUIVALENT: production builder retains a fast-commit/dummy branch
- RUNTIME_IDENTITY_NOT_DYNAMIC: runner requires a pre-frozen 240-task manifest
- CHECKPOINT_IDENTITY_NOT_BOUND_TO_RUNTIME_CHALLENGE_CONTENT_AND_TEST_SHAPE
- AMBIGUOUS_INPUT_DISCOVERY: builder uses first rglob match for source archive
- MODEL_STATE_PARITY_UNVERIFIED: failed TTT48 cross-score run had numerical anchor mismatches
- QUOTA_BLOCKED: remaining quota is below a fully costed 4xL4 adaptation/generation/rescoring run plus recovery reserve

## Required before any release re-evaluation

1. Establish the TTT48 numerical anchor parity using the same verified adaptation path; do not relax its tolerance.
2. Replace the fast-commit/rerun split with one runtime-challenge-derived, fail-closed production entry path.
3. Bind checkpoints to challenge content and test-index structure, then pass failure-injection tests.
4. Re-evaluate GPU quota only after those CPU/model-state gates pass.
