# Governance V1 CPU verification

Run date: 2026-09-24 (Australia/Adelaide)

No model was loaded, no CUDA code was run, and no Kaggle mutation was made.

## Governance and release contracts

- Focused governance + existing D1/release suite: **54 passed**.
- The focused suite exercises unset and false experiment rerun dispatch with
  an injected solver, true rerun rejection, CPU-only validate/prepare/dry-run,
  target separation, order/hash preservation, malformed inputs, immutable run
  identities, reuse decisions, support preservation, release hashes, thin
  shared-solver wiring, same-identity resume/config mutation rejection, and
  the no-submission experiment notebook contract.
- All 16 tracked governance JSON files parse; both JSON Schemas validate; the
  locally staged notebook compiles.
- Frozen package bytes and actual pulled notebook bytes match the hashes in
  `release.lock.json`.
- The authoritative solver/finalizer/contract/selector have no diff from
  wrapper commit `141a1a852b6fcfca612f35cea627ba4812326c21`.

## Frozen D1 evidence replay

`scripts/replay_d1_release_contract.py` was run against the existing local,
unmodified Eval60 evidence through the current
`pool_for_output -> select_record -> finalize` contract.

- Status: **PASS**
- Outputs: **89**
- Top-1: **20/89**
- Top-2: **28/89**
- Pool oracle: **30/89**
- Candidate membership/support/RRF mismatches: **0**
- Non-empty attempt mismatches: **0**
- Ephemeral replay report SHA256:
  `f1ce717d9e42535eff8be04f6d794080eac96eeea47c6b7cf7899a8e8527ef77`

This remains retrospective exposed-development evidence, not independent
validation, live-model parity, or leaderboard evidence.

## Full repository CPU suite

- Latest full pass: **455 passed, 2 failed** in the isolated governance
  worktree.
- One failure was the timing-sensitive existing dynamic-scheduler assertion;
  its immediate isolated rerun passed (**1 passed**), without a code change.
- The remaining failure is the pre-existing V3 byte-hash test
  `test_track_u_source_has_no_downstream_or_gold_import_and_freeze_hashes_match`.
  The isolated Windows worktree checked those four unchanged text files out as
  CRLF, while the frozen hashes cover LF bytes. The same test passes in the
  original shared worktree. Governance V1 does not rewrite those unrelated V3
  files or weaken the frozen-byte assertion.
- Historical local data/artifact fixtures were exposed to the isolated
  worktree only through temporary read-only junctions for this test run; the
  junctions were removed and are not part of the commit.

## Evidence boundaries

- GPU/model validation: **NOT_RUN** (forbidden by this task).
- Experiment notebook publication: **NOT_RUN**.
- Kaggle run/dataset publication/submission: **NOT_RUN**.
- Full live model parity and runtime: **UNVERIFIED** and remains a documented
  release limitation.
