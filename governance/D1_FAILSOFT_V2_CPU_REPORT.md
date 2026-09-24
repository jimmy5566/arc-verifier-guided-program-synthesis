# D1 fail-soft V2 CPU report

Date: 2026-09-24 (Australia/Adelaide)

## Scope

- Base: `777047996aee4733d4deae4837142d5a6663a718`
- Implementation source commit packaged by the candidate:
  `c4d5be2d2454babca03f07f04e9f2d73970604da`
- GPU/model runs: **0**
- Kaggle pushes: **0**
- Competition submissions: **0**
- Frozen release `governance/releases/d1-submitted-v1`: unchanged; its lock
  and all frozen source/notebook hashes pass.

## Shared runtime

Both the release candidate and governed experiment workbench use:

- `scripts/run_d1_failsoft_4gpu.py:run_live_failsoft`
- `src/inference/d1_failsoft_runtime.py:FailsoftRuntime`
- `src/inference/d1_failsoft_runtime.py:finalize_failsoft`

The shared state machine covers dual-source, both single-source directions,
completed-empty, recoverable task fallback, active-worker crash fallback, and
deadline fallback. Pending full-rerun outputs are initialized as challenge-
bound input copies; the normal-save `[[0]]` placeholder is removed before the
full-rerun path.

Global identity, challenge mapping, configuration, model-file, environment,
selector/finalizer and schema failures remain hard failures. Adapter/reset or
base-model integrity failures mark the worker unsafe rather than becoming a
normal single-source result.

## CPU verification

- Focused fail-soft/governance/release suite: **62 passed**.
- Full repository suite in this isolated Windows worktree: **467 passed, 1
  failed**. The sole failure is the pre-existing CRLF byte-hash mismatch in
  `test_track_u_source_has_no_downstream_or_gold_import_and_freeze_hashes_match`.
  The same exact test passes (**1 passed**) in the original shared checkout;
  no unrelated V3 source or assertion was changed.
- New fail-soft replay of frozen Eval60 evidence: **PASS**.
  - Top-1: **20/89**
  - Top-2: **28/89**
  - Pool oracle: **30/89**
  - Historical non-empty attempt mismatches: **0**
  - Versioned completed-empty input-copy differences: **1** (the historical
    development artifact stored null attempts for that empty output).

## Candidate identities

- Source archive SHA256:
  `50a3ad1ba6cb87e601bdf1f836621a23ac24a3415ca4bb6db652061e3081656b`
- Config SHA256:
  `8f68d5365223695086998861402c026e8f87b5d268d85664cdba70df93d74398`
- Notebook SHA256:
  `85e3fada1724da88fdd04cba40020da9706565edf708a61b6e277211f74341ad`
- Replay report SHA256:
  `d7591389266bc5126d56d6aef8c81e094ffb1ab17337aa6551e080d6ba79bb26`

This is an unpublished CPU-verified release candidate. Live CUDA execution
and full-run timing remain **NOT_RUN / UNVERIFIED**.

