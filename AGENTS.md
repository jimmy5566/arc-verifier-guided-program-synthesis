# ARC2 project rules

## Default mode

- **Single Codex agent** is the normal ARC2 development mode.
- Herdr, worktrees, and multi-agent roles are optional isolation tools, never
  mandatory gates for ordinary development.
- Use multi-agent work only when parallel value clearly exceeds coordination
  overhead. Historical material in `experiments/agent_system/` is optional.

## Experiment discipline

- Freeze scientific configuration before GPU inference; change one scientific
  variable at a time unless the written protocol says otherwise.
- Reuse valid frozen artifacts. A CPU-answerable question must not trigger a
  GPU rerun. Keep smoke studies small and do not expand automatically.
- Freeze predictions/candidate pools before opening targets or scoring.
- Explicit user authorization is required for GPU launch, a materially
  GPU-consuming Kaggle Save Version, push, destructive Git, production
  release, competition submission, or a release-time science change.

## Production and release

- Production packaging is a separate release phase. Scientific configuration
  may be frozen in advance; production **task identity may not** be frozen
  from a visible, development, evaluation, or historical cohort.
- At runtime, derive one manifest from the actually mounted challenge. The
  challenge hash, task IDs, test-index structure, queue, checkpoints,
  candidates, selector, finalizer, and submission must share that identity.
- Reject stale checkpoints/caches when any identity component changes; matching
  a task ID alone is insufficient.
- Before every competition submission, read and execute
  [`docs/RELEASE_CHECKLIST.md`](docs/RELEASE_CHECKLIST.md). Submission is
  forbidden unless its final status is `RELEASE_READY`.

## Evidence language

Distinguish confirmed defects, locally reproduced mechanisms,
development-set observations, public-LB observations, suspected hidden-run
mechanisms, and unverified hypotheses. Never promote suspicion to fact.
