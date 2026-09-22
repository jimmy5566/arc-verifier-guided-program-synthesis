# ARC2 project rules

## Default development mode

- **Single Codex agent** is the default. Herdr, worktrees, and multi-agent
  roles are optional isolation tools, not ordinary workflow gates.
- A single Codex may inspect and edit the current repository, run CPU tests
  and static checks, create experiment scripts, analyse frozen artifacts, and
  prepare Kaggle packages.

## Experiment discipline

- Freeze scientific configuration before GPU inference. Prefer one variable at
  a time, reuse valid frozen artifacts, and keep smoke studies small.
- CPU-answerable questions must remain CPU-only. Freeze predictions and
  candidate pools before opening targets. Never expand an experiment
  automatically.
- Explicit user authorization is required for GPU launch, a materially
  GPU-consuming Kaggle Save Version, push, destructive Git, production release,
  competition submission, or a release-time science change.

## Production and submission

- Production packaging is a distinct release phase. Freeze science if needed,
  but **never freeze production task identity** from a visible, development,
  evaluation, historical, or cached cohort.
- At runtime, construct one manifest from the mounted challenge. Its content
  hash, task/test-index structure, queue, checkpoints, candidates, selector,
  finalizer, and submission must share one identity. A matching task ID alone
  never validates cache reuse.
- Before any competition submission, read and execute
  [`docs/RELEASE_CHECKLIST.md`](docs/RELEASE_CHECKLIST.md). Submission is
  forbidden unless that checklist ends `RELEASE_READY`.

## Evidence language

Distinguish confirmed defects, locally reproduced mechanisms, development-set
observations, public-LB observations, suspected hidden-run mechanisms, and
unverified hypotheses. Never promote a historical suspicion to fact.
