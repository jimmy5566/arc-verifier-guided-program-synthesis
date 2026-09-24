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
- A new experiment normally means a new resolved config and isolated run
  directory under the shared workbench, not a copied inference implementation.
- Experiment and release entry identities are distinct. Experiment launchers
  never FAST_SAVE, create the official top-level `submission.json`, or submit.
- Test targets are evaluation-only and remain outside staged inference bundles.
- Artifact reuse requires stage-compatible provenance; missing provenance is
  `UNKNOWN`, never implicit reuse. Cached retrieval does not increase support.

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
- Frozen release snapshots are immutable. Promotion packages the approved
  experiment's exact core code/config; it does not reimplement generation or
  scoring and never publishes or submits automatically.

## Evidence language

Distinguish confirmed defects, locally reproduced mechanisms, development-set
observations, public-LB observations, suspected hidden-run mechanisms, and
unverified hypotheses. Never promote a historical suspicion to fact.

- CPU mocks, frozen replay, and real GPU execution are different evidence.
- Report missing checks as `NOT_RUN` or `UNVERIFIED`; do not relax tolerances,
  scientific settings, or fallback policy merely to make a gate pass.
