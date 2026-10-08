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
- After every completed or controlled-stop experiment, freeze/hash its small
  reproducibility artifacts, commit the intended source and artifact manifest,
  and push it to the configured GitHub branch before reporting it as
  preserved.  In this repository, a user request to "提交" means commit **and
  push**; a local-only commit is not a submission/preservation event.
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

## ARC2 file and artifact transfer policy

### Default: GitHub

Use GitHub for source code, configuration, protocols, manifests, scripts,
tests, reports, small JSON/CSV artifacts, orchestration code, and
reproducibility metadata. Prefer this sequence:

```text
local commit -> GitHub push -> RunPod git fetch -> checkout exact commit SHA
```

Do not use forced-PTY SSH transfer for these assets when GitHub can carry them.
Every remote execution records its exact Git commit SHA. Before scientific
execution where applicable, verify this provenance invariant:

```text
LOCAL HEAD = ORIGIN REF = RUNPOD CHECKOUT
```

### Large files: versioned artifact storage

Use Hugging Face Hub or another appropriate versioned artifact store for model
weights, LoRA adapters, large parquet datasets, checkpoints, and large binary
scientific artifacts. Record the repository/model/dataset ID, pinned revision
or commit, filename, expected SHA256, and actual downloaded SHA256. Never
silently substitute a different revision.

Do not put large binary files into ordinary Git history merely because GitHub
authentication is available. GitHub Release or LFS may be used when already
appropriate for the repository.

### Authentication

Authentication is environment-specific. The local Windows/Herdr environment
and the RunPod runtime may have different `GH_TOKEN` and `HF_TOKEN` presence.
Check the boolean only in the environment that performs the operation; a local
false result does not establish that RunPod authentication is absent. RunPod
may consume its own environment variables directly, but never transfer a token
between environments. Never print, echo, log, hash, serialize, commit, put in
remotes, paste into prompts, or copy token values into RunPod receipts.
Artifacts may record only `GH_TOKEN_PRESENT = true/false` and
`HF_TOKEN_PRESENT = true/false`, together with the environment they describe.

If GitHub or Hugging Face authentication is missing or invalid, report
`AUTH_MISSING_GITHUB` or `AUTH_MISSING_HUGGINGFACE` and identify the blocked
operation. Do not ask for a secret in chat or place one in a file. Existing
safe SSH Git authentication may be used when configured; do not rewrite
credentials unnecessarily.

### RunPod synchronization and verification

Synchronize code and small artifacts through GitHub with `git fetch` and an
exact-SHA checkout. Download large assets at an exact pinned revision from
Hugging Face or the selected versioned artifact store. Verify SHA256 after any
transfer or download before an asset becomes scientifically usable; a
successful download alone is not identity verification.

For each scientific round, preserve immutable provenance sufficient to answer:

- Which Git commit produced this result?
- Which model revision was loaded?
- Which dataset revision and file were used?
- What SHA256 identities were verified?
- Where did each artifact originate?

Filesystem paths are runtime locations, not scientific identity.

### Forced-PTY / SSH fallback

The RunPod gateway may force an interactive PTY. Treat PTY/SSH as the
command/control channel and as a fallback for small status queries, receipts,
or bounded recovery. Use it for large binary transfer only when GitHub is
unsuitable, versioned artifact storage is unavailable or unsuitable, and exact
bytes cannot otherwise be recovered.

When a PTY fallback is necessary, use byte-preserving encoding/chunking,
verify every chunk and the final SHA256, publish atomically, and record why the
fallback was required.

### Exact-byte scientific assets

For an exact historical scientific asset, the expected SHA256 is authoritative:
a filename, row count, or semantic similarity is insufficient. Require:

```text
SOURCE SHA256 = TRANSFERRED OR DOWNLOADED SHA256 = RUNTIME SHA256
```

Fail closed if exact-byte recovery fails, unless a new scientific subprotocol
explicitly authorizes logical reconstruction.

### Default transfer priority

For code, configuration, and small artifacts, use: GitHub; existing safe Git
SSH transport; then bounded SSH/PTY fallback.

For models, datasets, and other large binaries, use: Hugging Face or another
versioned artifact store; existing GitHub Release/LFS where appropriate;
existing persistent RunPod storage; then verified SSH/PTY fallback.

For the current RunPod workflow: GitHub is the primary code/protocol transport;
Hugging Face is the preferred large model/data artifact transport; SSH is the
command/control channel; Supervisor receipts are small status/control
artifacts; and forced-PTY binary transfer is fallback only.
