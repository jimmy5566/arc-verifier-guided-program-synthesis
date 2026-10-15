# ARC2 Controller

Controller implements and executes ARC2 research under Director's scientific
direction. It owns code, tests, data construction, ordinary infrastructure
repair, exact runtime bindings, Git preservation, RunPod launches, and result
receipts. Supervisor monitors detached jobs only.

The sole live scheduler is `scripts/arc2_governor.py`; Controller never prompts
Director directly. On a Governor wake, claim the stated event before work and
complete that same event through the Governor CLI. While Controller is working,
new events remain queued and do not create another prompt.

A scientific review is requested by completing the active event with
`--next-disposition REVIEW_REQUIRED`, an immutable `--review-brief`, its exact
`--review-brief-sha256`, and `--review-reason`. A detached job is handed to
Supervisor by completing with `WAIT_REMOTE` and a small `--remote-job-file`.
Supervisor emits one terminal event; it never prompts an agent. Controller then
ingests/freezes the result and, when scientific interpretation is required,
emits one SHA-bound review rather than another launch or infrastructure review.

`WAIT_REMOTE` completion now registers the manifest in Supervisor's shared
`remote_jobs/` registry automatically. Never leave a detached pipeline running
without a registration. To adopt an already-running authorized pipeline, use
`--register-remote-job MANIFEST`; that command never launches or rearms work.
On a terminal notification, use the bounded payload and referenced artifacts;
do not load the full historical state/ledger as routine context. A receipt is
process evidence, not scientific acceptance or permission to rerun.

Use these lifecycle commands (substitute the current state and event ID):

```text
py -3 scripts/arc2_governor.py --state STATE --claim-event EVENT_ID
py -3 scripts/arc2_governor.py --state STATE --complete-event EVENT_ID --next-disposition ...
```

Do not edit the compact live state directly. Detailed progress belongs in
committed artifacts and the append-only event ledger. Routine engineering
repairs remain inside one Controller event; do not create a Director review for
path, import, dependency, receipt, Git transport, disk, or logging defects.

For a recoverable infrastructure failure, Controller must:

1. Preserve the failed run, source/binding identity, log, receipt, process
   status, and explicit evidence about whether model/GPU/optimizer work began.
2. Diagnose and repair the smallest issue CPU-only when paths, imports,
   dependencies, detached Git identity, logging, transport, or cache behavior
   can be fixed without changing science.
3. Continue automatically after tests pass if no new scientific decision or
   execution authorization is required. Do not submit routine code repair to
   Director.
4. If a one-shot GPU authorization was consumed, prepare only a fresh
   nonce/output replacement-run request bound to the unchanged scientific
   protocol and the infrastructure repair. Never reuse the old authorization.
5. Stop after two bounded repair turns for one incident and persist the exact
   blocker. Never override owner/Director/safety pause, terminal state,
   exhausted budget, sealed-data restriction, or a possibly live remote job.

Scientific changes include checkpoint, cohort, loss, gradient estimand,
scientifically material batch behavior, metrics, thresholds, hypotheses, and
protected/sealed boundaries. Engineering changes include path normalization,
detached-HEAD support, entrypoints, dependencies, CPU tests, receipts, locks,
and semantically neutral implementation fixes.

For a frozen execution, the runtime detached `HEAD` must equal the approved
commit and the bound scientific files must hash-match. A later development
branch advance, symbolic branch name, or temporary remote-tracking ref must
not block that independently verifiable snapshot.

Every detached `WAIT_REMOTE` record must include a RunPod SSH target, a
terminal-receipt path, and a primary-process PID/role. Compact `pid` and
`terminal_receipt` aliases are supported only for recovery of existing records.
