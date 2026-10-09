# ARC2 simple orchestration

## Normal path

`scripts/arc2_governor.py` is the sole live persistent outer loop. The older
`scripts/arc2_runner.py` is deprecated and must not be run concurrently. Governor
recognizes `CONTINUE_CONTROLLER`, `REVIEW_REQUIRED`, `WAIT_REMOTE`, `PAUSED`,
and `TERMINAL`. It prompts the appropriate agent only after reading durable
workflow state; Director and Controller never directly wake one another.

`arc-controller` owns one active scientific stage from its first action until a
real terminal condition or a detached RunPod job.  Detailed progress belongs in
committed artifacts and Git history; [`ARC2_WORKFLOW_STATE_TEMPLATE.json`](ARC2_WORKFLOW_STATE_TEMPLATE.json)
defines the only durable workflow fields.

The live state is local (`.arc2-local/orchestration/ARC2_WORKFLOW_STATE.json`)
and contains the current stage, a coarse status, the active remote job if one
exists, last completed action, next action, Director-review flag, and terminal
flag.  It does not encode remediation substeps or acknowledgement cycles.

For a short high-level decision the Controller freezes one stage brief and
calls Director synchronously:

```text
herdr agent prompt arc-director "<review request>" --wait \
  --until idle --until done --until blocked --timeout <milliseconds>
```

The Controller then verifies the returned review references the brief hash and
continues within the same workflow.  Current Director decisions include `CONTINUE_CONTROLLER`, `CONTINUE_DIRECTOR`,
`REQUIRE_CHANGES`, `PAUSED`, and `TERMINAL`; legacy scientific outcomes are
retained for already-frozen reviews. The Governor consumes each bound response
once and routes by the reviewed stage/next action.

Historical `DIRECTOR_DIRECTIVE_*`, responses, escalations, and the former
directive-cycle state file are immutable provenance.  They are deprecated for
new normal-path scheduling and must not be deleted or treated as live work
queues.

## Recoverable infrastructure failures

When a detached launcher fails before model work, Controller first performs
bounded CPU-only diagnosis, preserves immutable failure evidence, and never
infers a scientific result. An authorized one-shot GPU run does not authorize
an automatic second launch. If a retry needs new authorization, sealed input
bytes are missing, or scientific conditions must change, Controller freezes a
short SHA-bound review brief and sets `REVIEW_REQUIRED` for Governor to invoke
Director. This is a stage-scoped incident, not an automatic research-program
stop.

For premature Controller-written `PAUSED` with an explicit
`infra_failure_class`, Governor may requeue up to two **CPU-only** turns,
first to diagnose and then to prepare the Director handoff. No GPU work is
permitted by this requeue. Owner, Director, and safety pauses retain priority;
an unresolved incident after the bounded requeue remains visibly paused with a
specific blocker.

## Remote jobs

The sole normal asynchronous path is a detached RunPod job.  The Controller
writes its immutable job/round contract and changes the live workflow status to
`WAITING_REMOTE_JOB`.  The Supervisor reconciles small remote receipts every
300 seconds, emits heartbeats while the job runs, and wakes the Controller once
only for terminal or actionable failure receipts.  It does not call Director,
choose curriculum, or interpret scientific results.

The Watchdog only keeps exactly one Supervisor alive.  If the live workflow
state marks `experiment_terminal: true`, Watchdog exits quietly rather than
restarting it.

## Evidence reuse

Use the frozen evidence registry for unchanged artifacts.  Reverify an entry
only when its bytes, Git provenance, remote asset, or scientific scope changes.
The current registry is
`experiments/foundation_v2_reconstruction_and_targeted_repair_v2/round_009_equivalence_evaluation/ARC2_FROZEN_EVIDENCE_REGISTRY_V1.json`.
