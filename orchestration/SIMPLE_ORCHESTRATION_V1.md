# ARC2 simple orchestration

## Normal path

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
continues within the same workflow.  Normal review decisions are `CONTINUE`,
`CONTINUE_WITH_WARNING`, `REQUIRE_CHANGES`, `NEW_SUBPROTOCOL_REQUIRED`,
`PAUSE`, and `STOP`.  Only `PAUSE` and `STOP` end ordinary execution.

Historical `DIRECTOR_DIRECTIVE_*`, responses, escalations, and the former
directive-cycle state file are immutable provenance.  They are deprecated for
new normal-path scheduling and must not be deleted or treated as live work
queues.

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
