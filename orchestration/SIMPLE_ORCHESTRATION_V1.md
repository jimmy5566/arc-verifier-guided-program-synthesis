# ARC2 simple orchestration

## Normal path

`scripts/arc2_governor.py --daemon` is the sole persistent outer loop. It owns a process-lifetime lock and service heartbeat, recognizes
`CONTINUE_CONTROLLER`, `REVIEW_REQUIRED`, `WAIT_REMOTE`, `PAUSED`, and `TERMINAL`, and uses a dispatch lease to prevent duplicate Controller prompts.
`PAUSED` remains observable and restartable; only `TERMINAL` exits. `scripts/arc2_runner.py` is disabled historical provenance and must not run concurrently.

`arc-controller` owns one active scientific stage from its first action until a
real terminal condition or a detached RunPod job.  Detailed progress belongs in
committed artifacts and Git history; [`ARC2_WORKFLOW_STATE_TEMPLATE.json`](ARC2_WORKFLOW_STATE_TEMPLATE.json)
defines the only durable workflow fields.

The live state is local (`.arc2-local/orchestration/ARC2_WORKFLOW_STATE.json`)
and contains the current stage, disposition, the active remote job if one
exists, last completed action, next action, Director-review flag, and terminal
flag.  It does not encode remediation substeps or acknowledgement cycles.

For a scientific decision the Controller freezes one stage brief and returns
`REVIEW_REQUIRED`. Governor calls Director synchronously:

```text
herdr agent prompt arc-director "<review request>" --wait \
  --until idle --until done --until blocked --timeout <milliseconds>
```

Governor verifies that the returned review references the brief hash and
continues within the same workflow. Normal live decisions are
`CONTINUE_CONTROLLER`, `CONTINUE_DIRECTOR`, `REQUIRE_CHANGES`, `PAUSED`, and
`TERMINAL`.

Routine infrastructure failures return to Controller for at most two bounded
CPU-only repair turns. Successful semantically neutral repair continues without
Director. A consumed one-shot GPU authorization cannot be reused; Controller
then freezes one concise replacement-run brief and Governor routes only that
authorization decision.

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

The Supervisor Watchdog only keeps exactly one Supervisor alive. If the live
workflow state marks `experiment_terminal: true`, it exits quietly rather than
restarting it. Separately, the existing Governor process host checks the
Governor daemon's PID and command line, restarts only that daemon, and after
bounded repeated host failures wakes Controller for CPU-only repair. Neither
watchdog interprets science, contacts Director, or launches remote work.

## Evidence reuse

Use the frozen evidence registry for unchanged artifacts.  Reverify an entry
only when its bytes, Git provenance, remote asset, or scientific scope changes.
The current registry is
`experiments/foundation_v2_reconstruction_and_targeted_repair_v2/round_009_equivalence_evaluation/ARC2_FROZEN_EVIDENCE_REGISTRY_V1.json`.
