# ARC2 dual-agent RunPod orchestrator

This branch is rooted exactly at frozen Foundation-V2 alignment commit
`b26e469d1bd8d757727d4f34aff0e967207d1e62`. It provides infrastructure only.
No training, inference, or scientific acceptance is performed by these files.

## Roles

One Herdr workspace, `ARC2`, hosts the logical roles:

- `arc-director`: low-frequency Codex scientific reviewer; directive-only writes.
- `arc-controller`: primary Codex research/execution agent.
- `arc-supervisor`: deterministic Python receipt reconciler.
- `arc-monitor`: optional read-only status pane.
- `arc-watchdog`: bounded-restart PowerShell process watchdog.

## Safety properties

- Preservation stash refs are recorded in `bootstrap/PRE_ORCHESTRATOR_PRESERVATION.json`.
- Remote terminal states are process facts, never scientific acceptance.
- Controller notification identity is the terminal-receipt SHA-256; restart cannot duplicate it.
- Director invocation is milestone/escalation driven, never round-by-round.
- The wrapper uses detached `nohup setsid` execution and atomically publishes a receipt.

## Controller handoff

After all smoke checks are recorded PASS, the Controller prepares
`TARGETED_CAPABILITY_REPAIR_V1`, submits it via the structured Director
directive/response protocol, and waits for `CONTINUE` or
`CONTINUE_WITH_WARNING` before any real GPU training.

## Completed smoke evidence

- Remote round `901`: detached 65-second shell sleep, `SUCCESS` receipt and
  event after local SSH monitoring was disconnected and reconnected.
- Remote round `902`: deliberate exit 23, `TRAIN_FAILED` receipt and event;
  it was never treated as scientific success.
- Supervisor: first reconciliation emitted two notifications, and a restart
  emitted none (the receipt hash is durable state).
- Director: live Director produced the dummy milestone `CONTINUE` directive
  and the dummy `REQUIRE_INFRA_REPAIR` escalation directive; Controller
  acknowledgement files exist for both. Both directives prohibit training.
- Watchdog: a non-scientific process failed once, was restarted once, and the
  durable state ended `STOPPED_CLEANLY`.
