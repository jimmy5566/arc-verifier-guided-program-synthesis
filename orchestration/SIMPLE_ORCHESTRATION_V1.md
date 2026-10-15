# ARC2 thin orchestration

## Roles

- Controller owns implementation, execution, routine repair, artifact freeze,
  Git preservation, and result ingestion.
- Director synchronously reviews one SHA-bound scientific brief and returns one
  structured decision.
- Supervisor monitors detached remote work and writes one immutable
  `REMOTE_TERMINAL` event. It never prompts Controller or Director.
- Governor is a deterministic event router. It does not poll RunPod, interpret
  science, repair infrastructure, or manage authorization.

Owner-assigned direct Director execution uses `DIRECTOR_TASK`, not a fabricated
scientific review brief. A registered long job may specify
`completion_recipient=director`; Supervisor then emits one Director continuation
on every mechanical terminal status, including failure and UNKNOWN. Director
claims/completes the same event and yields with WAIT_REMOTE for further long
work. This is a status/task handoff, not an extra approval layer or Agent.

## Event lifecycle

Each event has a content-derived ID and exactly four states:
`QUEUED -> DELIVERED -> CLAIMED -> COMPLETE`.

Governor persists `DELIVERED` with `delivery_state=ATTEMPTING` before calling
Herdr. This is an intent, not acknowledgement. A successful blocking wake is
`CONFIRMED`; a Controller claim is `ACKNOWLEDGED`. An ambiguous caller return
is `UNKNOWN_DELIVERY` and is never blindly repeated. Only documented pre-input
`agent_blocked` rejection allows one retry, after 60 seconds, with a total
two-attempt ceiling. A late claim/result always wins over a caller error.

If the recipient is working,
the event stays `QUEUED`; Governor does not prompt, poll the turn, or append a
second message. When the recipient becomes idle, Governor delivers the oldest
queued event once.

The existing exception is one terminal notice to an unclaimed Controller pane
whose UI still reports working; a claimed event is never interrupted. There is
no periodic reminder prompt. Unconfirmed delivery produces one local incident
file in `delivery_alerts/` and an actionable service status, not a prompt flood.
Confirmed Director reviews have a 30-minute grace period for their response.
State mutations use a short cross-process transaction lock, released before
Herdr calls; completion/claim changes during a wake cannot be overwritten.

Controller claims and completes events with `scripts/arc2_governor.py` CLI
commands. A Director response is consumed only when its
`reviewed_brief_sha256` matches the immutable brief. Missing files, invalid
JSON, and hash mismatches create one content-addressed `PRODUCER_ERROR` event
for the producer; they never create an unbounded review loop.

## Remote completion

Supervisor writes terminal events to
`.arc2-local/orchestration/events/`. Governor notices the event and wakes an
idle Controller once. Controller ingests and freezes the receipt/result. A
complete result then becomes one SHA-bound Director review. This is the P0
path: remote completion must not require a user to wake Controller manually.

All handoff paths persist the same immutable manifest in `remote_jobs/`.
Normal Controller completion with `WAIT_REMOTE` registers it automatically.
An already-running pipeline can be adopted, without launch or rearm, using:

```text
py -3 scripts/arc2_governor.py --state STATE --register-remote-job MANIFEST
```

For legacy pipelines without a terminal receipt, the optional
`pipeline_status_path` must be exactly `output_root/PIPELINE_STATE.json`.
Supervisor observes only that metadata and the bound PID, boot identity,
start ticks and command hash. It records an explicitly local mechanical
receipt: matching `COMPLETE` -> completion; `FAILED` -> infrastructure failure;
missing/reused process or wrong source -> `UNKNOWN`, never automatic rerun.
The original pipeline and experiment bytes remain untouched. This bridge is
monitoring only, not a replacement scientific launcher or authorization gate.

## State and provenance

The live schema is deliberately small and defined by
`ARC2_WORKFLOW_STATE_TEMPLATE.json`. Event transitions append to
`.arc2-local/orchestration/ARC2_EVENT_LEDGER.jsonl`. Before schema migration,
the legacy state is copied byte-for-byte to `state_backups/`; historical
experiment artifacts are never rewritten.

`orchestration/governor/Start-Arc2Governor.ps1` runs the sole persistent router.
It stays alive across `PAUSED` and stage-level `TERMINAL` dispositions. Its
watchdog emits at most one repair notification for a continuous process failure.
`scripts/arc2_runner.py` and historical directive/acknowledgement state machines
remain provenance only.
