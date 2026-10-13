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

## Event lifecycle

Each event has a content-derived ID and exactly four states:
`QUEUED -> DELIVERED -> CLAIMED -> COMPLETE`.

Governor persists `DELIVERED` before calling Herdr. Therefore a restart or an
ambiguous CLI return cannot duplicate the prompt. If the recipient is working,
the event stays `QUEUED`; Governor does not prompt, poll the turn, or append a
second message. When the recipient becomes idle, Governor delivers the oldest
queued event once.

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
