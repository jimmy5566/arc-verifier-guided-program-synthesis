# Bounded Governor handoff repair and validation

Scope: deterministic ARC2 routing only. No Agent layers, scientific gates,
model loading, experiment reruns, inference changes or target access.

## Root causes addressed

- Independently running pipelines could be absent from the Supervisor registry.
- Durable prompt intent was indistinguishable from delivery acknowledgement.
- Saving a pre-prompt snapshot could overwrite concurrent claim/completion.
- Imported outbox files could be removed before their event queue was durable.
- Routine prompts requested the entire historical workflow as context.

## Implementation

One shared immutable `remote_jobs/` registry is used by normal WAIT_REMOTE
completion, recovery and explicit adoption. The optional existing-pipeline
bridge reads PIPELINE_STATE metadata plus PID creation identity, produces an
explicitly local mechanical receipt and routes exactly one terminal event.
The bridge does not alter the remote process or its immutable source checkout.

Delivery intent, confirmed wake, acknowledgement and ambiguous delivery are
distinct metadata. Only proven pre-input rejection permits one retry after a
60-second backoff (two attempts total). Unknown delivery is not blindly retried;
late authoritative claims/results remain consumable. One incident alert is
written locally instead of repeated Agent messages. Short transaction locks
protect state mutations and are released during Herdr calls.

## CPU validation

- Governor handoff, thin router and Supervisor integration: **28 passed**.
- Existing Supervisor UTF-8, observability and directive compatibility: **19 passed**.
- Tests cover normal handoff registration, same-job idempotency, active-job
  protection, process exit/PID reuse, wrong source, repeated polling/restart,
  timeout without duplicate prompt, pre-input rejection backoff, claim during
  a nonzero prompt return, owner pause during observation, late Director result,
  durable outbox handling and registry-to-terminal-to-wake-to-claim integration.
- All Herdr dispatches in these tests are mocks; no test launches an Agent or GPU.
- Legacy Foundation reconstruction controls were also attempted: two unrelated
  tests fail because their historical commands omit required `--accounting-state`.
  They were not changed. The existing `src/orchestration` package shadows the
  orchestration scripts for direct pytest collection; the 19 compatibility tests
  were run with an explicit root namespace in the test process only.

## Rollout and rollback

Preserve live workflow bytes and service metadata before adoption. Stop only
the verified local Governor watchdog/Governor and Supervisor, never the remote
scientific process. Start one refreshed Governor, one watchdog and one
Supervisor using the existing direct Python interpreter. Adopt the verified
Eval60 pipeline identity; confirm WAIT_REMOTE, fresh service heartbeat and
successful metadata observation. Real final completion wake remains pending
until the running experiment terminates; CPU tests are not that evidence.

Rollback uses the prior committed service source plus the preserved state;
do not reset research branches, replay old events or relaunch remote work.

## Live rollout verification

At `2026-10-15T18:20:55Z`:

- Prior state/service metadata preserved under local
  `state_backups/handoff_v2_20261016T045034/` before registration.
- Exactly one Governor PID **204**, parent Watchdog PID **27464**; one
  Supervisor PID **17704**. Direct Python avoids a launcher-shim duplicate.
- Governor service state: `WAIT_REMOTE`, job
  `V7_TTT24_EVAL60_READY_RESTORE_V2_cde188b6`, zero queued or unacknowledged events.
- Supervisor heartbeat: `BOUND_REMOTE_JOB_RECONCILIATION_OK`; metadata polling
  interval **45 seconds**, with no LLM call while awaiting the remote terminal.
- Governor Watchdog healthy, zero consecutive failures; refreshed service
  stderr files empty.
- Remote original pipeline PID **147516** and worker PID **147688** still alive,
  source still `cde188b600600c17cf5c9419db1cfd8cd4898f74`; generation **83/89**,
  current attempt zero. No remote source checkout or experimental restart.
- Existing scientific/historical Director responses were deliberately excluded
  from these infrastructure commits; unrelated dirty files were preserved.
- Actual terminal-to-Agent acknowledgement remains **PENDING_REMOTE_COMPLETION**,
  not claimed as proven by CPU mocks or a healthy service heartbeat.
