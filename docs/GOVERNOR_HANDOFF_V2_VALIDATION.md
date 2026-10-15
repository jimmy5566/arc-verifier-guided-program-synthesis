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

## Owner-requested functional acceptance, 2026-10-16 Adelaide

Expanded core suite: **35 passed**; Supervisor UTF-8/observability/directive
compatibility: **19 passed**, **54 total**. These include actual isolated
Windows process-host tests, not only Python mocks.

| Function | Evidence | Result |
| --- | --- | --- |
| Busy queue, exactly-once dispatch, active-event exclusivity | CPU router tests | PASS |
| Timeout, bounded pre-input rejection recovery, late result | CPU injected failures | PASS |
| Concurrent state mutation and two competing claims | Mock transition plus two real CLI processes | PASS |
| Registration, process exit/PID reuse, source mismatch | CPU registry/Supervisor integration | PASS |
| Controller-to-Director-to-Controller result routing | Isolated bound-brief CPU integration | PASS |
| Singleton lock and quiescent heartbeat | Real isolated Governor processes | PASS |
| Watchdog start and recovery after intentional child termination | Real isolated Windows Watchdog/daemon | PASS |
| Existing Controller wake, claim and completion | One actual Herdr prompt through Governor; isolated state | PASS |
| Current Eval60 metadata transport | Real authorized SSH metadata-only probe; process/source match | PASS |
| Real Eval60 terminal-to-result-review handoff | Experiment still running | PENDING |

Real smoke event `f2e71ae91ef8ef6b3b0b8dc2efb618bf9cbec332c0d8ad6f9cb47b016d08750d`
was delivered once at `18:33:55Z`, claimed at `18:34:06Z`, and completed at
`18:34:06Z` on 2026-10-15 UTC. Two subsequent real Governor `--once` calls were
IDLE and did not issue another prompt. Controller returned to done. Its only
permitted action was claim/complete the isolated NOOP event; no scientific
artifacts, model or GPU work were part of this smoke.

Production state remained revision **275**, WAIT_REMOTE, no active event. The
deferred eight-hour window stayed unactivated. Test state/ledger are retained
under local `service_validation/governor_20261016/`; not imported into the live
event inbox.

Two issues found during expanded testing were fixed:

- The old Watchdog test tried to clean up a launcher parent that no longer
  exists with direct Python. Cleanup now terminates only its exact test-owned
  daemon; the initially orphaned test daemon was identified and stopped.
- A replay of an already-completed old-job event could be misclassified as a
  conflict with the current job. Known event identity is now checked before
  active-job validation. A regression test confirms no new prompt, producer
  error, remote-binding change or workflow transition on that replay.
