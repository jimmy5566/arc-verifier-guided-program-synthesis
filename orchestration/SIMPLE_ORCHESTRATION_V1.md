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

The single event table is a concurrent per-agent queue: each cycle examines
the oldest pending task for each exact Agent identity. A working, held,
unavailable or already-reserved Agent is skipped so other idle Agents can
receive their work in the same cycle. Each Agent has at most one DELIVERED or
CLAIMED event; different Agents may claim and complete concurrently. No
secondary queue or persistent worker service is introduced.

Governor persists `DELIVERED` with `delivery_state=ATTEMPTING` before calling
Herdr. This is an intent, not acknowledgement. A successful blocking wake is
`CONFIRMED`; a Controller claim is `ACKNOWLEDGED`. An ambiguous caller return
is `UNKNOWN_DELIVERY` and is never blindly repeated. Every event permits only
one delivery attempt, including timeouts and pre-input `agent_blocked`
rejections. Automatic retries and explicit claimed-payload re-prompting are
disabled. A late claim/result always wins over a caller error. An unresolved
delivery is reported once, not treated as permission to create a duplicate task.

If the recipient is working,
the event stays `QUEUED`; Governor does not prompt, poll the turn, or append a
second message. Governor checks the next Agent's task instead. When this
recipient becomes idle and has no existing reservation, its oldest queued
event is delivered once. UI idleness alone never releases an unacknowledged
delivery reservation.

Remote terminal notices obey exactly the same busy/idle rule; they never
interrupt a working Agent. There is no periodic reminder prompt. Unconfirmed delivery produces one local incident
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

Key-node status reports are distinct from scientific approvals. Controller may
attach one immutable `--stage-report PATH --stage-report-sha256 SHA` to normal
completion for a verified MILESTONE, EXPERIMENT_RESULT or genuine BLOCKED state.
The router enqueues one informational Director task in the same transaction;
Controller continues without waiting. A Director `--ack-stage-report EVENT_ID`
completes only that notice and preserves other jobs/stages/holds. Ordinary retries
stay local and results already requesting scientific review are not double-reported.

Kaggle jobs use typed kernel/version/source/nonce bindings rather than fake SSH
PIDs. Cached, bound worker terminal receipts are checked before download. Missing
supplementary artifacts are listed for exact recovery; callback transport alone
never certifies scientific completeness. Named-receipt fetching stops when the
requested files arrive, without scanning installed-package output pages. A
platform-terminal job without a verifiable worker receipt routes UNKNOWN promptly;
it does not silently wait until an hour-long deadline. The existing Supervisor
does metadata reconciliation, Governor delivers once, and neither reruns kernels.

Supervisor writes terminal events to
`.arc2-local/orchestration/events/`. Governor notices the event and wakes an
idle Controller once. Controller ingests and freezes the receipt/result. A
complete result then becomes one SHA-bound Director review. This is the P0
path: remote completion must not require a user to wake Controller manually.

All handoff paths persist the same immutable manifest in `remote_jobs/`.
Remote execution itself starts from a normal Governor Controller/Director task
whose payload references its explicit launch authority; queueing never grants
GPU permission. The executor registers the detached job with its exact PID,
source and output identity. For named Controllers the remote manifest must
carry `controller_role=oracle|selector` matching the claiming Agent.
Supervisor preserves that role on its one terminal outbox event, for success,
failure and UNKNOWN alike; only Governor delivers it. `completion_recipient`
may explicitly send a result to Director without losing Controller provenance.
Multiple Agents' jobs coexist by job ID. Importing one terminal event clears
only that matching job, and another Agent's local completion clears none.
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

The active owner queue is bound with `--bind-experiment-queue QUEUE_JSON`.
This adds only `experiment_queue_path` to live state. Director owns scientific
status and evidence. Governor allows TERMINAL only when all required items
are COMPLETE; otherwise a stage-only terminal becomes PAUSED without another
review, producer-error prompt or dispatch. This never grants GPU permission
or interrupts WAIT_REMOTE work. A completed stage is not the whole queue.

The live schema is deliberately small and defined by
`ARC2_WORKFLOW_STATE_TEMPLATE.json`. Event transitions append to
`.arc2-local/orchestration/ARC2_EVENT_LEDGER.jsonl`. Before schema migration,
the legacy state is copied byte-for-byte to `state_backups/`; historical
experiment artifacts are never rewritten.

`pending_events` is authoritative for delivery/claim ownership. `active_events`
is derived by Agent; `active_event` is only a legacy single-item display.
`remote_jobs` holds independent job bindings; `remote_job` is its legacy display.
GPU launchers use `scripts/arc2_gpu_lock.py` on the actual Linux GPU host.
One kernel lock per GPU UUID lives at `/run/lock/arc2-gpu/`, never FUSE.
Workers inherit the lock FD, so launcher death does not unlock a live worker.
Never unlink/age-steal a lock. The wrapper bounds worker wall time, kills its
process group on exit/timeout, and writes a distinct cost receipt. It does not
grant/consume authorization: the worker still verifies its own fresh lease
and Supervisor registration before CUDA. CPU preparation can run concurrently;
blocked GPU launchers wait in the OS without consuming their compute lease.
New GPU manifests bind `gpu_resource` (UUID) and
`gpu_lock_policy=HOST_LOCAL_INHERITED_FD_V1`. A waiting job may be registered
while another role owns the device. Owner priority is an explicit predecessor
barrier, not a claim that kernel advisory locks provide FIFO fairness.
The latest global stage/disposition fields are not another Agent's queue.
Normal named-Agent PAUSED/TERMINAL stage results pause that Agent only, using
`paused_roles` and `pause_scope=agent`; an explicit global owner STOP remains a
global stop and survives concurrent result ingestion/completion. New explicit
tasks can reopen their own scoped lane but never lift `agent_routes.*.held`.
An actual global owner STOP is reopened only by an owner-produced outbox task
with explicit `resume_global=true`, not an ordinary result or continuation.
Controller completions atomically enqueue their own review/successor so two
parallel completions cannot overwrite a singleton review pointer. No new task
is synthesized from another Agent's last-written display fields.

`orchestration/governor/Start-Arc2Governor.ps1` runs the sole persistent router.
It stays alive across `PAUSED` and queue-complete `TERMINAL` dispositions. Its
watchdog emits at most one durable repair-task outbox event for a continuous
process failure, never an out-of-band Agent prompt. If the router cannot be
recovered, that outbox remains pending rather than pretending a wake succeeded.
`scripts/arc2_runner.py` and historical directive/acknowledgement state machines
remain provenance only.
