# Single-delivery concurrent Agent queues

Owner requested one-shot delivery, skip-busy dispatch and the same route for
remote work. No new Agent or service layer is introduced.

- One authoritative event table; FIFO within each exact Agent identity.
- Skip busy/held/reserved/unavailable Agents and inspect the next Agent.
- Independent Agents may be DELIVERED/CLAIMED concurrently. One reservation per
  Agent survives UI idleness, caller timeouts and service restarts.
- One delivery attempt per event. No rejection retry, periodic reminder,
  claimed-payload resume or implicit task synthesis from shared display fields.
- Completion/review callbacks retain their originating Controller role and do
  not clear another Agent's claim, review or detached-job binding.
- Remote execution is an authorized Agent task; detached jobs register immutable
  per-job manifests. Supervisor emits one terminal outbox event; Governor alone
  delivers to the exact idle recipient. Success/failure/UNKNOWN use this route.
- Stage pauses are scoped; global owner STOP and exact owner-held routes stay
  effective. Queueing never grants GPU, launch, retry or scientific authority.
- A stalled unacknowledged historical delivery remains diagnostic evidence;
  upgrading the service never replays its scientific payload.

Validation command (all isolated/model-free):

```
py -3 -m pytest -q tests/test_arc2_governor_thin_router.py tests/test_arc2_governor_named_routes.py tests/test_arc2_governor_handoff.py tests/test_arc2_governor_concurrent_queue.py tests/test_governor_queue_completion.py tests/test_arc2_orchestration_supervisor.py tests/test_arc2_persistent_governor.py
```

Coverage includes real CLI claims on different Agents, same-Agent exclusion,
busy-head skipping, uncertain-delivery restart, concurrent reviews, two remote
jobs, exact role propagation, cross-role rejection, owner STOP and Windows
single-daemon/watchdog lifecycle. These are CPU/control-plane tests, not an
actual remote job or scientific result.
