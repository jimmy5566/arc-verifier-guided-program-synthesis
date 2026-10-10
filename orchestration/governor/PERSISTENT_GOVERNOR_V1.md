# Persistent ARC2 Governor

`Start-Arc2Governor.ps1` starts the only persistent local scheduler. It owns a
process-lifetime lock at `.arc2-local/orchestration/arc2_governor.lock`, writes
an observable heartbeat at `ARC2_GOVERNOR_SERVICE_STATE.json`, and keeps
running while a workflow is `PAUSED`; it exits only at `TERMINAL` or when
explicitly stopped.

The process never starts a second Controller turn for the same durable action
until its dispatch lease expires (five minutes by default). A Controller that
has persisted an updated action is dispatched immediately. A live Controller
pane is never prompted recursively.

`-Action Install` registers one per-user `ARC2-Governor` Windows Scheduled
Task. The task starts the same locked process at logon and restarts it after an
unexpected failure. It is a process host, not a second scheduler.

The existing Supervisor Watchdog remains responsible only for a detached
RunPod-job Supervisor. It must not start Governor or Controller work.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File orchestration/governor/Start-Arc2Governor.ps1 -Action Start
powershell -NoProfile -ExecutionPolicy Bypass -File orchestration/governor/Start-Arc2Governor.ps1 -Action Install
powershell -NoProfile -ExecutionPolicy Bypass -File orchestration/governor/Start-Arc2Governor.ps1 -Action Status
powershell -NoProfile -ExecutionPolicy Bypass -File orchestration/governor/Start-Arc2Governor.ps1 -Action Stop
```

For an immutable worktree checkout, pass `-GovernorScript` with that checkout's
`scripts/arc2_governor.py`; the durable state still remains in the active ARC2
workspace. This is the supported way to run an exact committed scheduler while
leaving unrelated worktree changes untouched.
