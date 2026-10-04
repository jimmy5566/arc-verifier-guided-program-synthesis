"""Bounded POSIX process-group cleanup for Kaggle benchmark controllers.

The outer notebook starts every benchmark phase in a new process session.  A
successful controller leader can still leave a worker/loader descendant in its
process group, so cleanup is intentionally based on *PGID membership*, never
on the leader's ``wait()`` result alone.
"""
from __future__ import annotations

import os
import signal
import subprocess
import time
from typing import Any, Callable


ProcessRows = Callable[[int], list[dict[str, Any]]]
_SIGTERM = getattr(signal, "SIGTERM", 15)
_SIGKILL = getattr(signal, "SIGKILL", 9)


def pg_rows(pgid: int) -> list[dict[str, Any]]:
    """Return a bounded best-effort snapshot of live members of ``pgid``."""
    try:
        lines = subprocess.check_output(
            ["ps", "-eo", "pid=,pgid=,stat=,cmd="], text=True, timeout=10
        ).splitlines()
        rows: list[dict[str, Any]] = []
        for line in lines:
            parts = line.split(None, 3)
            if len(parts) >= 3 and int(parts[1]) == pgid:
                rows.append({
                    "pid": int(parts[0]), "pgid": int(parts[1]), "stat": parts[2],
                    "cmd": parts[3] if len(parts) > 3 else "",
                })
        return rows
    except Exception as exc:
        # Fail closed: an unobservable process group must never be treated as
        # empty by a controller that is trying to prove cleanup.
        return [{"pid": None, "pgid": pgid, "ps_error": repr(exc)}]


def assert_no_process_group_survivors(
    pgid: int, *, rows: ProcessRows = pg_rows
) -> None:
    """Fail closed when a phase leader exited but its PGID is still live."""
    survivors = rows(pgid)
    if survivors:
        raise RuntimeError(
            f"PROCESS_GROUP_SURVIVOR_AFTER_SUCCESS: pgid={pgid} survivors={survivors}"
        )


def cleanup_process_group(
    pgid: int,
    *,
    grace_seconds: float = 10.0,
    kill_grace_seconds: float = 10.0,
    poll_interval_seconds: float = 0.1,
    rows: ProcessRows = pg_rows,
    killpg: Callable[[int, signal.Signals], None] | None = None,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Terminate a full process group with bounded TERM/KILL polling.

    The returned ``remaining_surviving_pids`` is empty only when a real final
    PGID snapshot is empty.  Callers must fail closed otherwise.
    """
    if killpg is None:
        if not hasattr(os, "killpg"):
            raise RuntimeError("POSIX_PROCESS_GROUP_CLEANUP_UNAVAILABLE")
        killpg = os.killpg
    actions: list[str] = []
    initial_rows = rows(pgid)
    try:
        killpg(pgid, _SIGTERM)
        actions.append("SIGTERM_PROCESS_GROUP")
    except ProcessLookupError:
        actions.append("PROCESS_GROUP_ALREADY_GONE")

    def poll_until(deadline: float) -> list[dict[str, Any]]:
        current = rows(pgid)
        while current and monotonic() < deadline:
            sleep(max(0.0, min(poll_interval_seconds, deadline - monotonic())))
            current = rows(pgid)
        return current

    survivors = poll_until(monotonic() + grace_seconds)
    if survivors:
        try:
            killpg(pgid, _SIGKILL)
            actions.append("SIGKILL_PROCESS_GROUP")
        except ProcessLookupError:
            actions.append("PROCESS_GROUP_ALREADY_GONE_AFTER_TERM")
        survivors = poll_until(monotonic() + kill_grace_seconds)
    final_rows = rows(pgid) if not survivors else survivors
    return {
        "pgid": pgid,
        "initial_processes": initial_rows,
        "termination_actions": actions,
        "remaining_surviving_pids": [row.get("pid") for row in final_rows if row.get("pid") is not None],
        "remaining_surviving_processes": final_rows,
        "cleanup_complete": not final_rows,
    }
