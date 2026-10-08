"""Immediate, durable Controller continuation driver for ARC2.

This is deliberately a small local state driver, not another reasoning agent.
It watches the Controller's durable directive-cycle state and asks the real
Herdr Controller target for one continuation only after its prior turn has
ended.  Initial acknowledgements, Director briefs, remediation, commits, and
GPU launches remain owned by the Controller cycle itself; this driver never
writes any of those artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


ACTIVE_STATES = {
    "PROCESSING",
    "REMEDIATION",
    "VALIDATING",
    "COMMITTING",
    "RESUBMITTING",
    "STAGE_PREPARATION",
    "DIRECTOR_REMEDIATION",
}
IDLE_ALLOWED_STATES = {
    "WAITING_REMOTE_JOB",
    "WAITING_DIRECTOR_ACTIVE",
    "AUTHORIZED_STAGE_COMPLETE",
    "HARD_BLOCKED",
    "PAUSED",
    "STOPPED",
    "TERMINAL",
}
LOCK_SUFFIX = ".immediate-requeue.lock"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def load_state(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != 1 or not isinstance(value.get("directives"), dict):
        raise RuntimeError("INVALID_CONTROLLER_DIRECTIVE_STATE")
    return value


@contextmanager
def exclusive_lock(state_path: Path) -> Iterator[None]:
    """Serialize local driver invocations without mutating scientific state."""
    lock = state_path.with_name(state_path.name + LOCK_SUFFIX)
    try:
        descriptor = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError("CONTROLLER_REQUEUE_DRIVER_ALREADY_RUNNING") from exc
    try:
        os.write(descriptor, f"pid={os.getpid()} started_at={utc_now()}\n".encode("ascii"))
        yield
    finally:
        os.close(descriptor)
        try:
            lock.unlink()
        except FileNotFoundError:
            pass


def progress_marker(record: dict[str, Any]) -> str:
    fields = {
        key: record.get(key)
        for key in (
            "state", "next_step", "last_progress_at", "last_completed_step",
            "commit", "brief", "evidence", "validation",
        )
    }
    return hashlib.sha256(json.dumps(fields, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def herdr_agents() -> list[dict[str, Any]]:
    completed = subprocess.run(
        ["herdr", "agent", "list"], text=True, encoding="utf-8", errors="strict",
        capture_output=True, check=False, timeout=15,
    )
    if completed.returncode != 0:
        raise RuntimeError("HERDR_AGENT_LIST_FAILED")
    try:
        value = json.loads(completed.stdout)
        agents = value["result"]["agents"]
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError("HERDR_AGENT_LIST_INVALID") from exc
    if not isinstance(agents, list):
        raise RuntimeError("HERDR_AGENT_LIST_INVALID")
    return [agent for agent in agents if isinstance(agent, dict)]


def resolve_controller(agents: list[dict[str, Any]], requested: str) -> tuple[str, str, str]:
    """Resolve the logical arc-controller role to a live Herdr target.

    Existing ARC2 installations often expose the primary Controller as the
    focused ``codex`` agent rather than a separately named ``arc-controller``.
    Prefer the named role; use only that focused Codex fallback; otherwise fail
    closed rather than sending to an unrelated agent.
    """
    for agent in agents:
        if agent.get("agent") == requested or agent.get("name") == requested:
            return str(agent.get("agent", requested)), "EXACT_ROLE", str(agent.get("agent_status", "unknown"))
    focused = [a for a in agents if a.get("agent") == "codex" and a.get("focused") is True]
    if len(focused) == 1:
        return "codex", "FOCUSED_CODEX_CONTROLLER_FALLBACK", str(focused[0].get("agent_status", "unknown"))
    raise RuntimeError("CONTROLLER_AGENT_UNRESOLVED")


def invoke_controller(target: str, prompt: str) -> tuple[int, str]:
    """Submit one continuation and wait only until Herdr accepts the turn."""
    completed = subprocess.run(
        ["herdr", "agent", "prompt", target, prompt, "--wait", "--until", "working", "--timeout", "15000"],
        text=True, encoding="utf-8", errors="strict", capture_output=True, check=False, timeout=30,
    )
    return completed.returncode, (completed.stderr or completed.stdout).strip()


def active_and_executable(record: dict[str, Any]) -> bool:
    return str(record.get("state", "")) in ACTIVE_STATES and bool(record.get("next_step"))


def requeue_once(state_path: Path, controller_agent: str, max_no_progress: int = 2) -> list[str]:
    """Prompt at most one eligible Controller continuation.

    An outstanding lease survives process restarts.  A later invocation sees an
    idle Controller, records the prior lease as completed, compares durable
    progress, and either issues the next continuation or requests escalation
    after two no-progress turns.  It never alters acknowledgements, briefs,
    commits, launch contracts, or GPU state.
    """
    with exclusive_lock(state_path):
        state = load_state(state_path)
        directives = state["directives"]
        eligible = [(key, value) for key, value in sorted(directives.items()) if active_and_executable(value)]
        if not eligible:
            return []

        agents = herdr_agents()
        target, resolution, status = resolve_controller(agents, controller_agent)
        if status == "working":
            return []

        directive_id, record = eligible[0]
        requeue = record.setdefault("immediate_requeue", {})
        marker = progress_marker(record)
        outstanding = requeue.get("active_invocation")
        if outstanding:
            requeue["completed_invocation"] = outstanding
            requeue["completion_observed_at"] = utc_now()
            requeue["active_invocation"] = None
            # Compare the durable state after this Controller turn to the
            # marker captured before that exact turn.  The first returned turn
            # with no state/artifact advancement is already one no-progress
            # continuation; two such turns are the bounded stall threshold.
            if requeue.get("progress_marker_before") == marker:
                requeue["consecutive_no_progress"] = int(requeue.get("consecutive_no_progress", 0)) + 1
            else:
                requeue["consecutive_no_progress"] = 0
            requeue["last_completed_progress_marker"] = marker

        no_progress = int(requeue.get("consecutive_no_progress", 0))
        if no_progress >= max_no_progress:
            record["controller_cycle_complete"] = False
            record["local_requeue_escalation_required"] = True
            record["local_requeue_escalation_type"] = "CONTROLLER_STALL"
            record["local_requeue_escalation_reason"] = "TWO_CONSECUTIVE_REQUEUES_WITHOUT_DURABLE_PROGRESS"
            record["local_requeue_escalation_at"] = utc_now()
            atomic_json(state_path, state)
            return []

        sequence = int(requeue.get("continuation_sequence", 0)) + 1
        invocation_id = f"{directive_id}:local-requeue:{sequence}:{uuid.uuid4().hex[:12]}"
        requeue.update({
            "schema_version": 1,
            "controller_target": target,
            "controller_target_resolution": resolution,
            "continuation_sequence": sequence,
            "active_invocation": invocation_id,
            "lease_started_at": utc_now(),
            "state_before": record["state"],
            "next_step_before": record["next_step"],
            "progress_marker_before": marker,
            "last_prompt_at": utc_now(),
            "prompt_status": "LEASED_PENDING_DELIVERY",
        })
        record["controller_cycle_complete"] = False
        atomic_json(state_path, state)

        prompt = (
            f"ARC2 Controller immediate continuation {invocation_id}: resume the SAME durable directive cycle "
            f"{directive_id} from state {record['state']}. Read {state_path.resolve()}. "
            f"Execute its next unfinished requirement ({record['next_step']}). Do not create another acknowledgement, "
            "Director brief, remediation action, commit, or GPU launch already recorded in durable state. "
            "Do not stop after identifying the action. Continue until a permitted WAITING or terminal state. "
            "Do not start GPU/model work without explicit scientific authorization."
        )
        code, output = invoke_controller(target, prompt)
        # Persist delivery outcome but retain an ambiguous failed lease. Retrying
        # an unknown Herdr delivery could duplicate the same continuation.
        state = load_state(state_path)
        latest = state["directives"].get(directive_id, {})
        latest_requeue = latest.setdefault("immediate_requeue", {})
        latest_requeue["prompt_exit_code"] = code
        latest_requeue["prompt_completed_at"] = utc_now()
        latest_requeue["prompt_status"] = "SENT" if code == 0 else "DELIVERY_UNKNOWN_FAIL_CLOSED"
        if code != 0:
            latest_requeue["prompt_error"] = output
        atomic_json(state_path, state)
        return [directive_id] if code == 0 else []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--controller-agent", default="arc-controller")
    parser.add_argument("--max-no-progress", type=int, default=2)
    parser.add_argument("--watch", action="store_true", help="run local one-second reconciliation until interrupted")
    parser.add_argument("--interval-seconds", type=float, default=1.0)
    args = parser.parse_args()
    if args.max_no_progress < 1 or args.interval_seconds <= 0:
        parser.error("invalid retry or interval")
    while True:
        requeue_once(args.state, args.controller_agent, args.max_no_progress)
        if not args.watch:
            return 0
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)
