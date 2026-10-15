#!/usr/bin/env python3
"""Thin, deterministic ARC2 event router.

The Governor only routes immutable events, consumes SHA-bound Director
responses, and reports one producer error for invalid bindings. It never polls
RunPod, interprets science, repairs infrastructure, or invents authorization.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from functools import wraps
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DISPOSITIONS = {"CONTINUE_CONTROLLER", "REVIEW_REQUIRED", "WAIT_REMOTE", "PAUSED", "TERMINAL"}
EVENT_STATUSES = {"QUEUED", "DELIVERED", "CLAIMED", "COMPLETE"}
DECISIONS = {"CONTINUE_CONTROLLER", "CONTINUE_DIRECTOR", "REQUIRE_CHANGES", "PAUSED", "TERMINAL"}
LIVE_KEYS = {
    "schema_version", "revision", "disposition", "stage", "next_action",
    "controller_target", "director_target", "pending_events", "active_event", "review",
    "remote_job", "last_transition", "pause_reason", "terminal_reason", "updated_at",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def windows_creationflags() -> int:
    return getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_id(kind: str, recipient: str, payload: dict[str, Any]) -> str:
    material = json.dumps(
        {"kind": kind, "recipient": recipient, "payload": payload},
        sort_keys=True, separators=(",", ":"), default=str,
    ).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        for attempt in range(5):
            try:
                os.replace(temporary, path)
                return
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.1)
    finally:
        temporary.unlink(missing_ok=True)


def append_ledger(state_path: Path, record: dict[str, Any]) -> None:
    ledger = state_path.parent / "ARC2_EVENT_LEDGER.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps({"at": now(), **record}, sort_keys=True) + "\n")


def compact_state(value: dict[str, Any]) -> dict[str, Any]:
    state = {key: value.get(key) for key in LIVE_KEYS if key in value}
    defaults = {
        "schema_version": 3, "revision": 0, "disposition": "PAUSED",
        "stage": "UNSPECIFIED", "next_action": None, "controller_target": None,
        "director_target": None,
        "pending_events": [], "active_event": None, "review": None,
        "remote_job": None, "last_transition": "STATE_LOADED", "pause_reason": None,
        "terminal_reason": None, "updated_at": now(),
    }
    for key, value in defaults.items():
        state.setdefault(key, value)
    return state


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if value.get("schema_version") != 3:
        raise RuntimeError("LEGACY_STATE_REQUIRES_EXPLICIT_MIGRATION")
    state = compact_state(value)
    if state["disposition"] not in DISPOSITIONS:
        raise RuntimeError("INVALID_GOVERNOR_DISPOSITION")
    if not isinstance(state["pending_events"], list):
        raise RuntimeError("INVALID_PENDING_EVENTS")
    for item in state["pending_events"]:
        if item.get("status") not in EVENT_STATUSES:
            raise RuntimeError("INVALID_EVENT_STATUS")
    return state


def save(path: Path, state: dict[str, Any], transition: str) -> None:
    compacted = compact_state(state)
    state.clear()
    state.update(compacted)
    state["revision"] = int(state.get("revision") or 0) + 1
    state["last_transition"] = transition
    state["updated_at"] = now()
    atomic(path, state)
    append_ledger(path, {"transition": transition, "revision": state["revision"],
                         "disposition": state["disposition"], "stage": state["stage"]})


def migrate_legacy(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    legacy = json.loads(raw.decode("utf-8-sig"))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive = path.parent / "state_backups" / f"{path.stem}.pre_thin.{stamp}.json"
    archive.parent.mkdir(parents=True, exist_ok=True)
    if archive.exists():
        raise RuntimeError("MIGRATION_ARCHIVE_ALREADY_EXISTS")
    archive.write_bytes(raw)
    disposition = legacy.get("disposition")
    if disposition not in DISPOSITIONS:
        disposition = {
            "ACTIVE": "CONTINUE_CONTROLLER", "WAITING_REMOTE_JOB": "WAIT_REMOTE",
            "PAUSED": "PAUSED", "TERMINAL": "TERMINAL",
        }.get(legacy.get("status"), "PAUSED")
    review = None
    raw_brief = legacy.get("review_brief")
    if raw_brief:
        if isinstance(raw_brief, dict):
            review = {"brief_path": raw_brief.get("path"), "brief_sha256": raw_brief.get("sha256"),
                      "reason": legacy.get("review_reason")}
        else:
            review = {"brief_path": raw_brief, "brief_sha256": legacy.get("review_brief_sha256"),
                      "reason": legacy.get("review_reason")}
    state = compact_state({
        "schema_version": 3, "revision": 0, "disposition": disposition,
        "stage": legacy.get("stage") or "MIGRATED_LEGACY_STATE",
        "next_action": legacy.get("next_action"), "controller_target": legacy.get("controller_target"),
        "pending_events": [], "active_event": None, "review": review,
        "remote_job": legacy.get("remote_job") or legacy.get("active_remote_job"),
        "pause_reason": legacy.get("pause_reason"), "terminal_reason": legacy.get("terminal_reason"),
    })
    atomic(path, state)
    append_ledger(path, {"transition": "LEGACY_STATE_MIGRATED", "archive": str(archive.resolve()),
                         "legacy_sha256": hashlib.sha256(raw).hexdigest()})
    return state


class GovernorLock:
    def __init__(self, path: Path):
        self.path = path
        self.handle = None

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+b")
        if self.path.stat().st_size == 0:
            self.handle.write(b" ")
            self.handle.flush()
        self.handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.handle.close()
            self.handle = None
            return False
        self.handle.seek(0)
        self.handle.write(json.dumps({"pid": os.getpid(), "started_at": now()}).encode("utf-8"))
        self.handle.truncate()
        self.handle.flush()
        return True

    def release(self) -> None:
        if not self.handle:
            return
        try:
            self.handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        finally:
            self.handle.close()
            self.handle = None


@contextmanager
def state_transaction(path: Path):
    """Serialize short state mutations, never network calls or Agent execution."""
    lock = GovernorLock(path.with_suffix('.transaction.lock'))
    deadline = time.monotonic() + 10
    while not lock.acquire():
        if time.monotonic() >= deadline:
            raise RuntimeError('STATE_TRANSACTION_BUSY')
        time.sleep(0.05)
    try:
        yield
    finally:
        lock.release()


def state_mutation(function):
    @wraps(function)
    def serialized(state_path, *args, **kwargs):
        with state_transaction(state_path):
            return function(state_path, *args, **kwargs)
    return serialized


def event(kind: str, recipient: str, producer: str, payload: dict[str, Any]) -> dict[str, Any]:
    event_id = stable_id(kind, recipient, payload)
    return {"event_id": event_id, "kind": kind, "recipient": recipient, "producer": producer,
            "payload": payload, "status": "QUEUED", "attempts": 0, "created_at": now()}


def enqueue(state: dict[str, Any], item: dict[str, Any]) -> bool:
    if any(entry.get("event_id") == item["event_id"] for entry in state["pending_events"]):
        return False
    state["pending_events"].append(item)
    return True


def find_event(state: dict[str, Any], event_id: str) -> dict[str, Any]:
    for item in state["pending_events"]:
        if item.get("event_id") == event_id:
            return item
    raise RuntimeError("EVENT_NOT_FOUND")


def resolve_artifact(raw: str) -> Path:
    candidate = Path(raw)
    if candidate.is_file():
        return candidate.resolve()
    if not candidate.is_absolute() and (PROJECT_ROOT / candidate).is_file():
        return (PROJECT_ROOT / candidate).resolve()
    raise RuntimeError("BOUND_ARTIFACT_MISSING")


def validate_bound_artifact(path_value: str, declared_sha: str | None) -> tuple[Path, str]:
    path = resolve_artifact(path_value)
    actual = sha256_file(path)
    if not declared_sha:
        raise RuntimeError("BOUND_ARTIFACT_SHA256_MISSING")
    if actual.lower() != str(declared_sha).lower():
        raise RuntimeError("BOUND_ARTIFACT_SHA256_MISMATCH")
    return path, actual


def producer_error(state: dict[str, Any], source: dict[str, Any], reason: str) -> bool:
    producer = source.get("producer") or "controller"
    recipient = producer if producer in {"controller", "director"} else "controller"
    payload = {"source_event_id": source.get("event_id"), "error": reason,
               "required_action": "REPAIR_ONLY_THE_REFERENCED_FILE_OR_FORMAT_AND_REEMIT_ONCE"}
    return enqueue(state, event("PRODUCER_ERROR", recipient, "governor", payload))


def import_external_events(state: dict[str, Any], state_path: Path) -> bool:
    changed = False
    event_dir = state_path.parent / "events"
    processed = event_dir / "processed"
    event_dir.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)
    for source in sorted(event_dir.glob("*.json")):
        try:
            value = json.loads(source.read_text(encoding="utf-8-sig"))
            if value.get("schema_version") != 1 or value.get("kind") not in {"REMOTE_TERMINAL", "CONTROLLER_TASK", "REVIEW_REQUEST"}:
                raise RuntimeError("EXTERNAL_EVENT_SCHEMA_INVALID")
            payload = value.get("payload")
            if not isinstance(payload, dict):
                raise RuntimeError("EXTERNAL_EVENT_PAYLOAD_INVALID")
            recipient = value.get("recipient") or ("director" if value["kind"] == "REVIEW_REQUEST" else "controller")
            item = event(value["kind"], recipient, value.get("producer") or "supervisor", payload)
            already_known = any(entry['event_id'] == item['event_id'] for entry in state['pending_events'])
            if not already_known and value['kind'] == 'REMOTE_TERMINAL' and state.get('remote_job'):
                if payload.get('round_id') != (state['remote_job'].get('job_id') or state['remote_job'].get('run_id')):
                    raise RuntimeError('TERMINAL_EVENT_ACTIVE_JOB_MISMATCH')
            added = enqueue(state, item)
            changed = added or changed
            if added and value["kind"] == "REMOTE_TERMINAL" and state['disposition'] != 'PAUSED':
                state.update({"disposition": "CONTINUE_CONTROLLER",
                              "next_action": "INGEST_FREEZE_AND_ROUTE_REMOTE_RESULT",
                              "remote_job": None, "terminal_reason": None})
            elif added and value["kind"] == "CONTROLLER_TASK":
                state.update({"disposition": "CONTINUE_CONTROLLER",
                              "stage": payload.get("stage") or state["stage"],
                              "next_action": payload.get("next_action"),
                              "pause_reason": None, "terminal_reason": None})
            elif added and value["kind"] == "REVIEW_REQUEST":
                state.update({"disposition": "REVIEW_REQUIRED", "review": {
                    "brief_path": payload.get("brief_path"),
                    "brief_sha256": payload.get("brief_sha256"),
                    "reason": payload.get("review_reason"),
                }})
            target = processed / f"{item['event_id']}.json"
            # Persist the queue before removing its producer's durable outbox.
            # A crash in between is a harmless idempotent import on restart.
            if added:
                save(state_path, state, f'EXTERNAL_EVENT_IMPORTED:{item["event_id"]}')
            if target.exists():
                source.unlink()
            else:
                os.replace(source, target)
        except Exception as exc:
            digest = sha256_file(source)
            marker = processed / f"INVALID_{digest}.json"
            if not marker.exists():
                shutil.copyfile(source, marker)
                changed = producer_error(state, {"event_id": digest, "producer": "controller"},
                                         f"{type(exc).__name__}:{exc}") or changed
                save(state_path, state, f'EXTERNAL_EVENT_REJECTED:{digest}')
            source.unlink(missing_ok=True)
    return changed


def synthesize_state_event(state: dict[str, Any]) -> bool:
    if state["disposition"] == "CONTINUE_CONTROLLER":
        if any(item["recipient"] == "controller" and item["status"] != "COMPLETE"
               for item in state["pending_events"]):
            return False
        payload = {"stage": state["stage"], "next_action": state["next_action"]}
        return enqueue(state, event("CONTROLLER_TASK", "controller", "director", payload))
    if state["disposition"] == "REVIEW_REQUIRED" and state.get("review"):
        if any(item["recipient"] == "director" and item["status"] != "COMPLETE"
               for item in state["pending_events"]):
            return False
        review = state["review"]
        payload = {"brief_path": review.get("brief_path"), "brief_sha256": review.get("brief_sha256"),
                   "review_reason": review.get("reason")}
        return enqueue(state, event("DIRECTOR_REVIEW", "director", "controller", payload))
    return False


def validate_queued_reviews(state: dict[str, Any]) -> bool:
    """Reject an invalid or response-derived brief before spending a Director turn."""
    changed = False
    for item in state["pending_events"]:
        if item["kind"] != "DIRECTOR_REVIEW" or item["status"] != "QUEUED":
            continue
        payload = item["payload"]
        try:
            validate_director_brief(payload["brief_path"])
            validate_bound_artifact(payload["brief_path"], payload.get("brief_sha256"))
        except Exception as exc:
            item.update({"status": "COMPLETE", "completed_at": now(), "outcome": "INVALID_BOUND_BRIEF"})
            producer_error(state, item, f"{type(exc).__name__}:{exc}")
            changed = True
    return changed


def herdr_agents() -> list[dict[str, Any]]:
    result = subprocess.run(["herdr", "agent", "list"], capture_output=True, check=False,
                            text=True, encoding="utf-8", errors="replace",
                            creationflags=windows_creationflags(), timeout=20)
    if result.returncode:
        raise RuntimeError(f"HERDR_AGENT_LIST_FAILED:{result.returncode}")
    try:
        agents = json.loads(result.stdout)['result']['agents']
        if not isinstance(agents, list):
            raise ValueError('agents is not a list')
        return agents
    except (ValueError, KeyError, TypeError) as exc:
        raise RuntimeError('HERDR_AGENT_LIST_FAILED:INVALID_RESPONSE') from exc


def resolve_agent(role: str, state: dict[str, Any]) -> dict[str, Any]:
    name = "arc-controller" if role == "controller" else "arc-director"
    agents = herdr_agents()
    target_key = f"{role}_target"
    saved_target = state.get(target_key)
    if saved_target:
        for record in agents:
            if record.get("pane_id") == saved_target and record.get("agent") == "codex":
                return record
    # Herdr's compact agent-list response may omit the optional human-readable
    # name.  Recover the last successfully used pane for this role from the
    # durable event history instead of treating that omission as fatal.
    for item in reversed(state.get("pending_events") or []):
        historical_target = item.get("target")
        if item.get("recipient") != role or not historical_target:
            continue
        for record in agents:
            if record.get("pane_id") == historical_target and record.get("agent") == "codex":
                state[target_key] = historical_target
                return record
    for record in agents:
        if record.get("name") == name and record.get("agent") == "codex":
            state[target_key] = record.get("pane_id")
            return record
    # With two unnamed Codex panes, the already-bound peer makes the remaining
    # pane unambiguous.  Never guess if zero or multiple candidates remain.
    peer_key = "director_target" if role == "controller" else "controller_target"
    peer_target = state.get(peer_key)
    candidates = [record for record in agents if record.get("agent") == "codex"
                  and record.get("pane_id") != peer_target]
    if len(candidates) == 1:
        state[target_key] = candidates[0].get("pane_id")
        return candidates[0]
    raise RuntimeError(f"{role.upper()}_TARGET_UNRESOLVED")


def prompt_text(item: dict[str, Any], state_path: Path) -> str:
    payload = item["payload"]
    event_id = item["event_id"]
    if item["recipient"] == "controller":
        return (
            f"ARC2 Governor event {event_id}. Read {PROJECT_ROOT / 'AGENTS.md'} and "
            f"{PROJECT_ROOT / 'orchestration/agents/ARC_CONTROLLER_SYSTEM.md'}. Claim with: py -3 {PROJECT_ROOT / 'scripts/arc2_governor.py'} "
            f"--state \"{state_path.resolve()}\" --claim-event {event_id}. Execute only this event payload: "
            f"{json.dumps(payload, sort_keys=True)}. Complete it through the same Governor CLI; never prompt Director directly. "
            "This payload is the bounded handoff. Do not read the full workflow/event history unless diagnosing a specific binding error. "
            "A terminal event authorizes ingestion of existing artifacts, not a rerun or new scientific execution."
        )
    brief_path = payload.get("brief_path")
    brief_sha = payload.get("brief_sha256")
    response_path = response_directory() / f"{Path(str(brief_path)).stem}_RESPONSE.json"
    return (
        f"ARC2 Governor review event {event_id}. Read {PROJECT_ROOT / 'orchestration/agents/ARC_DIRECTOR_SYSTEM.md'}, "
        f"{PROJECT_ROOT / 'orchestration/director/SCIENTIFIC_REVIEW_GUIDANCE.md'}, and {brief_path}. Write exactly one JSON response "
        f"to {response_path} with reviewed_brief_sha256={brief_sha} and one valid decision. "
        "CONTINUE_CONTROLLER requires next_stage and next_action; CONTINUE_DIRECTOR requires a new bound brief; "
        "REQUIRE_CHANGES requires root cause, smallest repair, frozen conditions, forbidden actions, and review requirement. "
        "Do not schedule or prompt Controller."
    )


def resume_prompt_text(item: dict[str, Any], state_path: Path) -> str:
    event_id = item["event_id"]
    return (
        f"ARC2 Governor resume event {event_id}. This event is already CLAIMED by this exact Controller pane; "
        "do not claim it again and do not create a successor event. "
        f"Read {state_path.resolve()}, {PROJECT_ROOT / 'AGENTS.md'}, and "
        f"{PROJECT_ROOT / 'orchestration/agents/ARC_CONTROLLER_SYSTEM.md'}. Continue only the existing payload: "
        f"{json.dumps(item['payload'], sort_keys=True)}. Complete the same event through the Governor CLI; "
        "never prompt Director directly."
    )


def dispatch(state: dict[str, Any], state_path: Path, timeout: int) -> str:
    queued = next((item for item in state["pending_events"] if item["status"] == "QUEUED"), None)
    if not queued:
        return "IDLE"
    if queued.get('retry_not_before', 0) > time.time():
        return 'CONFIRMED_NON_DELIVERY_BACKOFF'
    if state.get('active_event'):
        return 'RECIPIENT_WORKING_EVENT_QUEUED'
    target_key = f"{queued['recipient']}_target"
    previous_target = state.get(target_key)
    try:
        target = resolve_agent(queued["recipient"], state)
    except (RuntimeError, subprocess.TimeoutExpired) as exc:
        detail = str(exc)
        if isinstance(exc, subprocess.TimeoutExpired) or detail.startswith("HERDR_AGENT_LIST_FAILED:") or detail.endswith("_TARGET_UNRESOLVED"):
            return "RECIPIENT_TARGET_UNAVAILABLE_EVENT_QUEUED"
        raise
    # A terminal Supervisor receipt must wake an otherwise unclaimed Controller even
    # when Herdr still reports its pane as working. Suppressing that prompt used to
    # strand completed remote runs until a user message arrived. A claimed event
    # remains exclusive, and ordinary Controller tasks retain the queue-only rule.
    wake_working_controller = (
        queued.get("kind") == "REMOTE_TERMINAL"
        and queued.get("recipient") == "controller"
        and state.get("active_event") is None
    )
    if target.get("agent_status") == "working" and not wake_working_controller:
        return "RECIPIENT_WORKING_EVENT_QUEUED"
    # DELIVERED remains the compatible lifecycle state; delivery_state explicitly
    # distinguishes durable intent from confirmed input. Re-read after observation
    # so another CLI transition can never be overwritten by this snapshot.
    with state_transaction(state_path):
        fresh = load(state_path)
        queued = find_event(fresh, queued['event_id'])
        if queued['status'] != 'QUEUED' or fresh.get('active_event'):
            return 'EVENT_CHANGED_DURING_OBSERVATION'
        if fresh['disposition'] in {'PAUSED', 'TERMINAL'} and queued['kind'] != 'PRODUCER_ERROR':
            return 'WORKFLOW_QUIESCENT'
        queued.update({"status": "DELIVERED", "delivery_state": "ATTEMPTING",
                       "attempts": int(queued.get("attempts") or 0) + 1,
                       "delivered_at": now(), "target": target.get("pane_id")})
        fresh[target_key] = target.get('pane_id')
        save(state_path, fresh, f"EVENT_DELIVERY_RESERVED:{queued['event_id']}")
    command = ["herdr", "agent", "prompt", target["pane_id"], prompt_text(queued, state_path),
               "--wait", "--until", "working", "--timeout", str(timeout * 1000)]
    classification, error_code = 'UNKNOWN_DELIVERY', None
    try:
        completed = subprocess.run(command, capture_output=True, check=False, text=True,
                                   encoding="utf-8", errors="replace", timeout=timeout + 10,
                                   creationflags=windows_creationflags())
    except (OSError, subprocess.TimeoutExpired) as exc:
        error_code = type(exc).__name__
    else:
        error_code = f'EXIT_{completed.returncode}'
        if completed.returncode == 0:
            classification = 'CONFIRMED'
        elif 'agent_blocked' in (completed.stdout + completed.stderr).casefold():
            # Documented pre-input rejection only. Generic timeouts are ambiguous.
            classification = 'CONFIRMED_NON_DELIVERY'
    with state_transaction(state_path):
        fresh = load(state_path)
        item = find_event(fresh, queued['event_id'])
        # A quick claim/completion or bound Director response is stronger evidence
        # than the caller return code. Never resurrect or overwrite that transition.
        if item['status'] == 'DELIVERED':
            item.update(delivery_state=classification, delivery_error=error_code)
            if classification == 'CONFIRMED':
                item['delivery_confirmed_at'] = now()
            if classification == 'CONFIRMED_NON_DELIVERY' and item['attempts'] < 2:
                item['status'] = 'QUEUED'
                item['retry_not_before'] = time.time() + 60
            save(state_path, fresh, f'EVENT_DELIVERY_CLASSIFIED:{item["event_id"]}:{classification}')
        append_ledger(state_path, {'transition': 'HERDR_PROMPT_RESULT',
                                  'event_id': queued['event_id'], 'delivery_state': classification,
                                  'error_code': error_code})
    return 'EVENT_DELIVERED' if classification == 'CONFIRMED' else 'EVENT_DELIVERY_UNKNOWN_FAIL_CLOSED'


def resume_claimed_event(state_path: Path, event_id: str, timeout: int) -> str:
    """Wake the bound idle Controller for one already-claimed event.

    This is deliberately explicit rather than part of the daemon cycle: an
    idle pane is not sufficient evidence that arbitrary claimed work should be
    replayed.  The event identity and bound pane remain unchanged.
    """
    state = load(state_path)
    item = find_event(state, event_id)
    if item.get("recipient") != "controller":
        raise RuntimeError("ONLY_CONTROLLER_CLAIM_CAN_BE_RESUMED")
    if item.get("status") != "CLAIMED" or state.get("active_event") != event_id:
        raise RuntimeError("CLAIMED_EVENT_NOT_ACTIVE")
    target = resolve_agent("controller", state)
    pane_id = target.get("pane_id")
    if not item.get("target") or pane_id != item.get("target"):
        raise RuntimeError("CLAIMED_EVENT_TARGET_MISMATCH")
    if target.get("agent_status") == "working":
        return "CLAIMED_EVENT_ALREADY_WORKING"
    if target.get("agent_status") not in {"idle", "done"}:
        raise RuntimeError("CLAIMED_EVENT_TARGET_NOT_RESUMABLE")

    item.update({"resume_attempts": int(item.get("resume_attempts") or 0) + 1,
                 "resumed_at": now()})
    save(state_path, state, f"CLAIMED_EVENT_RESUME_RESERVED:{event_id}")
    command = ["herdr", "agent", "prompt", pane_id, resume_prompt_text(item, state_path),
               "--wait", "--until", "working", "--timeout", str(timeout * 1000)]
    try:
        completed = subprocess.run(command, capture_output=True, check=False, text=True,
                                   encoding="utf-8", errors="replace", timeout=timeout + 10,
                                   creationflags=windows_creationflags())
    except (OSError, subprocess.TimeoutExpired) as exc:
        append_ledger(state_path, {"transition": "HERDR_RESUME_PROMPT_RESULT", "event_id": event_id,
                                   "outcome": "UNKNOWN_FAIL_CLOSED", "error": type(exc).__name__})
        return "CLAIMED_EVENT_RESUME_UNKNOWN_FAIL_CLOSED"
    append_ledger(state_path, {"transition": "HERDR_RESUME_PROMPT_RESULT", "event_id": event_id,
                               "exit_code": completed.returncode})
    return "CLAIMED_EVENT_RESUMED" if completed.returncode == 0 else "CLAIMED_EVENT_RESUME_UNKNOWN_FAIL_CLOSED"


def response_directory() -> Path:
    return PROJECT_ROOT / "orchestration" / "director" / "responses"


def validate_director_brief(path_value: str) -> Path:
    """A Director response can never be recursively accepted as a new brief."""
    path = resolve_artifact(path_value)
    responses = response_directory().resolve()
    if path == responses or responses in path.parents or path.name.upper().endswith("_RESPONSE.JSON"):
        raise RuntimeError("DIRECTOR_RESPONSE_CANNOT_BE_REVIEW_BRIEF")
    return path


def repair_recursive_review_state(state_path: Path) -> None:
    """Close legacy response-derived review entries without rewriting history."""
    state = load(state_path)
    changed = False
    for item in state.get("pending_events", []):
        if item.get("kind") != "DIRECTOR_REVIEW":
            continue
        path_value = str((item.get("payload") or {}).get("brief_path") or "")
        try:
            validate_director_brief(path_value)
        except RuntimeError as exc:
            if str(exc) != "DIRECTOR_RESPONSE_CANNOT_BE_REVIEW_BRIEF":
                continue
            if item.get("status") != "COMPLETE":
                item.update({"status": "COMPLETE", "completed_at": now(),
                             "outcome": "RECURSIVE_RESPONSE_BRIEF_REJECTED"})
                changed = True
    review = state.get("review") or {}
    if review:
        try:
            validate_director_brief(str(review.get("brief_path") or ""))
        except RuntimeError as exc:
            if str(exc) == "DIRECTOR_RESPONSE_CANNOT_BE_REVIEW_BRIEF":
                state["review"] = None
                changed = True
    if changed:
        save(state_path, state, "RECURSIVE_RESPONSE_REVIEW_POINTER_CLEARED")
    else:
        append_ledger(state_path, {"transition": "RECURSIVE_RESPONSE_REVIEW_REPAIR_IDEMPOTENT"})


def consume_director_response(state: dict[str, Any], state_path: Path) -> bool:
    item = next((entry for entry in state["pending_events"]
                 if entry["kind"] == "DIRECTOR_REVIEW" and entry["status"] in {"DELIVERED", "CLAIMED"}), None)
    if not item:
        return False
    payload = item["payload"]
    try:
        validate_director_brief(payload["brief_path"])
        _, digest = validate_bound_artifact(payload["brief_path"], payload.get("brief_sha256"))
    except Exception as exc:
        item.update({"status": "COMPLETE", "completed_at": now(), "outcome": "INVALID_BOUND_BRIEF"})
        producer_error(state, item, f"{type(exc).__name__}:{exc}")
        save(state_path, state, f"DIRECTOR_BRIEF_REJECTED:{item['event_id']}")
        return True
    matches: list[tuple[Path, dict[str, Any]]] = []
    for candidate in sorted(response_directory().glob("*.json")):
        try:
            value = json.loads(candidate.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            continue
        if value.get("reviewed_brief_sha256") == digest:
            matches.append((candidate, value))
    if not matches:
        return False
    response_path, response = matches[-1]
    decision = response.get("decision") or response.get("scientific_outcome")
    if decision not in DECISIONS:
        item.update({"status": "COMPLETE", "completed_at": now(), "outcome": "INVALID_RESPONSE"})
        producer_error(state, {**item, "producer": "director"}, "DIRECTOR_RESPONSE_DECISION_INVALID")
        save(state_path, state, f"DIRECTOR_RESPONSE_REJECTED:{item['event_id']}")
        return True
    item.update({"status": "COMPLETE", "completed_at": now(), "outcome": decision,
                 "response_path": str(response_path.resolve()), "response_sha256": sha256_file(response_path)})
    state["review"] = None
    if decision == "CONTINUE_CONTROLLER":
        if not response.get("next_stage") or not response.get("next_action"):
            producer_error(state, {**item, "producer": "director"}, "CONTINUE_CONTROLLER_FIELDS_MISSING")
        else:
            state.update({"disposition": "CONTINUE_CONTROLLER", "stage": response["next_stage"],
                          "next_action": response["next_action"], "pause_reason": None})
            enqueue(state, event("DIRECTOR_DECISION", "controller", "director", {
                "decision": decision, "response_path": str(response_path.resolve()),
                "response_sha256": sha256_file(response_path), "stage": response["next_stage"],
                "next_action": response["next_action"],
            }))
    elif decision == "CONTINUE_DIRECTOR":
        successor = response.get("next_review_brief")
        successor_sha = response.get("next_review_brief_sha256")
        reason = response.get("next_review_reason")
        if not successor or not successor_sha or not reason:
            producer_error(state, {**item, "producer": "director"}, "CONTINUE_DIRECTOR_FIELDS_MISSING")
        else:
            state.update({"disposition": "REVIEW_REQUIRED", "review": {
                "brief_path": successor, "brief_sha256": successor_sha, "reason": reason,
            }})
            enqueue(state, event("DIRECTOR_REVIEW", "director", "director", state["review"]))
    elif decision == "REQUIRE_CHANGES":
        state.update({"disposition": "CONTINUE_CONTROLLER", "next_action": "APPLY_DIRECTOR_REQUIRED_CHANGES"})
        enqueue(state, event("DIRECTOR_DECISION", "controller", "director", {
            "decision": decision, "response_path": str(response_path.resolve()),
            "response_sha256": sha256_file(response_path),
        }))
    elif decision == "PAUSED":
        state.update({"disposition": "PAUSED", "pause_reason": "PAUSED_BY_DIRECTOR"})
    else:
        state.update({"disposition": "TERMINAL", "terminal_reason": "TERMINAL_BY_DIRECTOR"})
    save(state_path, state, f"DIRECTOR_RESPONSE_CONSUMED:{item['event_id']}")
    return True


@state_mutation
def claim_event(state_path: Path, event_id: str) -> None:
    state = load(state_path)
    item = find_event(state, event_id)
    if item["status"] == "CLAIMED":
        if state.get("active_event") != event_id:
            raise RuntimeError("CLAIMED_EVENT_NOT_ACTIVE")
        caller_pane = os.environ.get("HERDR_PANE_ID")
        if not caller_pane or item.get("target") != caller_pane:
            raise RuntimeError("EVENT_ALREADY_CLAIMED_BY_OTHER_PANE")
        append_ledger(state_path, {"transition": "EVENT_CLAIM_IDEMPOTENT", "event_id": event_id,
                                   "target": caller_pane})
        return
    if item["status"] not in {"DELIVERED", "QUEUED"}:
        raise RuntimeError("EVENT_NOT_CLAIMABLE")
    if state.get("active_event") not in {None, event_id}:
        raise RuntimeError("ANOTHER_EVENT_ALREADY_ACTIVE")
    item.update({"status": "CLAIMED", "claimed_at": now(), "delivery_state": "ACKNOWLEDGED"})
    state["active_event"] = event_id
    save(state_path, state, f"EVENT_CLAIMED:{event_id}")


@state_mutation
def complete_event(state_path: Path, args: argparse.Namespace) -> None:
    state = load(state_path)
    item = find_event(state, args.complete_event)
    if item["status"] not in {"CLAIMED", "DELIVERED"}:
        raise RuntimeError("EVENT_NOT_COMPLETABLE")
    disposition = args.next_disposition or state["disposition"]
    if disposition not in DISPOSITIONS:
        raise RuntimeError("INVALID_COMPLETION_DISPOSITION")
    if disposition == "REVIEW_REQUIRED":
        if not args.review_brief or not args.review_brief_sha256 or not args.review_reason:
            raise RuntimeError("REVIEW_BINDING_REQUIRED")
        validate_director_brief(args.review_brief)
        validate_bound_artifact(args.review_brief, args.review_brief_sha256)
        state["review"] = {"brief_path": args.review_brief, "brief_sha256": args.review_brief_sha256,
                           "reason": args.review_reason}
    if disposition == "WAIT_REMOTE":
        if not args.remote_job_file:
            raise RuntimeError("REMOTE_JOB_FILE_REQUIRED")
        _, remote = _read_remote_job_manifest(args.remote_job_file)
        if state.get('remote_job') and state['remote_job'] != remote:
            raise RuntimeError('ACTIVE_REMOTE_JOB_PROTECTED')
        persist_remote_binding(state_path, remote)
        state["remote_job"] = remote
    else:
        state["remote_job"] = None
    item.update({"status": "COMPLETE", "completed_at": now(), "outcome": args.outcome or "COMPLETE"})
    state.update({"active_event": None, "disposition": disposition,
                  "stage": args.stage or state["stage"], "next_action": args.next_action,
                  "pause_reason": args.pause_reason if disposition == "PAUSED" else None,
                  "terminal_reason": args.terminal_reason if disposition == "TERMINAL" else None})
    save(state_path, state, f"EVENT_COMPLETED:{args.complete_event}")


def _read_remote_job_manifest(value: str) -> tuple[Path, dict[str, Any]]:
    remote_path = resolve_artifact(value)
    remote = json.loads(remote_path.read_text(encoding="utf-8-sig"))
    required = ("job_id", "ssh_target", "output_root", "terminal_receipt", "primary_process")
    if any(not remote.get(key) for key in required):
        raise RuntimeError("REMOTE_JOB_MANIFEST_INCOMPLETE")
    if not re.fullmatch(r'[A-Za-z0-9_-]+', str(remote['job_id'])):
        raise RuntimeError('REMOTE_JOB_ID_UNSAFE')
    if not str(remote['output_root']).startswith('/') or remote['terminal_receipt'] != remote['output_root'].rstrip('/') + '/TERMINAL_RECEIPT.json':
        raise RuntimeError('REMOTE_TERMINAL_PATH_NOT_BOUND')
    primary = remote["primary_process"]
    if not isinstance(primary, dict) or not isinstance(primary.get("pid"), int) or not primary.get("role"):
        raise RuntimeError("REMOTE_JOB_PRIMARY_PROCESS_INCOMPLETE")
    return remote_path, remote


def persist_remote_binding(state_path: Path, remote: dict) -> None:
    """One registry for normal handoff, recovery and adoption paths."""
    destination = state_path.parent / 'remote_jobs' / f'{remote["job_id"]}.json'
    if destination.exists():
        if json.loads(destination.read_text(encoding='utf-8-sig')) != remote:
            raise RuntimeError('IMMUTABLE_REMOTE_BINDING_CONFLICT')
    else:
        atomic(destination, remote)


@state_mutation
def replace_stale_wait_remote_job(state_path: Path, args: argparse.Namespace) -> None:
    """Atomically replace a proven pre-model failed remote registration."""
    state = load(state_path)
    previous = state.get("remote_job")
    if state.get("disposition") != "WAIT_REMOTE" or state.get("active_event") is not None or not isinstance(previous, dict):
        raise RuntimeError("STALE_WAIT_REMOTE_REPLACE_NOT_ALLOWED")
    receipt_path = resolve_artifact(args.failure_receipt)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8-sig"))
    if receipt.get("job_id") != previous.get("job_id"):
        raise RuntimeError("STALE_FAILURE_RECEIPT_JOB_MISMATCH")
    if receipt.get("scientific_authorization_consumed") is not False or receipt.get("model_loaded") is not False or receipt.get("gpu_used") is not False:
        raise RuntimeError("STALE_FAILURE_RECEIPT_NOT_PREMODEL")
    _, remote = _read_remote_job_manifest(args.replace_stale_wait_remote_job)
    if remote["job_id"] == previous.get("job_id") or remote["output_root"] == previous.get("output_root"):
        raise RuntimeError("REMOTE_JOB_REPLACEMENT_NOT_FRESH")
    persist_remote_binding(state_path, remote)
    state.update({"remote_job": remote,
                  "next_action": args.next_action or "SUPERVISOR_MONITORS_REPAIRED_BOUND_REMOTE_JOB",
                  "pause_reason": None, "terminal_reason": None})
    save(state_path, state, f"STALE_WAIT_REMOTE_REPLACED:{previous['job_id']}:{remote['job_id']}")

@state_mutation
def resume_paused_remote_job(state_path: Path, args: argparse.Namespace) -> None:
    """Atomically register a detached job after a pre-model paused-stage repair.

    This narrow recovery path cannot change science or bypass an active event:
    it is available only from a quiescent PAUSED stage and only with a complete
    Supervisor-readable remote-job manifest.
    """
    state = load(state_path)
    if state.get("disposition") != "PAUSED" or state.get("active_event") is not None:
        raise RuntimeError("PAUSED_REMOTE_RESUME_NOT_ALLOWED")
    if state.get("remote_job") is not None:
        raise RuntimeError("REMOTE_JOB_ALREADY_REGISTERED")
    _, remote = _read_remote_job_manifest(args.resume_paused_remote_job)
    persist_remote_binding(state_path, remote)

    state.update({"disposition": "WAIT_REMOTE", "remote_job": remote,
                  "next_action": args.next_action or "SUPERVISOR_MONITORS_BOUND_REMOTE_JOB",
                  "pause_reason": None, "terminal_reason": None})
    save(state_path, state, f"PAUSED_REMOTE_JOB_REGISTERED:{remote['job_id']}")


@state_mutation
def register_remote_job(state_path: Path, value: str) -> None:
    """Adopt an existing job for monitoring only; never launch or rearm it."""
    state = load(state_path)
    _, remote = _read_remote_job_manifest(value)
    if state.get('active_event'):
        raise RuntimeError('ACTIVE_EVENT_MUST_REGISTER_ON_COMPLETION')
    previous = state.get('remote_job')
    if previous and previous != remote:
        raise RuntimeError('ACTIVE_REMOTE_JOB_PROTECTED')
    if state.get('review'):
        raise RuntimeError('PENDING_REVIEW_PROTECTED')
    if remote.get('pipeline_status_path'):
        process = remote['primary_process']
        if not process.get('start_ticks') or not process.get('boot_id') or not remote.get('source_commit'):
            raise RuntimeError('ADOPTED_PROCESS_CREATION_IDENTITY_REQUIRED')
        if remote['pipeline_status_path'] != remote['output_root'].rstrip('/') + '/PIPELINE_STATE.json':
            raise RuntimeError('PIPELINE_STATUS_MUST_BE_BOUND_TO_OUTPUT_ROOT')
    persist_remote_binding(state_path, remote)
    if previous == remote and state['disposition'] == 'WAIT_REMOTE':
        return
    state.update(remote_job=remote, disposition='WAIT_REMOTE',
                 stage=remote.get('stage') or state['stage'],
                 next_action='SUPERVISOR_MONITORS_BOUND_REMOTE_JOB',
                 pause_reason=None, terminal_reason=None)
    save(state_path, state, f'REMOTE_JOB_REGISTERED:{remote["job_id"]}')


def delivery_alerts(state: dict[str, Any], state_path: Path) -> list[str]:
    """One durable alert per incident, no repeated Agent wake or state churn."""
    alerts = []
    for item in state['pending_events']:
        if item['status'] != 'DELIVERED':
            continue
        stamp = item.get('delivered_at')
        grace = 1800 if item['recipient'] == 'director' and item.get('delivery_state') == 'CONFIRMED' else 120
        if not stamp or (datetime.now(timezone.utc) - datetime.fromisoformat(stamp.replace('Z', '+00:00'))).total_seconds() < grace:
            continue
        incident = item.get('delivery_state', 'LEGACY_UNCONFIRMED')
        marker = state_path.parent / 'delivery_alerts' / f'{item["event_id"]}_{item.get("attempts", 0)}.json'
        if not marker.exists():
            atomic(marker, {'event_id': item['event_id'], 'incident': incident,
                            'observed_at': now(), 'action': 'RECONCILE_ACK_OR_RESULT_WITHOUT_BLIND_REDELIVERY'})
        alerts.append(item['event_id'])
    return alerts


def write_service_state(path: Path | None, status: str, state: dict[str, Any], detail: str | None = None) -> None:
    if not path:
        return
    value = {"service": "ARC2_GOVERNOR", "pid": os.getpid(), "status": status,
             "heartbeat_at": now(), "workflow_disposition": state["disposition"],
             "workflow_stage": state["stage"], "queued_events": sum(
                 1 for item in state["pending_events"] if item["status"] == "QUEUED")}
    if detail:
        value["detail"] = detail
    value['remote_job_id'] = (state.get('remote_job') or {}).get('job_id')
    value['unacknowledged_events'] = [item['event_id'] for item in state['pending_events']
                                     if item['status'] == 'DELIVERED']
    atomic(path, value)


def cycle(state_path: Path, *, agent_timeout_seconds: int, **_: Any) -> tuple[str, dict[str, Any]]:
    with state_transaction(state_path):
        state = load(state_path)
        changed = import_external_events(state, state_path)
        changed = synthesize_state_event(state) or changed
        changed = validate_queued_reviews(state) or changed
        if changed:
            save(state_path, state, "EVENTS_RECONCILED")
            state = load(state_path)
        if consume_director_response(state, state_path):
            state = load(state_path)
        alerts = delivery_alerts(state, state_path)
    result = dispatch(state, state_path, agent_timeout_seconds)
    if result == 'IDLE' and alerts:
        result = 'DELIVERY_ACK_REQUIRED'
    return result, load(state_path)


def status_line(result: str, state: dict[str, Any]) -> str:
    queued = sum(1 for item in state["pending_events"] if item["status"] == "QUEUED")
    return f"{now()} [ARC2 {state['disposition']}] result={result} stage={state['stage']} queued={queued}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--daemon", action="store_true")
    parser.add_argument("--migrate-legacy", action="store_true")
    parser.add_argument("--repair-recursive-reviews", action="store_true")
    parser.add_argument("--claim-event")
    parser.add_argument("--resume-claimed-event")
    parser.add_argument("--complete-event")
    parser.add_argument("--resume-paused-remote-job")
    parser.add_argument('--register-remote-job', help='Monitor an existing immutable job; does not launch it')
    parser.add_argument("--replace-stale-wait-remote-job")
    parser.add_argument("--failure-receipt")
    parser.add_argument("--next-disposition", choices=sorted(DISPOSITIONS))
    parser.add_argument("--stage")
    parser.add_argument("--next-action")
    parser.add_argument("--outcome")
    parser.add_argument("--review-brief")
    parser.add_argument("--review-brief-sha256")
    parser.add_argument("--review-reason")
    parser.add_argument("--remote-job-file")
    parser.add_argument("--pause-reason")
    parser.add_argument("--terminal-reason")
    parser.add_argument("--agent-timeout-seconds", type=int, default=20)
    parser.add_argument("--idle-seconds", type=int, default=10)
    parser.add_argument("--paused-seconds", type=int, default=30)
    parser.add_argument("--controller-retry-seconds", type=int, default=300, help="deprecated compatibility option")
    parser.add_argument("--lock-path", type=Path)
    parser.add_argument("--pid-file", type=Path)
    parser.add_argument("--service-state", type=Path)
    args = parser.parse_args()
    modes = sum(bool(value) for value in (args.migrate_legacy, args.repair_recursive_reviews, args.claim_event,
                                          args.resume_claimed_event, args.complete_event,
                                          args.resume_paused_remote_job, args.replace_stale_wait_remote_job,
                                          args.register_remote_job))
    if modes > 1:
        parser.error("migration, claim, and completion are mutually exclusive")
    if args.migrate_legacy:
        migrate_legacy(args.state)
        return 0
    if args.repair_recursive_reviews:
        repair_recursive_review_state(args.state)
        return 0
    if args.claim_event:
        claim_event(args.state, args.claim_event)
        return 0
    if args.resume_claimed_event:
        print(resume_claimed_event(args.state, args.resume_claimed_event, args.agent_timeout_seconds))
        return 0
    if args.complete_event:
        complete_event(args.state, args)
        return 0
    if args.resume_paused_remote_job:
        resume_paused_remote_job(args.state, args)
        return 0
    if args.register_remote_job:
        register_remote_job(args.state, args.register_remote_job)
        return 0
    if args.replace_stale_wait_remote_job:
        if not args.failure_receipt:
            parser.error("--replace-stale-wait-remote-job requires --failure-receipt")
        replace_stale_wait_remote_job(args.state, args)
        return 0
    lock = GovernorLock(args.lock_path or args.state.parent / "arc2_governor.lock")
    if not lock.acquire():
        print("ARC2_GOVERNOR_ALREADY_RUNNING", file=sys.stderr)
        return 3
    try:
        if args.pid_file:
            args.pid_file.parent.mkdir(parents=True, exist_ok=True)
            args.pid_file.write_text(str(os.getpid()), encoding="ascii")
        while True:
            result, state = cycle(args.state, agent_timeout_seconds=args.agent_timeout_seconds)
            write_service_state(args.service_state, result, state)
            print(status_line(result, state), flush=True)
            if args.once or not args.daemon:
                return 0
            time.sleep(args.paused_seconds if result == "IDLE" and state["disposition"] == "PAUSED" else args.idle_seconds)
    except Exception as exc:
        try:
            write_service_state(args.service_state, "ERROR", load(args.state), f"{type(exc).__name__}:{exc}")
        except Exception:
            pass
        raise
    finally:
        if args.pid_file:
            try:
                if args.pid_file.read_text(encoding="ascii").strip() == str(os.getpid()):
                    args.pid_file.unlink()
            except OSError:
                pass
        lock.release()


if __name__ == "__main__":
    raise SystemExit(main())
