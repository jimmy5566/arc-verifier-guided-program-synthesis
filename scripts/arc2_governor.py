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
    "remote_job", "last_transition", "pause_reason", "terminal_reason", "updated_at", "experiment_queue_path",
    "agent_routes", "controller_role",
    "paused_roles", "pause_scope", "remote_jobs", "active_events",
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
        "terminal_reason": None, "updated_at": now(), "paused_roles": {},
        "pause_scope": "global", "remote_jobs": {},
    }
    for key, value in defaults.items():
        state.setdefault(key, value)
    # Event statuses and the immutable remote registry are authoritative. These
    # singular fields remain display projections for older CLI/status readers.
    if state.get("remote_job") and not state["remote_jobs"]:
        job = state["remote_job"]
        job_id = job.get('job_id') or job.get('run_id')
        if job_id:
            state["remote_jobs"][job_id] = job
    sync_activity(state)
    return state


def item_role(item: dict, state: dict) -> str:
    if item.get("recipient") == "controller" and "agent_routes" in state:
        return item.get("route_role") or item.get("payload", {}).get("controller_role") or "unbound"
    return item.get("recipient", "unbound")


def occupied(state: dict, role: str, excluding: str | None = None) -> bool:
    return any(e["event_id"] != excluding and e["status"] in {"DELIVERED", "CLAIMED"}
               and item_role(e, state) == role for e in state["pending_events"])


def sync_activity(state: dict) -> None:
    claimed = [e for e in state.get("pending_events", []) if e.get("status") == "CLAIMED"]
    state["active_events"] = {item_role(e, state): e["event_id"] for e in claimed}
    state["active_event"] = claimed[0]["event_id"] if claimed else None


def globally_stopped(state: dict) -> bool:
    return state["disposition"] in {"PAUSED", "TERMINAL"} and state.get("pause_scope", "global") == "global"


def owner_stop_snapshot(state: dict) -> dict:
    if 'agent_routes' in state and globally_stopped(state):
        return {k: state.get(k) for k in ('disposition', 'pause_scope', 'pause_reason', 'terminal_reason')}
    return {}


def remote_role(remote: dict) -> str:
    return remote.get("controller_role") or remote.get("completion_recipient", "controller")


def sync_remote_display(state: dict) -> None:
    state["remote_job"] = next(iter(state["remote_jobs"].values()), None)


def protect_remote_owner(state: dict, remote: dict, origin: dict | None = None) -> None:
    if "agent_routes" in state:
        if remote.get("completion_recipient", "controller") == "controller" and remote.get("controller_role") not in {"oracle", "selector"}:
            raise RuntimeError("REMOTE_CONTROLLER_ROLE_REQUIRED")
        if origin and origin.get("recipient") == "controller" and remote.get("controller_role") != item_role(origin, state):
            raise RuntimeError("REMOTE_JOB_CONTROLLER_ROLE_MISMATCH")
    for previous in state["remote_jobs"].values():
        if remote_role(previous) == remote_role(remote) and previous != remote:
            raise RuntimeError("ACTIVE_REMOTE_JOB_PROTECTED")
        # A waiter is CPU-only and may be monitored concurrently. GPU workers
        # serialize at the host-local inherited-FD lock, not per Agent role.
        resource = remote.get('gpu_resource')
        if resource and resource == previous.get('gpu_resource') and previous != remote:
            if remote.get('gpu_lock_policy') != 'HOST_LOCAL_INHERITED_FD_V1':
                raise RuntimeError('SHARED_GPU_REQUIRES_ATOMIC_LOCK')


def retire_completed_role_jobs(state: dict, remote: dict) -> list[dict]:
    """Reconcile a terminal job re-registered after its terminal event import.

    Only a completed same-role terminal handoff with an exact Supervisor
    COMPLETE receipt and absent observed process can retire a stale entry.
    Registry manifests and historical events remain immutable. Live/unknown
    jobs still reach protect_remote_owner and fail closed.
    """
    retired = []
    for job_id, previous in list(state['remote_jobs'].items()):
        if previous == remote or remote_role(previous) != remote_role(remote):
            continue
        for item in state.get('pending_events', []):
            payload = item.get('payload', {})
            if (item.get('status') != 'COMPLETE' or
                    payload.get('round_id') != job_id or
                    payload.get('controller_role') != previous.get('controller_role') or
                    payload.get('terminal_status') != 'COMPLETE' or
                    not payload.get('bound_remote_job')):
                continue
            try:
                path, digest = validate_bound_artifact(
                    payload['terminal_receipt'], payload.get('terminal_receipt_sha256'))
                receipt = json.loads(path.read_text(encoding='utf-8-sig'))
            except (KeyError, RuntimeError, ValueError, OSError):
                continue
            if (receipt.get('job_id') != job_id or receipt.get('status') != 'COMPLETE' or
                    receipt.get('execution_source_commit') != previous.get('source_commit') or
                    receipt.get('receipt_origin') != 'LOCAL_SUPERVISOR_PIPELINE_OBSERVATION_V1' or
                    receipt.get('observed_process_identity') != {}):
                continue
            state['remote_jobs'].pop(job_id)
            retired.append({'job_id': job_id, 'completed_terminal_event': item['event_id'],
                            'terminal_receipt_sha256': digest})
            break
    return retired


def retire_claimed_kaggle_terminal(state: dict, item: dict) -> None:
    """Retire only the exact verified terminal being ingested in this event.

    A fast completion can be imported before its launch event finishes its
    WAIT_REMOTE handoff. Preserve the registry; reconcile that stale live view.
    """
    payload = item.get('payload', {})
    old = state['remote_jobs'].get(payload.get('round_id'))
    if not old or old.get('backend') != 'kaggle':
        return
    if (item.get('status') != 'CLAIMED' or not payload.get('bound_remote_job') or
            item_role(item, state) != remote_role(old)):
        raise RuntimeError('KAGGLE_TERMINAL_OWNER_NOT_CLAIMED')
    path, digest = validate_bound_artifact(payload['terminal_receipt'], payload['terminal_receipt_sha256'])
    receipt = json.loads(path.read_text(encoding='utf-8-sig'))
    if (receipt.get('receipt_origin') != 'LOCAL_SUPERVISOR_KAGGLE_OBSERVATION_V1' or
            receipt.get('job_id') != old['job_id'] or
            receipt.get('execution_source_commit') != old['source_commit'] or
            receipt.get('status') != payload.get('terminal_status') or
            receipt.get('status') not in {'COMPLETE', 'INFRA_FAILED', 'CONTROLLED_STOP', 'FAILED_OR_PARTIAL_NO_RETRY'}):
        raise RuntimeError('KAGGLE_TERMINAL_NOT_VERIFIED')
    worker_path, _ = validate_bound_artifact(receipt['worker_terminal_local'], receipt['worker_terminal_sha256'])
    sys.path.insert(0, str(PROJECT_ROOT / 'orchestration' / 'supervisor'))
    from kaggle_jobs import bound_terminal
    worker = bound_terminal(old, json.loads(worker_path.read_bytes()), receipt['platform_identity'])
    if worker['status'] != receipt['status']:
        raise RuntimeError('KAGGLE_TERMINAL_STATUS_MISMATCH')
    state['remote_jobs'].pop(old['job_id'])
    sync_remote_display(state)
    item['verified_terminal_binding_retired'] = {'job_id': old['job_id'], 'terminal_receipt_sha256': digest}


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
    # A completed stage is not a completed owner queue. Mechanical check only:
    # Director owns item status/evidence; Governor never interprets science.
    if state.get('disposition') == 'TERMINAL' and state.get('experiment_queue_path'):
        issue, queue = experiment_queue_terminal_issue(state['experiment_queue_path'])
        if issue:
            state.update(disposition='PAUSED', terminal_reason=None, pause_reason=issue,
                         next_action=queue.get('next_action') or 'READ_PENDING_EXPERIMENT_QUEUE')
    compacted = compact_state(state)
    state.clear()
    state.update(compacted)
    state["revision"] = int(state.get("revision") or 0) + 1
    state["last_transition"] = transition
    state["updated_at"] = now()
    atomic(path, state)
    append_ledger(path, {"transition": transition, "revision": state["revision"],
                         "disposition": state["disposition"], "stage": state["stage"]})


def experiment_queue_terminal_issue(value: str) -> tuple[str | None, dict]:
    try:
        queue = json.loads(resolve_artifact(value).read_text(encoding='utf-8-sig'))
        items = queue['items']
        if not isinstance(items, list) or not items or any(not isinstance(item, dict) for item in items):
            return 'EXPERIMENT_QUEUE_UNVERIFIED', queue
        required = [item for item in items if item.get('required', True)]
        if not required or any(item.get('status') != 'COMPLETE' for item in required):
            return 'EXPERIMENT_QUEUE_INCOMPLETE', queue
        return None, queue
    except (OSError, ValueError, KeyError, TypeError, RuntimeError):
        return 'EXPERIMENT_QUEUE_UNVERIFIED', {}


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
    # Preserve the originating Controller identity, never borrow the current one.
    if source.get("payload", {}).get("controller_role"):
        payload["controller_role"] = source["payload"]["controller_role"]
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
            if value.get("schema_version") != 1 or value.get("kind") not in {"REMOTE_TERMINAL", "CONTROLLER_TASK", "REVIEW_REQUEST", "DIRECTOR_TASK"}:
                raise RuntimeError("EXTERNAL_EVENT_SCHEMA_INVALID")
            payload = value.get("payload")
            if not isinstance(payload, dict):
                raise RuntimeError("EXTERNAL_EVENT_PAYLOAD_INVALID")
            owner_stop = owner_stop_snapshot(state)
            if value.get('producer') == 'owner' and value.get('resume_global') is True:
                owner_stop = {}
            recipient = value.get("recipient") or ("director" if value["kind"] in {"REVIEW_REQUEST", "DIRECTOR_TASK"} else "controller")
            if value['kind'] == 'DIRECTOR_TASK' and recipient != 'director':
                raise RuntimeError('DIRECTOR_TASK_RECIPIENT_MISMATCH')
            item = event(value["kind"], recipient, value.get("producer") or "supervisor", payload)
            already_known = any(entry['event_id'] == item['event_id'] for entry in state['pending_events'])
            terminal = value['kind'] == 'REMOTE_TERMINAL' or payload.get('bound_remote_job')
            if not already_known and terminal:
                job_id = payload.get('round_id')
                if not re.fullmatch(r'[A-Za-z0-9_-]+', str(job_id)):
                    raise RuntimeError('REMOTE_JOB_ID_UNSAFE')
                registered = state['remote_jobs'].get(job_id)
                if not registered:
                    registry = state_path.parent / 'remote_jobs' / f'{job_id}.json'
                    if registry.is_file():
                        registered = json.loads(registry.read_text(encoding='utf-8-sig'))
                    elif state['remote_jobs'] or payload.get('bound_remote_job'):
                        raise RuntimeError('TERMINAL_EVENT_ACTIVE_JOB_MISMATCH')
                if registered:
                    if recipient != registered.get('completion_recipient', 'controller'):
                        raise RuntimeError('TERMINAL_EVENT_RECIPIENT_MISMATCH')
                    if registered.get('controller_role') != payload.get('controller_role'):
                        raise RuntimeError('TERMINAL_EVENT_CONTROLLER_ROLE_MISMATCH')
            added = enqueue(state, item)
            changed = added or changed
            if added and terminal:
                state['remote_jobs'].pop(payload.get('round_id'), None)
                sync_remote_display(state)
            if added and value["kind"] == "REMOTE_TERMINAL" and not globally_stopped(state):
                state.update({"disposition": "CONTINUE_CONTROLLER",
                              "next_action": "INGEST_FREEZE_AND_ROUTE_REMOTE_RESULT",
                              "terminal_reason": None})
            elif added and value["kind"] == "CONTROLLER_TASK":
                state.update({"disposition": "CONTINUE_CONTROLLER",
                              "stage": payload.get("stage") or state["stage"],
                              "controller_role": payload.get("controller_role"),
                              "next_action": payload.get("next_action"),
                              "pause_reason": None, "terminal_reason": None, "pause_scope": "agent"})
                state['paused_roles'].pop(item_role(item, state), None)
            elif added and value["kind"] == "REVIEW_REQUEST":
                state.update({"disposition": "REVIEW_REQUIRED", "review": {
                    "brief_path": payload.get("brief_path"),
                    "brief_sha256": payload.get("brief_sha256"),
                    "reason": payload.get("review_reason"),
                    **({"controller_role": payload["controller_role"]} if payload.get("controller_role") else {}),
                }})
            elif added and value['kind'] == 'DIRECTOR_TASK' and (not (globally_stopped(state) and state['disposition'] == 'PAUSED')
                    or (value.get('producer') == 'owner' and value.get('resume_global') is True)):
                state.update(disposition='REVIEW_REQUIRED', review=None,
                             stage=payload.get('stage') or state['stage'],
                             next_action='DIRECTOR_CONTINUES_BOUND_OWNER_TASK', terminal_reason=None)
                state['paused_roles'].pop('director', None)
            # A routine result or queued work cannot revoke a global owner STOP.
            state.update(owner_stop)
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
    # Named multi-agent operation enqueues successors atomically at completion.
    # Never manufacture work from another agent's last-written display fields.
    if 'agent_routes' in state:
        return False
    if state["disposition"] == "CONTINUE_CONTROLLER":
        if any(item["recipient"] == "controller" and item["status"] != "COMPLETE"
               for item in state["pending_events"]):
            return False
        payload = {"stage": state["stage"], "next_action": state["next_action"]}
        if state.get("controller_role"):
            payload["controller_role"] = state["controller_role"]
        return enqueue(state, event("CONTROLLER_TASK", "controller", "director", payload))
    if state["disposition"] == "REVIEW_REQUIRED" and state.get("review"):
        if any(item["recipient"] == "director" and item["status"] != "COMPLETE"
               for item in state["pending_events"]):
            return False
        review = state["review"]
        payload = {"brief_path": review.get("brief_path"), "brief_sha256": review.get("brief_sha256"),
                   "review_reason": review.get("reason")}
        if review.get("controller_role"):
            payload["controller_role"] = review["controller_role"]
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


def validate_agent_routes(routes: Any) -> None:
    if not isinstance(routes, dict) or not routes or set(routes) - {"director", "oracle", "selector"}:
        raise RuntimeError("AGENT_ROUTES_INVALID")
    names, panes, threads = set(), set(), set()
    for role, route in routes.items():
        if not isinstance(route, dict) or not re.fullmatch(r"[a-z][a-z0-9_-]{0,31}", str(route.get("agent_name", ""))):
            raise RuntimeError("AGENT_ROUTE_NAME_INVALID")
        if not isinstance(route.get("pane_id"), str) or not route["pane_id"]:
            raise RuntimeError("AGENT_ROUTE_PANE_REQUIRED")
        if not isinstance(route.get("held"), bool):
            raise RuntimeError("AGENT_ROUTE_HOLD_REQUIRED")
        thread = route.get("thread_id")
        if thread is not None and not re.fullmatch(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", str(thread)):
            raise RuntimeError("AGENT_ROUTE_THREAD_INVALID")
        if thread and thread in threads:
            raise RuntimeError("AGENT_ROUTE_THREAD_NOT_UNIQUE")
        if route["agent_name"] in names or route["pane_id"] in panes:
            raise RuntimeError("AGENT_ROUTES_NOT_UNIQUE")
        names.add(route["agent_name"])
        panes.add(route["pane_id"])
        if thread:
            threads.add(thread)


def event_role(item: dict[str, Any], state: dict[str, Any]) -> str:
    role = item.get("recipient")
    if role not in {"controller", "director"}:
        raise RuntimeError("EVENT_RECIPIENT_INVALID")
    if role == "controller" and "agent_routes" in state:
        role = item.get("payload", {}).get("controller_role")
        if role not in {"oracle", "selector"}:
            raise RuntimeError("CONTROLLER_ROLE_REQUIRED")
    return role


def route_held(role: str, state: dict[str, Any]) -> bool:
    routes = state.get("agent_routes")
    if routes is None:
        return False
    validate_agent_routes(routes)
    if role not in routes:
        raise RuntimeError("AGENT_ROUTE_BINDING_REQUIRED")
    return routes[role]["held"]


def resolve_agent(role: str, state: dict[str, Any], *, identity_only: bool = False) -> dict[str, Any]:
    """Exact live identity only: no historical-pane or focused-pane fallback."""
    routes = state.get("agent_routes")
    if routes is not None:
        validate_agent_routes(routes)
        if role not in routes:
            raise RuntimeError("AGENT_ROUTE_BINDING_REQUIRED")
        route = routes[role]
        if route["held"] and not identity_only:
            raise RuntimeError("RECIPIENT_OWNER_HELD")
        matches = [a for a in herdr_agents() if a.get("agent") == "codex"
                   and a.get("name") == route["agent_name"]]
        if len(matches) != 1 or matches[0].get("pane_id") != route["pane_id"]:
            raise RuntimeError("AGENT_ROUTE_IDENTITY_MISMATCH")
        if route.get("terminal_id") and matches[0].get("terminal_id") != route["terminal_id"]:
            raise RuntimeError("AGENT_ROUTE_TERMINAL_MISMATCH")
        return matches[0]
    # Legacy single-Controller operation accepts an explicitly bound pane or an
    # exact unique role name, but never guesses from event history/other panes.
    if role not in {"controller", "director"}:
        raise RuntimeError("AGENT_ROUTE_BINDING_REQUIRED")
    name = "arc-controller" if role == "controller" else "arc-director"
    agents = [a for a in herdr_agents() if a.get("agent") == "codex"]
    saved = state.get(f"{role}_target")
    matches = [a for a in agents if a.get("pane_id") == saved] if saved else [a for a in agents if a.get("name") == name]
    if len(matches) != 1:
        raise RuntimeError(f"{role.upper()}_TARGET_UNRESOLVED")
    if saved and matches[0].get("name") not in {None, name}:
        raise RuntimeError("AGENT_ROUTE_IDENTITY_MISMATCH")
    return matches[0]


def event_caller_pane(item: dict[str, Any], state: dict[str, Any]) -> str:
    """Validate a caller without trusting a shared executor's inherited pane.

    A thread binding is installed only after observing that thread in the exact
    visible agent. It never changes environment variables or permits another
    thread/pane/session to claim work. Older standalone callers remain pane-bound.
    """
    role = event_role(item, state)
    binding = state["agent_routes"][role]
    if (item.get("target") != binding["pane_id"]
            or item.get("target_agent") != binding["agent_name"]):
        raise RuntimeError("EVENT_CLAIM_TARGET_MISMATCH")
    if binding.get("thread_id"):
        if os.environ.get("CODEX_THREAD_ID") != binding["thread_id"]:
            raise RuntimeError("EVENT_CALLER_THREAD_MISMATCH")
        live = resolve_agent(role, state, identity_only=True)
        if (live.get("pane_id") != item["target"]
                or (binding.get("terminal_id") and item.get("target_terminal_id") != binding["terminal_id"])):
            raise RuntimeError("EVENT_CALLER_LIVE_IDENTITY_MISMATCH")
    elif os.environ.get("HERDR_PANE_ID") != item.get("target"):
        raise RuntimeError("EVENT_CLAIM_TARGET_MISMATCH")
    return binding["pane_id"]


@state_mutation
def configure_agent_routes(state_path: Path, value: str) -> None:
    """Operator-only binding/hold edit; never activates or prompts any task."""
    routes = json.loads(resolve_artifact(value).read_text(encoding="utf-8-sig"))["agent_routes"]
    validate_agent_routes(routes)
    state = load(state_path)
    for active in (e for e in state['pending_events'] if e['status'] in {'DELIVERED', 'CLAIMED'}):
        role = event_role(active, state)
        old = state.get("agent_routes", {}).get(role)
        if old and any(old.get(k) != routes.get(role, {}).get(k) for k in ("agent_name", "pane_id", "terminal_id", "thread_id")):
            raise RuntimeError("ACTIVE_EVENT_ROUTE_REBIND_FORBIDDEN")
    previous_bytes = state_path.read_bytes()
    backup = state_path.parent / "state_backups" / f"{state_path.stem}.before_named_routes.{hashlib.sha256(previous_bytes).hexdigest()[:16]}.json"
    backup.parent.mkdir(parents=True, exist_ok=True)
    if not backup.exists():
        backup.write_bytes(previous_bytes)
    state["agent_routes"] = routes
    state["controller_target"] = None
    state["director_target"] = routes.get("director", {}).get("pane_id")
    save(state_path, state, "OWNER_AGENT_ROUTES_CONFIGURED_NO_ACTIVATION")


def prompt_text(item: dict[str, Any], state_path: Path) -> str:
    payload = item["payload"]
    event_id = item["event_id"]
    if item['kind'] == 'DIRECTOR_TASK' and payload.get('status_report_only') is True:
        return (
            f'ARC2 Controller stage report {event_id}, STATUS ONLY, not an approval request. '
            f'Claim once: py -3 {PROJECT_ROOT / "scripts/arc2_governor.py"} --state "{state_path.resolve()}" --claim-event {event_id}. '
            f'Read {payload["stage_report"]} and verify SHA256={payload["stage_report_sha256"]}. '
            'Summarize the verified milestone/result/blocker to the owner without inferring scientific acceptance. '
            'Controller continues independently; do not launch, rearm, review, modify scientific assets or prompt Controller. '
            f'Acknowledge without changing its job/queue: py -3 {PROJECT_ROOT / "scripts/arc2_governor.py"} '
            f'--state "{state_path.resolve()}" --ack-stage-report {event_id}. '
            'Do not use --complete-event or create a response/review loop for this informational report.'
        )
    if item['kind'] == 'DIRECTOR_TASK' or (item['kind'] == 'PRODUCER_ERROR' and item['recipient'] == 'director'):
        return (
            f'ARC2 Governor Director continuation event {event_id}. Owner assigned this work directly to Director. '
            f'Claim once: py -3 {PROJECT_ROOT / "scripts/arc2_governor.py"} --state "{state_path.resolve()}" --claim-event {event_id}. '
            f'Execute only this bounded payload and its owner-authorized references: {json.dumps(payload, sort_keys=True)}. '
            'This is a task/status handoff, not a new scientific approval request. Preserve failed evidence; '
            'an infrastructure failure does not silently terminate a pending owner task. Do not poll long jobs in chat. '
            'Before yielding to a long job, register it with completion_recipient=director and complete this same event '
            'with WAIT_REMOTE. Supervisor/Governor will wake this same Director pane once on success OR failure. '
            'When fully complete or genuinely blocked, close this same event with an honest outcome; never rerun completed generation.'
        )
    if item["recipient"] == "controller":
        return (
            f"ARC2 Governor event {event_id}. Read {PROJECT_ROOT / 'AGENTS.md'} and "
            f"{PROJECT_ROOT / 'orchestration/agents/ARC_CONTROLLER_SYSTEM.md'}. Claim with: py -3 {PROJECT_ROOT / 'scripts/arc2_governor.py'} "
            f"--state \"{state_path.resolve()}\" --claim-event {event_id}. Execute only this event payload: "
            f"{json.dumps(payload, sort_keys=True)}. Complete it through the same Governor CLI; never prompt Director directly. "
            "This payload is the bounded handoff. Do not read the full workflow/event history unless diagnosing a specific binding error. "
            "Invoke the claim CLI once; it checks actual thread plus live Agent identity. A shared executor's inherited HERDR_PANE_ID mismatch alone is not a blocker; never rewrite that environment. "
            "A terminal event authorizes ingestion of existing artifacts, not a rerun or new scientific execution. "
            "After a verified key milestone, experiment result or genuine blocker, send one immutable stage report using "
            "--stage-report PATH --stage-report-sha256 SHA on SAME-event completion. Follow ARC_CONTROLLER_SYSTEM.md schema; "
            "this is nonblocking status, not a review. Do not report every retry or duplicate an existing scientific review."
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
    candidates = [item for item in state["pending_events"] if item["status"] == "QUEUED"]
    if not candidates:
        return "IDLE"
    outcomes, skipped, visited_roles = [], [], set()
    for candidate in candidates:
        # Preserve FIFO within each agent, not a global head-of-line lock.
        state = load(state_path)
        try:
            role = event_role(candidate, state)
            if role in visited_roles:
                continue
            visited_roles.add(role)
            if route_held(role, state) or role in state['paused_roles']:
                skipped.append('RECIPIENT_OWNER_HOLD_EVENT_QUEUED')
                continue
            if globally_stopped(state) and candidate['kind'] != 'PRODUCER_ERROR':
                skipped.append('WORKFLOW_QUIESCENT')
                continue
            if occupied(state, role, candidate['event_id']):
                skipped.append('RECIPIENT_WORKING_EVENT_QUEUED')
                continue
        except RuntimeError as exc:
            record_route_issue(state_path, candidate, str(exc))
            skipped.append('RECIPIENT_ROUTE_INVALID_EVENT_QUEUED')
            continue
        try:
            target = resolve_agent(role, state)
        except (RuntimeError, subprocess.TimeoutExpired) as exc:
            detail = str(exc)
            if isinstance(exc, subprocess.TimeoutExpired) or detail.startswith('HERDR_AGENT_LIST_FAILED:') or detail.endswith('_TARGET_UNRESOLVED'):
                skipped.append('RECIPIENT_TARGET_UNAVAILABLE_EVENT_QUEUED')
            else:
                record_route_issue(state_path, candidate, detail)
                skipped.append('RECIPIENT_ROUTE_INVALID_EVENT_QUEUED')
            continue
        if target.get('agent_status') not in {'idle', 'done'}:
            skipped.append('RECIPIENT_WORKING_EVENT_QUEUED' if target.get('agent_status') == 'working'
                           else 'RECIPIENT_NOT_READY_EVENT_QUEUED')
            continue
        outcomes.append(deliver_once(candidate, role, target, state, state_path, timeout))
    if 'EVENT_DELIVERED' in outcomes:
        return 'EVENT_DELIVERED'
    return next(iter(outcomes or skipped), 'IDLE')


def deliver_once(queued: dict, role: str, target: dict, state: dict,
                 state_path: Path, timeout: int) -> str:
    """One durable delivery reservation; timeout/rejection never redelivers."""
    target_key = f"{queued['recipient']}_target"
    # DELIVERED remains the compatible lifecycle state; delivery_state explicitly
    # distinguishes durable intent from confirmed input. Re-read after observation
    # so another CLI transition can never be overwritten by this snapshot.
    with state_transaction(state_path):
        fresh = load(state_path)
        queued = find_event(fresh, queued['event_id'])
        if queued['status'] != 'QUEUED' or occupied(fresh, role, queued['event_id']):
            return 'EVENT_CHANGED_DURING_OBSERVATION'
        if globally_stopped(fresh) and queued['kind'] != 'PRODUCER_ERROR':
            return 'WORKFLOW_QUIESCENT'
        if event_role(queued, fresh) != role or route_held(role, fresh) or role in fresh['paused_roles']:
            return 'RECIPIENT_OWNER_HOLD_EVENT_QUEUED'
        if fresh.get('agent_routes') != state.get('agent_routes'):
            return 'EVENT_ROUTE_CHANGED_DURING_OBSERVATION'
        if queued.get('attempts', 0):
            return 'EVENT_SINGLE_DELIVERY_ALREADY_ATTEMPTED'
        queued.update({"status": "DELIVERED", "delivery_state": "ATTEMPTING",
                       "attempts": int(queued.get("attempts") or 0) + 1,
                       "delivered_at": now(), "target": target.get("pane_id"),
                       "route_role": role, "target_agent": target.get("name"),
                       "target_terminal_id": target.get("terminal_id")})
        if 'agent_routes' not in fresh or queued['recipient'] == 'director':
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
            # Even a confirmed pre-input rejection stays reserved. Reconcile
            # through one incident; never spend another agent turn automatically.
            save(state_path, fresh, f'EVENT_DELIVERY_CLASSIFIED:{item["event_id"]}:{classification}')
        append_ledger(state_path, {'transition': 'HERDR_PROMPT_RESULT',
                                  'event_id': queued['event_id'], 'delivery_state': classification,
                                  'error_code': error_code})
    return 'EVENT_DELIVERED' if classification == 'CONFIRMED' else 'EVENT_DELIVERY_UNKNOWN_FAIL_CLOSED'


def record_route_issue(state_path: Path, item: dict[str, Any], reason: str) -> None:
    """One local diagnostic per event/defect; never wake another guessed agent."""
    key = stable_id("ROUTE_ISSUE", item.get("recipient", "unknown"),
                    {"event_id": item["event_id"], "reason": reason})
    marker = state_path.parent / "routing_alerts" / f"{key}.json"
    if not marker.exists():
        atomic(marker, {"event_id": item["event_id"], "reason": reason,
                        "action": "REPAIR_EXPLICIT_BINDING_WITHOUT_REDELIVERY", "at": now()})


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
    if item.get("status") != "CLAIMED":
        raise RuntimeError("CLAIMED_EVENT_NOT_ACTIVE")
    role = event_role(item, state)
    if route_held(role, state):
        return "RECIPIENT_OWNER_HOLD_EVENT_QUEUED"
    # Owner selected strict one-shot delivery. A stalled claim is diagnostic
    # evidence, not permission to inject its payload into the agent again.
    return 'CLAIMED_EVENT_RESUME_DISABLED_SINGLE_DELIVERY'


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
    owner_stop = owner_stop_snapshot(state)
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
    if (state.get('review') or {}).get('brief_sha256') == payload.get('brief_sha256'):
        state["review"] = None
    controller_role = response.get("controller_role") or payload.get("controller_role")
    if payload.get('controller_role') and controller_role != payload['controller_role']:
        producer_error(state, {**item, 'producer': 'director'}, 'DIRECTOR_RESPONSE_CONTROLLER_ROLE_MISMATCH')
        save(state_path, state, f'DIRECTOR_RESPONSE_REJECTED:{item["event_id"]}')
        return True
    context = {"controller_role": controller_role} if controller_role else {}
    if decision == "CONTINUE_CONTROLLER":
        if not response.get("next_stage") or not response.get("next_action"):
            producer_error(state, {**item, "producer": "director"}, "CONTINUE_CONTROLLER_FIELDS_MISSING")
        else:
            state.update({"disposition": "CONTINUE_CONTROLLER", "stage": response["next_stage"],
                          "next_action": response["next_action"], "pause_reason": None,
                          "controller_role": controller_role})
            enqueue(state, event("DIRECTOR_DECISION", "controller", "director", {
                **context,
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
                "brief_path": successor, "brief_sha256": successor_sha, "reason": reason, **context,
            }})
            enqueue(state, event("DIRECTOR_REVIEW", "director", "director", state["review"]))
    elif decision == "REQUIRE_CHANGES":
        state.update({"disposition": "CONTINUE_CONTROLLER", "next_action": "APPLY_DIRECTOR_REQUIRED_CHANGES"})
        enqueue(state, event("DIRECTOR_DECISION", "controller", "director", {
            **context,
            "decision": decision, "response_path": str(response_path.resolve()),
            "response_sha256": sha256_file(response_path),
        }))
    elif decision == "PAUSED":
        state.update({"disposition": "PAUSED", "pause_reason": "PAUSED_BY_DIRECTOR"})
    else:
        state.update({"disposition": "TERMINAL", "terminal_reason": "TERMINAL_BY_DIRECTOR"})
    if 'agent_routes' in state:
        role = controller_role or 'director'
        if decision in {'PAUSED', 'TERMINAL'}:
            state['paused_roles'][role] = decision + '_BY_DIRECTOR'
        else:
            state['paused_roles'].pop(role, None)
        state['pause_scope'] = 'agent'
    state.update(owner_stop)
    save(state_path, state, f"DIRECTOR_RESPONSE_CONSUMED:{item['event_id']}")
    return True


@state_mutation
def claim_event(state_path: Path, event_id: str) -> None:
    state = load(state_path)
    item = find_event(state, event_id)
    caller_pane = os.environ.get("HERDR_PANE_ID")
    if "agent_routes" in state:
        role = event_role(item, state)
        if route_held(role, state):
            raise RuntimeError("RECIPIENT_OWNER_HELD")
        if item.get("status") not in {"DELIVERED", "CLAIMED"}:
            raise RuntimeError("EVENT_CLAIM_TARGET_MISMATCH")
        caller_pane = event_caller_pane(item, state)
    if item["status"] == "CLAIMED":
        if not caller_pane or item.get("target") != caller_pane:
            raise RuntimeError("EVENT_ALREADY_CLAIMED_BY_OTHER_PANE")
        append_ledger(state_path, {"transition": "EVENT_CLAIM_IDEMPOTENT", "event_id": event_id,
                                   "target": caller_pane})
        return
    if item["status"] not in {"DELIVERED", "QUEUED"}:
        raise RuntimeError("EVENT_NOT_CLAIMABLE")
    if occupied(state, item_role(item, state), event_id):
        raise RuntimeError("ANOTHER_EVENT_ALREADY_ACTIVE")
    item.update({"status": "CLAIMED", "claimed_at": now(), "delivery_state": "ACKNOWLEDGED"})
    if "agent_routes" in state:
        item["claim_thread_id"] = os.environ.get("CODEX_THREAD_ID")
        item["claim_inherited_pane"] = os.environ.get("HERDR_PANE_ID")
        item["claim_identity"] = "EXACT_THREAD_AND_LIVE_AGENT" if state["agent_routes"][role].get("thread_id") else "EXACT_PANE"
    save(state_path, state, f"EVENT_CLAIMED:{event_id}")


def controller_stage_report(state: dict, item: dict, path: str, declared_sha: str) -> dict:
    """Controller classifies a key milestone; router only validates/routes it."""
    report_path, digest = validate_bound_artifact(path, declared_sha)
    report = json.loads(report_path.read_text(encoding='utf-8-sig'))
    if item.get('recipient') != 'controller' or report.get('source_event_id') != item['event_id']:
        raise RuntimeError('STAGE_REPORT_SOURCE_EVENT_MISMATCH')
    if report.get('report_kind') not in {'MILESTONE', 'EXPERIMENT_RESULT', 'BLOCKED'}:
        raise RuntimeError('STAGE_REPORT_KIND_INVALID')
    if report.get('controller_role') != item.get('payload', {}).get('controller_role'):
        raise RuntimeError('STAGE_REPORT_ROLE_MISMATCH')
    if any(not isinstance(report.get(k), str) or not report[k].strip()
           for k in ('experiment_id', 'stage', 'status', 'summary')):
        raise RuntimeError('STAGE_REPORT_FIELDS_REQUIRED')
    if report.get('schema_version') != 1 or len(report['summary']) > 4000:
        raise RuntimeError('STAGE_REPORT_SCHEMA_INVALID')
    payload = {'status_report_only': True, 'source_event_id': item['event_id'],
               'controller_role': report['controller_role'], 'experiment_id': report['experiment_id'],
               'stage': report['stage'], 'report_kind': report['report_kind'],
               'stage_report': str(report_path.resolve()), 'stage_report_sha256': digest}
    for old in state['pending_events']:
        previous = old.get('payload', {})
        if (previous.get('status_report_only') is True and
            all(previous.get(k) == payload[k] for k in ('source_event_id', 'stage', 'report_kind'))):
            if previous.get('stage_report_sha256') != digest:
                raise RuntimeError('STAGE_REPORT_IDENTITY_CONFLICT')
            return old
    return event('DIRECTOR_TASK', 'director', 'controller', payload)


@state_mutation
def acknowledge_stage_report(state_path: Path, event_id: str) -> None:
    state = load(state_path)
    item = find_event(state, event_id)
    if item.get('kind') != 'DIRECTOR_TASK' or item.get('payload', {}).get('status_report_only') is not True:
        raise RuntimeError('NOT_A_DIRECTOR_STAGE_REPORT')
    if 'agent_routes' in state:
        event_caller_pane(item, state)
    elif os.environ.get('HERDR_PANE_ID') != item.get('target'):
        raise RuntimeError('EVENT_CLAIM_TARGET_MISMATCH')
    if item['status'] == 'COMPLETE':
        return  # Idempotent; no successor, state reset or new notification.
    if item['status'] not in {'CLAIMED', 'DELIVERED'}:
        raise RuntimeError('STAGE_REPORT_NOT_ACKNOWLEDGEABLE')
    payload = item['payload']
    validate_bound_artifact(payload['stage_report'], payload['stage_report_sha256'])
    item.update(status='COMPLETE', completed_at=now(), outcome='DIRECTOR_STATUS_ACKNOWLEDGED')
    # Deliberately preserve global disposition, stage, remote jobs and other role holds.
    save(state_path, state, f'STAGE_REPORT_ACKNOWLEDGED:{event_id}')


@state_mutation
def complete_event(state_path: Path, args: argparse.Namespace) -> None:
    state = load(state_path)
    owner_stop = owner_stop_snapshot(state)
    item = find_event(state, args.complete_event)
    if item.get('payload', {}).get('status_report_only') is True:
        raise RuntimeError('USE_ACK_STAGE_REPORT_WITHOUT_CHANGING_CONTROLLER_STATE')
    role = item_role(item, state)
    if "agent_routes" in state:
        event_caller_pane(item, state)
    if item["status"] not in {"CLAIMED", "DELIVERED"}:
        raise RuntimeError("EVENT_NOT_COMPLETABLE")
    disposition = args.next_disposition or state["disposition"]
    if disposition not in DISPOSITIONS:
        raise RuntimeError("INVALID_COMPLETION_DISPOSITION")
    stage_report = None
    report_path, report_sha = getattr(args, 'stage_report', None), getattr(args, 'stage_report_sha256', None)
    if report_path or report_sha:
        if not report_path or not report_sha:
            raise RuntimeError('STAGE_REPORT_BINDING_REQUIRED')
        if disposition == 'REVIEW_REQUIRED':
            raise RuntimeError('REPORT_AND_SCIENTIFIC_REVIEW_MUST_NOT_DUPLICATE')
        stage_report = controller_stage_report(state, item, report_path, report_sha)
    if disposition == "REVIEW_REQUIRED":
        if not args.review_brief or not args.review_brief_sha256 or not args.review_reason:
            raise RuntimeError("REVIEW_BINDING_REQUIRED")
        validate_director_brief(args.review_brief)
        validate_bound_artifact(args.review_brief, args.review_brief_sha256)
        state["review"] = {"brief_path": args.review_brief, "brief_sha256": args.review_brief_sha256,
                           "reason": args.review_reason}
        controller_role = item.get("payload", {}).get("controller_role")
        if controller_role:
            state["review"]["controller_role"] = controller_role
        enqueue(state, event('DIRECTOR_REVIEW', 'director', 'controller', dict(state['review'])))
    if disposition == "WAIT_REMOTE":
        if not args.remote_job_file:
            raise RuntimeError("REMOTE_JOB_FILE_REQUIRED")
        _, remote = _read_remote_job_manifest(args.remote_job_file)
        retire_claimed_kaggle_terminal(state, item)
        retired = retire_completed_role_jobs(state, remote)
        protect_remote_owner(state, remote, item)
        persist_remote_binding(state_path, remote)
        state['remote_jobs'][remote['job_id']] = remote
        if retired:
            item['retired_completed_remote_jobs'] = retired
        sync_remote_display(state)
    if 'agent_routes' in state:
        if disposition in {'PAUSED', 'TERMINAL'}:
            state['paused_roles'][role] = args.pause_reason or args.terminal_reason or disposition
        else:
            state['paused_roles'].pop(role, None)
        if disposition == 'CONTINUE_CONTROLLER' and args.next_action:
            payload = {'stage': args.stage or item['payload'].get('stage'), 'next_action': args.next_action}
            if item['payload'].get('controller_role'):
                payload['controller_role'] = item['payload']['controller_role']
            enqueue(state, event('CONTROLLER_TASK', 'controller', item['recipient'], payload))
    if stage_report is not None:
        enqueue(state, stage_report)
        item['director_stage_report_event_id'] = stage_report['event_id']
    item.update({"status": "COMPLETE", "completed_at": now(), "outcome": args.outcome or "COMPLETE",
                 "completion_disposition": disposition})
    state.update({"disposition": disposition,
                  "stage": args.stage or state["stage"], "next_action": args.next_action,
                  "pause_reason": args.pause_reason if disposition == "PAUSED" else None,
                  "terminal_reason": args.terminal_reason if disposition == "TERMINAL" else None})
    if 'agent_routes' in state:
        state['pause_scope'] = 'agent'
    state.update(owner_stop)
    save(state_path, state, f"EVENT_COMPLETED:{args.complete_event}")


def _read_remote_job_manifest(value: str) -> tuple[Path, dict[str, Any]]:
    remote_path = resolve_artifact(value)
    remote = json.loads(remote_path.read_text(encoding="utf-8-sig"))
    if remote.get('backend') == 'kaggle':
        sys.path.insert(0, str(PROJECT_ROOT / 'orchestration' / 'supervisor'))
        from kaggle_jobs import validate
        validate(remote)
        return remote_path, remote
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
    if 'cmdline_sha256' in primary and not re.fullmatch(r'[0-9a-f]{64}', str(primary['cmdline_sha256'])):
        raise RuntimeError('REMOTE_JOB_CMDLINE_SHA256_INVALID')
    if remote.get('completion_recipient', 'controller') not in {'controller', 'director'}:
        raise RuntimeError('INVALID_COMPLETION_RECIPIENT')
    if remote.get('controller_role') is not None and remote['controller_role'] not in {'oracle', 'selector'}:
        raise RuntimeError('INVALID_REMOTE_CONTROLLER_ROLE')
    if remote.get('gpu_resource'):
        if not re.fullmatch(r'GPU-[0-9a-fA-F-]{36}', remote['gpu_resource']):
            raise RuntimeError('INVALID_GPU_RESOURCE_UUID')
        if remote.get('gpu_lock_policy') != 'HOST_LOCAL_INHERITED_FD_V1':
            raise RuntimeError('GPU_ATOMIC_LOCK_REQUIRED')
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
    state['remote_jobs'].pop(previous['job_id'], None)
    state['remote_jobs'][remote['job_id']] = remote
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
    state['remote_jobs'][remote['job_id']] = remote
    state.update({"disposition": "WAIT_REMOTE", "remote_job": remote,
                  "next_action": args.next_action or "SUPERVISOR_MONITORS_BOUND_REMOTE_JOB",
                  "pause_reason": None, "terminal_reason": None})
    save(state_path, state, f"PAUSED_REMOTE_JOB_REGISTERED:{remote['job_id']}")


@state_mutation
def register_remote_job(state_path: Path, value: str) -> None:
    """Adopt an existing job for monitoring only; never launch or rearm it."""
    state = load(state_path)
    owner_stop = owner_stop_snapshot(state)
    _, remote = _read_remote_job_manifest(value)
    role = remote_role(remote)
    if occupied(state, role):
        if remote.get('backend') != 'kaggle' or 'agent_routes' not in state:
            raise RuntimeError('ACTIVE_EVENT_MUST_REGISTER_ON_COMPLETION')
        active = [e for e in state['pending_events'] if item_role(e, state) == role and e['status'] in {'CLAIMED', 'DELIVERED'}]
        if len(active) != 1 or active[0]['status'] != 'CLAIMED':
            raise RuntimeError('KAGGLE_PRELAUNCH_REQUIRES_CLAIMED_EVENT')
        event_caller_pane(active[0], state)
        retire_claimed_kaggle_terminal(state, active[0])
        previous = state['remote_jobs'].get(remote['job_id'])
        if previous and previous != remote:
            from kaggle_jobs import verify_prelaunch_alias
            verify_prelaunch_alias(previous, remote)
            # Preserve the original registry bytes; only a proved platform alias
            # can replace this binding during the same claimed prelaunch event.
            registry = state_path.parent / 'remote_jobs' / f'{remote["job_id"]}.json'
            archived = registry.parent / 'prelaunch_alias_history' / f'{remote["job_id"]}_{sha256_file(registry)}.json'
            atomic(archived, previous)
            atomic(registry, remote)
            state['remote_jobs'][remote['job_id']] = remote
        protect_remote_owner(state, remote, active[0])
        persist_remote_binding(state_path, remote)
        state['remote_jobs'][remote['job_id']] = remote
        sync_remote_display(state)
        state.update(owner_stop)
        save(state_path, state, f'KAGGLE_PRELAUNCH_MONITOR_REGISTERED:{remote["job_id"]}')
        return
    previous = state['remote_jobs'].get(remote['job_id'])
    protect_remote_owner(state, remote)
    if 'agent_routes' not in state and state.get('review'):
        raise RuntimeError('PENDING_REVIEW_PROTECTED')
    if remote.get('pipeline_status_path'):
        process = remote['primary_process']
        if not process.get('start_ticks') or not process.get('boot_id') or not remote.get('source_commit'):
            raise RuntimeError('ADOPTED_PROCESS_CREATION_IDENTITY_REQUIRED')
        if remote['pipeline_status_path'] != remote['output_root'].rstrip('/') + '/PIPELINE_STATE.json':
            raise RuntimeError('PIPELINE_STATUS_MUST_BE_BOUND_TO_OUTPUT_ROOT')
    persist_remote_binding(state_path, remote)
    if previous == remote:
        return
    state['remote_jobs'][remote['job_id']] = remote
    sync_remote_display(state)
    state.update(disposition='WAIT_REMOTE',
                 stage=remote.get('stage') or state['stage'],
                 next_action='SUPERVISOR_MONITORS_BOUND_REMOTE_JOB',
                 pause_reason=None, terminal_reason=None)
    state.update(owner_stop)
    save(state_path, state, f'REMOTE_JOB_REGISTERED:{remote["job_id"]}')


@state_mutation
def bind_experiment_queue(state_path: Path, value: str) -> None:
    issue, _ = experiment_queue_terminal_issue(value)
    if issue == 'EXPERIMENT_QUEUE_UNVERIFIED':
        raise RuntimeError(issue)
    state = load(state_path)
    state['experiment_queue_path'] = str(resolve_artifact(value))
    save(state_path, state, 'OWNER_EXPERIMENT_QUEUE_BOUND')


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
    value['remote_job_ids'] = list(state['remote_jobs'])
    value['active_events_by_agent'] = state['active_events']
    value['reserved_events_by_agent'] = {
        item_role(e, state): e['event_id'] for e in state['pending_events']
        if e['status'] in {'DELIVERED', 'CLAIMED'}}
    value['queued_by_agent'] = {role: sum(e['status'] == 'QUEUED' and item_role(e, state) == role
                                        for e in state['pending_events'])
                                for role in state.get('agent_routes', {})}
    value['delivery_policy'] = 'ONCE_PER_EVENT_PER_AGENT_FIFO_SKIP_BUSY'
    value['source_sha256'] = sha256_file(Path(__file__))
    value['held_routes'] = [role for role, route in state.get('agent_routes', {}).items() if route.get('held')]
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
    parser.add_argument('--ack-stage-report', help='Acknowledge informational Director report without changing Controller work')
    parser.add_argument("--resume-paused-remote-job")
    parser.add_argument('--register-remote-job', help='Monitor an existing immutable job; does not launch it')
    parser.add_argument('--bind-experiment-queue', help='Bind owner queue; TERMINAL requires all required items COMPLETE')
    parser.add_argument('--configure-agent-routes', help='Operator-reviewed exact agent/pane/hold bindings; never activates work')
    parser.add_argument("--replace-stale-wait-remote-job")
    parser.add_argument("--failure-receipt")
    parser.add_argument("--next-disposition", choices=sorted(DISPOSITIONS))
    parser.add_argument("--stage")
    parser.add_argument("--next-action")
    parser.add_argument("--outcome")
    parser.add_argument("--review-brief")
    parser.add_argument("--review-brief-sha256")
    parser.add_argument("--review-reason")
    parser.add_argument('--stage-report', help='Immutable nonblocking key-milestone/result/blocker JSON for Director')
    parser.add_argument('--stage-report-sha256')
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
                                          args.resume_claimed_event, args.complete_event, args.ack_stage_report,
                                          args.resume_paused_remote_job, args.replace_stale_wait_remote_job,
                                          args.register_remote_job, args.bind_experiment_queue, args.configure_agent_routes))
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
    if args.ack_stage_report:
        acknowledge_stage_report(args.state, args.ack_stage_report)
        return 0
    if args.resume_paused_remote_job:
        resume_paused_remote_job(args.state, args)
        return 0
    if args.register_remote_job:
        register_remote_job(args.state, args.register_remote_job)
        return 0
    if args.bind_experiment_queue:
        bind_experiment_queue(args.state, args.bind_experiment_queue)
        return 0
    if args.configure_agent_routes:
        configure_agent_routes(args.state, args.configure_agent_routes)
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
