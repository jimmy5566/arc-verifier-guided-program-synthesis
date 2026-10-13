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
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DISPOSITIONS = {"CONTINUE_CONTROLLER", "REVIEW_REQUIRED", "WAIT_REMOTE", "PAUSED", "TERMINAL"}
EVENT_STATUSES = {"QUEUED", "DELIVERED", "CLAIMED", "COMPLETE"}
DECISIONS = {"CONTINUE_CONTROLLER", "CONTINUE_DIRECTOR", "REQUIRE_CHANGES", "PAUSED", "TERMINAL"}
LIVE_KEYS = {
    "schema_version", "revision", "disposition", "stage", "next_action",
    "controller_target", "pending_events", "active_event", "review",
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
    state = compact_state(state)
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
            changed = enqueue(state, item) or changed
            target = processed / f"{item['event_id']}.json"
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
            source.unlink(missing_ok=True)
    return changed


def synthesize_state_event(state: dict[str, Any]) -> bool:
    if state["disposition"] == "CONTINUE_CONTROLLER":
        payload = {"stage": state["stage"], "next_action": state["next_action"]}
        return enqueue(state, event("CONTROLLER_TASK", "controller", "director", payload))
    if state["disposition"] == "REVIEW_REQUIRED" and state.get("review"):
        review = state["review"]
        payload = {"brief_path": review.get("brief_path"), "brief_sha256": review.get("brief_sha256"),
                   "review_reason": review.get("reason")}
        return enqueue(state, event("DIRECTOR_REVIEW", "director", "controller", payload))
    return False


def validate_queued_reviews(state: dict[str, Any]) -> bool:
    """Reject an invalid brief before spending a Director turn."""
    changed = False
    for item in state["pending_events"]:
        if item["kind"] != "DIRECTOR_REVIEW" or item["status"] != "QUEUED":
            continue
        payload = item["payload"]
        try:
            validate_bound_artifact(payload["brief_path"], payload.get("brief_sha256"))
        except Exception as exc:
            item.update({"status": "COMPLETE", "completed_at": now(), "outcome": "INVALID_BOUND_BRIEF"})
            producer_error(state, item, f"{type(exc).__name__}:{exc}")
            changed = True
    return changed


def herdr_agents() -> list[dict[str, Any]]:
    result = subprocess.run(["herdr", "agent", "list"], capture_output=True, check=False,
                            text=True, encoding="utf-8", errors="replace",
                            creationflags=windows_creationflags())
    if result.returncode:
        raise RuntimeError(f"HERDR_AGENT_LIST_FAILED:{result.returncode}")
    return json.loads(result.stdout).get("result", {}).get("agents", [])


def resolve_agent(role: str, state: dict[str, Any]) -> dict[str, Any]:
    name = "arc-controller" if role == "controller" else "arc-director"
    agents = herdr_agents()
    if role == "controller" and state.get("controller_target"):
        for record in agents:
            if record.get("pane_id") == state["controller_target"] and record.get("agent") == "codex":
                return record
    for record in agents:
        if record.get("name") == name and record.get("agent") == "codex":
            if role == "controller":
                state["controller_target"] = record.get("pane_id")
            return record
    raise RuntimeError(f"{role.upper()}_TARGET_UNRESOLVED")


def prompt_text(item: dict[str, Any], state_path: Path) -> str:
    payload = item["payload"]
    event_id = item["event_id"]
    if item["recipient"] == "controller":
        return (
            f"ARC2 Governor event {event_id}. Read {state_path.resolve()}, {PROJECT_ROOT / 'AGENTS.md'}, and "
            f"{PROJECT_ROOT / 'orchestration/agents/ARC_CONTROLLER_SYSTEM.md'}. Claim with: py -3 {PROJECT_ROOT / 'scripts/arc2_governor.py'} "
            f"--state \"{state_path.resolve()}\" --claim-event {event_id}. Execute only this event payload: "
            f"{json.dumps(payload, sort_keys=True)}. Complete it through the same Governor CLI; never prompt Director directly."
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


def dispatch(state: dict[str, Any], state_path: Path, timeout: int) -> str:
    queued = next((item for item in state["pending_events"] if item["status"] == "QUEUED"), None)
    if not queued:
        return "IDLE"
    target = resolve_agent(queued["recipient"], state)
    if target.get("agent_status") == "working":
        return "RECIPIENT_WORKING_EVENT_QUEUED"
    queued.update({"status": "DELIVERED", "attempts": int(queued.get("attempts") or 0) + 1,
                   "delivered_at": now(), "target": target.get("pane_id")})
    save(state_path, state, f"EVENT_DELIVERY_RESERVED:{queued['event_id']}")
    command = ["herdr", "agent", "prompt", target["pane_id"], prompt_text(queued, state_path),
               "--wait", "--until", "working", "--timeout", str(timeout * 1000)]
    completed = subprocess.run(command, capture_output=True, check=False, text=True,
                               encoding="utf-8", errors="replace", timeout=timeout + 10,
                               creationflags=windows_creationflags())
    append_ledger(state_path, {"transition": "HERDR_PROMPT_RESULT", "event_id": queued["event_id"],
                               "exit_code": completed.returncode})
    return "EVENT_DELIVERED" if completed.returncode == 0 else "EVENT_DELIVERY_UNKNOWN_FAIL_CLOSED"


def response_directory() -> Path:
    return PROJECT_ROOT / "orchestration" / "director" / "responses"


def consume_director_response(state: dict[str, Any], state_path: Path) -> bool:
    item = next((entry for entry in state["pending_events"]
                 if entry["kind"] == "DIRECTOR_REVIEW" and entry["status"] in {"DELIVERED", "CLAIMED"}), None)
    if not item:
        return False
    payload = item["payload"]
    try:
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


def claim_event(state_path: Path, event_id: str) -> None:
    state = load(state_path)
    item = find_event(state, event_id)
    if item["status"] not in {"DELIVERED", "QUEUED"}:
        raise RuntimeError("EVENT_NOT_CLAIMABLE")
    if state.get("active_event") not in {None, event_id}:
        raise RuntimeError("ANOTHER_EVENT_ALREADY_ACTIVE")
    item.update({"status": "CLAIMED", "claimed_at": now()})
    state["active_event"] = event_id
    save(state_path, state, f"EVENT_CLAIMED:{event_id}")


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
        validate_bound_artifact(args.review_brief, args.review_brief_sha256)
        state["review"] = {"brief_path": args.review_brief, "brief_sha256": args.review_brief_sha256,
                           "reason": args.review_reason}
    if disposition == "WAIT_REMOTE":
        if not args.remote_job_file:
            raise RuntimeError("REMOTE_JOB_FILE_REQUIRED")
        remote_path = resolve_artifact(args.remote_job_file)
        state["remote_job"] = json.loads(remote_path.read_text(encoding="utf-8-sig"))
    else:
        state["remote_job"] = None
    item.update({"status": "COMPLETE", "completed_at": now(), "outcome": args.outcome or "COMPLETE"})
    state.update({"active_event": None, "disposition": disposition,
                  "stage": args.stage or state["stage"], "next_action": args.next_action,
                  "pause_reason": args.pause_reason if disposition == "PAUSED" else None,
                  "terminal_reason": args.terminal_reason if disposition == "TERMINAL" else None})
    save(state_path, state, f"EVENT_COMPLETED:{args.complete_event}")


def write_service_state(path: Path | None, status: str, state: dict[str, Any], detail: str | None = None) -> None:
    if not path:
        return
    value = {"service": "ARC2_GOVERNOR", "pid": os.getpid(), "status": status,
             "heartbeat_at": now(), "workflow_disposition": state["disposition"],
             "workflow_stage": state["stage"], "queued_events": sum(
                 1 for item in state["pending_events"] if item["status"] == "QUEUED")}
    if detail:
        value["detail"] = detail
    atomic(path, value)


def cycle(state_path: Path, *, agent_timeout_seconds: int, **_: Any) -> tuple[str, dict[str, Any]]:
    state = load(state_path)
    changed = import_external_events(state, state_path)
    changed = synthesize_state_event(state) or changed
    changed = validate_queued_reviews(state) or changed
    if changed:
        save(state_path, state, "EVENTS_RECONCILED")
        state = load(state_path)
    if consume_director_response(state, state_path):
        state = load(state_path)
    result = dispatch(state, state_path, agent_timeout_seconds)
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
    parser.add_argument("--claim-event")
    parser.add_argument("--complete-event")
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
    modes = sum(bool(value) for value in (args.migrate_legacy, args.claim_event, args.complete_event))
    if modes > 1:
        parser.error("migration, claim, and completion are mutually exclusive")
    if args.migrate_legacy:
        migrate_legacy(args.state)
        return 0
    if args.claim_event:
        claim_event(args.state, args.claim_event)
        return 0
    if args.complete_event:
        complete_event(args.state, args)
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
