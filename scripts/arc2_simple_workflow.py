"""Minimal durable state and synchronous Director call for ARC2 stages.

This file is intentionally not a task scheduler.  The Controller owns all
ordinary work; the only permissible idle state during an active stage is an
actual detached RunPod job.  Director review is a blocking native Herdr call.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STATUSES = {"ACTIVE", "WAITING_REMOTE_JOB", "PAUSED", "TERMINAL"}
DECISIONS = {"CONTINUE", "CONTINUE_WITH_WARNING", "REQUIRE_CHANGES", "NEW_SUBPROTOCOL_REQUIRED", "PAUSE", "STOP"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    os.replace(temp, path)


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != 1 or value.get("status") not in STATUSES:
        raise RuntimeError("INVALID_ARC2_WORKFLOW_STATE")
    return value


def matching_review(path: Path, brief_hash: str) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if value.get("reviewed_brief_sha256") != brief_hash or value.get("decision") not in DECISIONS:
        return None
    return value


def synchronous_review(brief: Path, directive_dir: Path, director: str, timeout_seconds: int) -> tuple[Path, dict[str, Any]]:
    """Call Director once and wait for its structured review in this process."""
    brief_hash = hashlib.sha256(brief.read_bytes()).hexdigest()
    started = time.time()
    prompt = (
        f"ARC2 stage-level synchronous review. Read {brief.resolve()} and write one structured review artifact "
        "with decision, root cause where applicable, minimum required resolution, frozen conditions, forbidden actions, "
        "and Controller next action. The Controller is blocking in this same workflow; do not request Supervisor handoff."
    )
    command = ["herdr", "agent", "prompt", director, prompt, "--wait", "--until", "idle", "--until", "done", "--until", "blocked", "--timeout", str(timeout_seconds * 1000)]
    result = subprocess.run(command, text=True, encoding="utf-8", errors="strict", capture_output=True, check=False, timeout=timeout_seconds + 15)
    if result.returncode:
        raise RuntimeError("DIRECTOR_SYNCHRONOUS_REVIEW_FAILED")
    deadline = time.time() + min(timeout_seconds, 15)
    while time.time() < deadline:
        for candidate in sorted(directive_dir.glob("DIRECTOR_DIRECTIVE_*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            if candidate.stat().st_mtime < started:
                continue
            review = matching_review(candidate, brief_hash)
            if review is not None:
                return candidate, review
        time.sleep(0.25)
    raise RuntimeError("DIRECTOR_REVIEW_ARTIFACT_MISSING")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--set-next-action")
    parser.add_argument("--set-status", choices=sorted(STATUSES))
    parser.add_argument("--brief", type=Path)
    parser.add_argument("--directive-dir", type=Path)
    parser.add_argument("--director-agent", default="arc-director")
    parser.add_argument("--timeout-seconds", type=int, default=120)
    args = parser.parse_args()
    state = load(args.state)
    if args.set_next_action is not None:
        state["next_action"] = args.set_next_action
        state["last_updated_at"] = now()
        atomic(args.state, state)
    if args.set_status is not None:
        state["status"] = args.set_status
        state["experiment_terminal"] = args.set_status == "TERMINAL"
        state["last_updated_at"] = now()
        atomic(args.state, state)
    if args.brief is None:
        return 0
    if not args.directive_dir or not args.brief.is_file() or args.timeout_seconds <= 0:
        raise RuntimeError("INVALID_SYNCHRONOUS_REVIEW_ARGUMENTS")
    state["director_review_in_progress"] = True
    state["last_updated_at"] = now()
    atomic(args.state, state)
    directive, review = synchronous_review(args.brief, args.directive_dir, args.director_agent, args.timeout_seconds)
    state = load(args.state)
    state["director_review_in_progress"] = False
    state["last_director_review"] = {"path": str(directive.resolve()), "sha256": hashlib.sha256(directive.read_bytes()).hexdigest(), "decision": review["decision"], "at": now()}
    if review["decision"] == "PAUSE":
        state.update({"status": "PAUSED", "next_action": None})
    elif review["decision"] == "STOP":
        state.update({"status": "TERMINAL", "next_action": None, "experiment_terminal": True})
    atomic(args.state, state)
    print(json.dumps({"directive": str(directive.resolve()), "decision": review["decision"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
