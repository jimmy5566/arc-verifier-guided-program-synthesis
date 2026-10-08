"""Controller-owned direct Director review, with durable fallback metadata.

This is deliberately a short synchronous handoff.  The Supervisor only uses
the recorded fallback timeout if this process or the Controller disappears.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def atomic(path: Path, value: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    temp.replace(path)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def compatible_directive(path: Path, started: float, brief_hash: str) -> bool:
    if path.stat().st_mtime < started:
        return False
    try:
        data = load(path)
    except (OSError, json.JSONDecodeError):
        return False
    # A new directive alone is insufficient: an unrelated Supervisor-delivered
    # directive must never satisfy this Controller-owned synchronous review.
    return (
        isinstance(data.get("directive_id"), str)
        and isinstance(data.get("decision"), str)
        and data.get("reviewed_brief_sha256") == brief_hash
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--brief", type=Path, required=True)
    parser.add_argument("--cycle-state", type=Path, required=True)
    parser.add_argument("--parent-directive-id", required=True)
    parser.add_argument("--directive-dir", type=Path, required=True)
    parser.add_argument("--director-agent", default="arc-director")
    parser.add_argument("--timeout-seconds", type=int, default=90)
    args = parser.parse_args()
    if args.timeout_seconds <= 0 or not args.brief.is_file():
        raise RuntimeError("INVALID_SYNCHRONOUS_REVIEW_INPUT")
    state = load(args.cycle_state); record = state["directives"][args.parent_directive_id]
    brief_hash = hashlib.sha256(args.brief.read_bytes()).hexdigest()
    review = record.setdefault("synchronous_review", {})
    if review.get("status") == "DIRECTIVE_RECEIVED":
        print(json.dumps({"status": "DIRECTIVE_ALREADY_RECEIVED", "directive": review.get("result_directive")}, sort_keys=True)); return 0
    started = time.time()
    review.update({"status": "DIRECT_PROMPT_SENT", "brief": str(args.brief.resolve()), "brief_sha256": brief_hash, "started_at": now(), "fallback_after_seconds": 120})
    atomic(args.cycle_state, state)
    prompt = (f"ARC2 synchronous Director review. Read {args.brief.resolve()} and write one structured directive. "
              "The Controller is waiting in this same cycle; do not request Supervisor handoff. No scientific execution is authorized by this prompt.")
    # Herdr's --wait is the authoritative completion primitive.  Do not submit
    # a prompt and then return control to the Supervisor/file watcher: this
    # process owns the short Controller -> Director -> Controller handoff.
    sent = subprocess.run(
        ["herdr", "agent", "prompt", args.director_agent, prompt,
         "--wait", "--until", "idle", "--until", "done", "--until", "blocked",
         "--timeout", str(args.timeout_seconds * 1000)],
        capture_output=True, text=True, check=False, timeout=args.timeout_seconds + 15,
    )
    if sent.returncode:
        review.update({"status": "DIRECT_PROMPT_FAILED", "error": (sent.stderr or sent.stdout)[-1000:], "failed_at": now()}); atomic(args.cycle_state, state); return 1
    # The blocking Herdr call has returned.  Allow a short filesystem commit
    # window for the directive, but remain in this same process and require it
    # to identify the exact brief just reviewed.
    deadline = time.time() + min(10, args.timeout_seconds)
    while time.time() < deadline:
        candidates = sorted(args.directive_dir.glob("DIRECTOR_DIRECTIVE_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        for candidate in candidates:
            if compatible_directive(candidate, started, brief_hash):
                review.update({"status": "DIRECTIVE_RECEIVED", "result_directive": str(candidate.resolve()), "result_directive_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(), "completed_at": now()})
                record.update({"state": "PROCESSING", "lifecycle_state": "PROCESSING", "next_step": "PROCESS_DIRECTOR_DIRECTIVE", "controller_cycle_complete": False, "waiting_reason": None})
                atomic(args.cycle_state, state)
                print(json.dumps({"status": "DIRECTIVE_RECEIVED", "directive": str(candidate.resolve())}, sort_keys=True)); return 0
        time.sleep(1)
    review.update({"status": "FALLBACK_REQUIRED", "timeout_at": now()}); atomic(args.cycle_state, state)
    print(json.dumps({"status": "FALLBACK_REQUIRED"}, sort_keys=True)); return 2


if __name__ == "__main__":
    raise SystemExit(main())
