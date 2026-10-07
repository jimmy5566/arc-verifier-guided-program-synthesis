#!/usr/bin/env python3
"""Deterministic, restart-safe ARC2 receipt reconciler (no LLM calls)."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import base64
import subprocess
import tempfile
from pathlib import Path
from typing import Any

TERMINAL_STATES = {"SUCCESS", "TRAIN_FAILED", "OOM", "INFRA_FAILED", "INTERRUPTED"}
DIRECTOR_DECISIONS = {
    "CONTINUE", "CONTINUE_WITH_WARNING", "REQUIRE_CURRICULUM_REVIEW",
    "REQUIRE_DIAGNOSTIC_REVIEW", "REQUIRE_INFRA_REPAIR", "PAUSE_SCIENTIFIC_EXPERIMENT",
    "STOP_CURRENT_TRACK", "NEW_SUBPROTOCOL_REQUIRED", "EMERGENCY_RECOVERY",
}


def canonical_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def default_state() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "rounds": {},
        "accepted_rounds_since_director_review": 0,
        "new_training_tokens_since_director_review": 0,
        "director_review_required": False,
        "director_notification_sent": False,
        "director_directive_id": None,
    }


class Supervisor:
    def __init__(self, state_path: Path, notification_dir: Path) -> None:
        self.state_path = state_path
        self.notification_dir = notification_dir
        self.state = self._load()

    def _load(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return default_state()
        loaded = json.loads(self.state_path.read_text(encoding="utf-8"))
        baseline = default_state()
        baseline.update(loaded)
        return baseline

    def save(self) -> None:
        atomic_json(self.state_path, self.state)

    def reconcile(self, receipt_root: Path) -> list[str]:
        """Reconcile receipt metadata and emit each Controller notification once."""
        notifications: list[str] = []
        for path in sorted(receipt_root.glob("ROUND_*/ROUND_*_TERMINAL_RECEIPT.json")):
            receipt = json.loads(path.read_text(encoding="utf-8"))
            status = receipt.get("status")
            round_id = str(receipt.get("round_id", ""))
            if status not in TERMINAL_STATES or not round_id:
                continue
            digest = canonical_hash(path)
            record = self.state["rounds"].setdefault(round_id, {})
            if record.get("terminal_receipt_hash") not in (None, digest):
                record["receipt_identity_conflict"] = True
                self.state["director_review_required"] = True
                continue
            record["terminal_receipt_hash"] = digest
            record["terminal_status"] = status
            if not record.get("controller_notification_sent"):
                notification = self.notification_dir / f"CONTROLLER_NOTIFICATION_{round_id}_{digest[:12]}.json"
                atomic_json(notification, {
                    "round_id": round_id,
                    "terminal_receipt": str(path),
                    "terminal_receipt_hash": digest,
                    "terminal_status": status,
                    "scientific_acceptance": "NOT_INFERRED_FROM_PROCESS_RECEIPT",
                })
                record["controller_notification_sent"] = True
                notifications.append(round_id)
        self.save()
        return notifications

    def record_controller_cycle(self, round_id: str, complete: bool) -> None:
        record = self.state["rounds"].setdefault(round_id, {})
        record["controller_cycle_complete"] = bool(complete)
        self.save()

    def request_director_review(self, reason: str) -> None:
        self.state["director_review_required"] = True
        self.state["director_review_reason"] = reason
        self.save()


def remote_shell(target: str, script: str, identity_file: str | None = None) -> str:
    """Run a tiny metadata-only script through RunPod's forced-PTY SSH gateway.

    The endpoint presents an interactive shell even for a command request, so
    the script is base64-encoded and sent on stdin. No secrets are logged.
    """
    command = ["ssh", "-F", "NUL", "-tt", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20"]
    if identity_file:
        command.extend(["-i", identity_file])
    command.append(target)
    encoded = base64.b64encode(script.encode("utf-8")).decode("ascii")
    # RunPod exposes an interactive bash shell with bracketed paste enabled.
    # Paste one command, then submit it with CR; LF-only piped input is merely
    # echoed by that gateway.
    line = f"echo {encoded} | base64 -d | bash; printf '__ARC2_REMOTE_END__\\n'; exit"
    payload = f"\x1b[200~{line}\x1b[201~\r"
    completed = subprocess.run(command, input=payload, text=True, capture_output=True, check=False, timeout=45)
    if completed.returncode != 0 or "__ARC2_REMOTE_END__" not in completed.stdout:
        raise RuntimeError(f"remote metadata scan failed (exit={completed.returncode})")
    return completed.stdout.split("__ARC2_REMOTE_END__", 1)[0]


def fetch_remote_receipts(target: str, remote_root: str, cache_root: Path, identity_file: str | None = None) -> Path:
    """Copy only JSON terminal-receipt metadata into a local ephemeral cache."""
    script = f'''python3 - <<'PY'
import base64, glob, json, os
for path in sorted(glob.glob({remote_root!r} + "/rounds/ROUND_*/ROUND_*_TERMINAL_RECEIPT.json")):
    with open(path, "rb") as handle:
        print(json.dumps({{"path": path, "content_b64": base64.b64encode(handle.read()).decode("ascii")}}, sort_keys=True))
PY'''
    output = remote_shell(target, script, identity_file)
    cache_root.mkdir(parents=True, exist_ok=True)
    for line in output.splitlines():
        if not line.startswith("{"):
            continue
        try:
            item = json.loads(line)
            relative = Path(item["path"]).name
            round_dir = cache_root / relative.split("_TERMINAL_RECEIPT", 1)[0].replace("ROUND_", "ROUND_")
            round_dir.mkdir(parents=True, exist_ok=True)
            (round_dir / relative).write_bytes(base64.b64decode(item["content_b64"]))
        except (KeyError, ValueError, TypeError):
            continue
    return cache_root


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt-root", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--notification-dir", type=Path, required=True)
    parser.add_argument("--ssh-target")
    parser.add_argument("--remote-root")
    parser.add_argument("--identity-file")
    args = parser.parse_args()
    supervisor = Supervisor(args.state, args.notification_dir)
    receipt_root = args.receipt_root
    if args.ssh_target or args.remote_root:
        if not (args.ssh_target and args.remote_root):
            parser.error("--ssh-target and --remote-root must be supplied together")
        receipt_root = fetch_remote_receipts(args.ssh_target, args.remote_root, args.receipt_root, args.identity_file)
    notified = supervisor.reconcile(receipt_root)
    print(json.dumps({"status": "OK", "controller_notifications": notified}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
