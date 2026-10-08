#!/usr/bin/env python3
"""Deterministic, restart-safe ARC2 receipt reconciler (no LLM calls)."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import base64
import secrets
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TERMINAL_STATES = {"SUCCESS", "TRAIN_FAILED", "OOM", "INFRA_FAILED", "INTERRUPTED"}
ROUND_CLASS_SCIENTIFIC = "SCIENTIFIC"
ROUND_CLASS_DUMMY = "DUMMY"
ROUND_CLASS_SMOKE = "SMOKE"
ROUND_CLASS_INFRASTRUCTURE = "INFRASTRUCTURE"
SCIENTIFIC_PROTOCOLS = {"FOUNDATION_V2_RECONSTRUCTION_AND_TARGETED_REPAIR_V1", "TARGETED_CAPABILITY_REPAIR_V1", "TARGETED_CAPABILITY_REPAIR_V1R"}
DIRECTOR_DECISIONS = {
    "CONTINUE", "CONTINUE_WITH_WARNING", "REQUIRE_CURRICULUM_REVIEW",
    "REQUIRE_DIAGNOSTIC_REVIEW", "REQUIRE_INFRA_REPAIR", "PAUSE_SCIENTIFIC_EXPERIMENT",
    "STOP_CURRENT_TRACK", "NEW_SUBPROTOCOL_REQUIRED", "EMERGENCY_RECOVERY",
}
CONTROLLER_CYCLE_TERMINAL = {"CLOSED", "AUTHORIZED", "PAUSED", "STOPPED", "HARD_BLOCKED"}
CONTROLLER_CYCLE_WAITING = {"WAITING_REMOTE_JOB", "WAITING_DIRECTOR"}
CONTROLLER_CYCLE_ACTIVE = {"RECEIVED", "ACKNOWLEDGED", "PROCESSING", "REMEDIATION", "VALIDATING", "COMMITTING", "RESUBMITTING", "REMEDIATION_COMPLETE", "VALIDATED", "COMMITTED", "RESUBMITTED"}
CONTROLLER_ESCALATION_STATES = {"BLOCKED_INFRA", "BLOCKED_SCIENCE", "HARD_BLOCKED", "NEEDS_HIGH_LEVEL_DECISION"}


def canonical_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def decode_structured_utf8(value: bytes, source: str) -> str:
    """Decode protocol output strictly; malformed bytes are a visible failure."""
    try:
        return value.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise RuntimeError(f"{source}_UTF8_DECODE_ERROR:{error.start}:{error.end}") from error


def decode_human_utf8(value: bytes) -> str:
    """Human diagnostics may be lossy, but never crash a reader thread."""
    return value.decode("utf-8", errors="replace")


def prompt_controller(command: list[str], timeout: int) -> tuple[int, str]:
    """Submit a Herdr prompt and fail closed on non-UTF-8 or malformed JSON."""
    completed = subprocess.run(command, capture_output=True, check=False, timeout=timeout)
    if completed.returncode != 0:
        return completed.returncode, decode_human_utf8(completed.stderr or completed.stdout)[-1000:]
    try:
        payload = json.loads(decode_structured_utf8(completed.stdout, "HERDR_STRUCTURED_OUTPUT"))
    except (RuntimeError, json.JSONDecodeError) as error:
        return 1, f"HERDR_STRUCTURED_JSON_PARSE_ERROR:{error}"[-1000:]
    if not isinstance(payload, dict):
        return 1, "HERDR_STRUCTURED_JSON_NOT_OBJECT"
    return 0, ""


def controller_idle(agent: str, timeout: int) -> bool:
    """Read Herdr's status only; the Supervisor never interprets science."""
    completed = subprocess.run(["herdr", "agent", "get", agent], capture_output=True, check=False, timeout=timeout)
    if completed.returncode != 0:
        return False
    try:
        payload = json.loads(decode_structured_utf8(completed.stdout, "HERDR_AGENT_STATUS"))
        return payload["result"]["agent"]["agent_status"] == "idle"
    except (KeyError, TypeError, RuntimeError, json.JSONDecodeError):
        return False


def utc_seconds() -> float:
    return time.time()


def scientific_terminal_receipt_issues(receipt: dict[str, Any], round_class: str) -> list[str]:
    """A SUCCESS receipt is scientific evidence only when its full contract exists."""
    if round_class != ROUND_CLASS_SCIENTIFIC or receipt.get("status") != "SUCCESS":
        return []
    required = (
        "worker_exit_code", "optimizer_steps", "processed_tokens",
        "scientific_gpu_training_seconds", "ledger_snapshot_sha256",
        "training_result_sha256", "checkpoint_identities", "final_audit_accessed",
    )
    return [key for key in required if key not in receipt]


def round_class_from_contract(round_id: str, receipt: dict[str, Any]) -> str:
    """Classify the round contract independently of optimizer progress."""
    if (
        receipt.get("round_class") == ROUND_CLASS_SCIENTIFIC
        or receipt.get("protocol_id") in SCIENTIFIC_PROTOCOLS
        or round_id.startswith(("RECONSTRUCTED_FOUNDATION_V2_", "TARGETED_CAPABILITY_REPAIR_"))
    ):
        return ROUND_CLASS_SCIENTIFIC
    declared = receipt.get("round_class")
    if declared in {ROUND_CLASS_DUMMY, ROUND_CLASS_SMOKE, ROUND_CLASS_INFRASTRUCTURE}:
        return declared
    return ROUND_CLASS_INFRASTRUCTURE


def default_state() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "rounds": {},
        "directives": {},
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
            # A scientific receipt can be emitted before the worker reaches an
            # optimizer step.  Round/protocol identity therefore controls routing;
            # ``scientific_training_started`` is evidence, never the classifier.
            round_class = round_class_from_contract(round_id, receipt)
            record["round_class"] = round_class
            record["scientific_round"] = round_class == ROUND_CLASS_SCIENTIFIC
            # This captures actual optimizer progress, not the round's contract.
            record["scientific_training_started"] = receipt.get("scientific_training_started") is True
            record["protocol_id"] = receipt.get("protocol_id")
            receipt_issues = scientific_terminal_receipt_issues(receipt, round_class)
            record["terminal_evidence_valid"] = not receipt_issues
            if receipt_issues:
                record["terminal_evidence_issues"] = receipt_issues
            record["remote_receipt_path"] = str(path.resolve())
            if not record.get("controller_notification_sent"):
                notification = self.notification_dir / f"CONTROLLER_NOTIFICATION_{round_id}_{digest[:12]}.json"
                atomic_json(notification, {
                    "round_id": round_id,
                    "terminal_receipt": str(path),
                    "terminal_receipt_hash": digest,
                    "terminal_status": status,
                    "round_class": round_class,
                    "protocol_id": receipt.get("protocol_id"),
                    "scientific_training_started": receipt.get("scientific_training_started") is True,
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

    def acknowledge_controller_cycles(self, acknowledgement_dir: Path) -> list[str]:
        """Record matching Controller acknowledgements without interpreting science."""
        acknowledged: list[str] = []
        changed = False
        for round_id, record in self.state["rounds"].items():
            acknowledgement = record.get("controller_ack_path")
            if not acknowledgement or (record.get("controller_cycle_complete") and record.get("round_class") != ROUND_CLASS_SCIENTIFIC):
                continue
            path = Path(acknowledgement)
            if not path.is_absolute():
                path = acknowledgement_dir / path.name
            if not path.exists():
                continue
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if (
                str(value.get("round_id", "")) == round_id
                and value.get("terminal_receipt_hash") == record.get("terminal_receipt_hash")
                and value.get("controller_role") == "arc-controller"
                and value.get("acknowledged") is True
                and value.get("training_started") is False
            ):
                record["controller_acknowledged"] = True
                changed = True
                # Receipt acknowledgement cannot terminate a scientific Controller
                # cycle.  In particular, a legacy dummy acknowledgement is a
                # conflict signal, never completion authority.
                if record.get("round_class") == ROUND_CLASS_SCIENTIFIC:
                    record["controller_cycle_complete"] = False
                    if value.get("action") == "DUMMY_NOTIFICATION_ACK_ONLY":
                        record["scientific_state_conflict_pending"] = True
                    continue
                record["controller_cycle_complete"] = True
                acknowledged.append(round_id)
        if changed:
            self.save()
        return acknowledged

    def wake_controller_once(
        self,
        round_ids: list[str],
        controller_agent: str,
        acknowledgement_dir: Path,
        herdr_timeout_seconds: int,
    ) -> list[str]:
        """Reserve and submit one idempotent infrastructure-only Controller wake."""
        woken: list[str] = []
        acknowledgement_dir.mkdir(parents=True, exist_ok=True)
        for round_id in round_ids:
            record = self.state["rounds"][round_id]
            if record.get("controller_wakeup_reserved"):
                continue
            digest = record["terminal_receipt_hash"]
            notification = self.notification_dir / f"CONTROLLER_NOTIFICATION_{round_id}_{digest[:12]}.json"
            acknowledgement = acknowledgement_dir / f"CONTROLLER_ACK_{round_id}_{digest[:12]}.json"

            # Persist the reservation before external delivery. An ambiguous CLI
            # failure therefore fails closed instead of risking a duplicate wake.
            record["controller_wakeup_reserved"] = True
            record["controller_wakeup_status"] = "RESERVED"
            record["controller_wakeup_count"] = 0
            record["controller_ack_path"] = str(acknowledgement.resolve())
            self.save()

            if record.get("round_class") == ROUND_CLASS_SCIENTIFIC:
                prompt = (
                    f"ARC2 scientific terminal receipt {digest}. Read {notification.resolve()} and its terminal receipt. "
                    f"Write one acknowledgement JSON to {acknowledgement.resolve()} with the receipt identity, then resume the durable scientific execution state. "
                    "Apply only the preregistered reconstruction equivalence gate; preserve all evidence, do not open FINAL_AUDIT, and do not retry or launch unrelated training. "
                    "If equivalence is confirmed, continue only under the frozen targeted-repair protocol and remaining budget; otherwise create a Director escalation request."
                )
            else:
                prompt = (
                    f"ARC2 infrastructure-only notification {digest}. "
                    f"Read {notification.resolve()}. Do not start training and do not invoke Director. "
                    f"Write one acknowledgement JSON to {acknowledgement.resolve()} with exactly these fields: "
                    f'{{"schema_version":1,"round_id":"{round_id}",'
                    f'"terminal_receipt_hash":"{digest}","controller_role":"arc-controller",'
                    f'"acknowledged":true,"action":"DUMMY_NOTIFICATION_ACK_ONLY",'
                    f'"training_started":false}}. '
                    "If that matching acknowledgement already exists, do not rewrite it."
                )
            exit_code, error = prompt_controller(
                ["herdr", "agent", "prompt", controller_agent, prompt], herdr_timeout_seconds,
            )
            record["controller_wakeup_cli_exit_code"] = exit_code
            if exit_code == 0:
                record["controller_wakeup_status"] = "SENT"
                record["controller_wakeup_count"] = 1
                woken.append(round_id)
            else:
                record["controller_wakeup_status"] = "DELIVERY_UNKNOWN_FAIL_CLOSED"
                record["controller_wakeup_error"] = error
            self.save()
        return woken

    def _round_contract_evidence(self, round_id: str, record: dict[str, Any], receipt: dict[str, Any]) -> tuple[str | None, str | None]:
        """Return a declared contract path/hash without treating paths as identity."""
        raw = receipt.get("round_contract_path") or record.get("round_contract_path")
        candidates: list[Path] = [Path(raw)] if isinstance(raw, str) else []
        repo_root = Path(__file__).resolve().parents[2]
        candidates.append(repo_root / "experiments" / "foundation_v2_reconstruction_and_targeted_repair_v1" / f"{round_id}_LAUNCH.json")
        for candidate in candidates:
            if candidate.is_file():
                return str(candidate.resolve()), canonical_hash(candidate)
        return None, None

    def detect_scientific_state_conflicts(
        self, receipt_root: Path, acknowledgement_dir: Path, escalation_dir: Path,
        director_agent: str, herdr_timeout_seconds: int,
    ) -> list[str]:
        """Fail closed and wake Director once for scientific/dummy contradictions.

        Receipt contract identity is authoritative; ``scientific_training_started``
        records optimizer progress only.  Existing receipts and acknowledgements are
        never rewritten.
        """
        conflicts: list[str] = []
        escalation_dir.mkdir(parents=True, exist_ok=True)
        existing = sorted(escalation_dir.glob("CONTROLLER_ESCALATION_REQUEST_*.json"))
        numbers = []
        for prior in existing:
            try:
                numbers.append(int(prior.stem.rsplit("_", 1)[1]))
            except ValueError:
                continue
        next_number = max(numbers, default=0) + 1
        for path in sorted(receipt_root.glob("ROUND_*/ROUND_*_TERMINAL_RECEIPT.json")):
            try:
                receipt = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            round_id = str(receipt.get("round_id", ""))
            if not round_id:
                continue
            record = self.state["rounds"].get(round_id)
            if not record:
                continue
            expected = round_class_from_contract(round_id, receipt)
            if expected != ROUND_CLASS_SCIENTIFIC:
                continue
            # A restart must preserve the fail-closed wait state without sending
            # another Director prompt for the same immutable conflict.
            if record.get("scientific_state_conflict_reserved"):
                if record.get("controller_cycle_complete") or record.get("controller_lifecycle_state") != "SCIENTIFIC_STATE_CONFLICT_WAITING_DIRECTOR":
                    record["controller_cycle_complete"] = False
                    record["controller_lifecycle_state"] = "SCIENTIFIC_STATE_CONFLICT_WAITING_DIRECTOR"
                    self.save()
                continue
            digest = canonical_hash(path)
            notification_path = self.notification_dir / f"CONTROLLER_NOTIFICATION_{round_id}_{digest[:12]}.json"
            notification: dict[str, Any] = {}
            if notification_path.is_file():
                try:
                    notification = json.loads(notification_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    notification = {"parse_failure": True}
            observed = notification.get("round_class", "UNKNOWN")
            acknowledgement = Path(record.get("controller_ack_path", acknowledgement_dir / f"CONTROLLER_ACK_{round_id}_{digest[:12]}.json"))
            acknowledgement_value: dict[str, Any] = {}
            if acknowledgement.is_file():
                try:
                    acknowledgement_value = json.loads(acknowledgement.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    acknowledgement_value = {"parse_failure": True}
            if acknowledgement_value.get("action") == "DUMMY_NOTIFICATION_ACK_ONLY":
                observed = "ACK_ONLY"
            invalid_success = bool(scientific_terminal_receipt_issues(receipt, expected))
            if invalid_success:
                observed = "INVALID_SCIENTIFIC_TERMINAL_RECEIPT"
            conflict = observed in {ROUND_CLASS_DUMMY, ROUND_CLASS_SMOKE, ROUND_CLASS_INFRASTRUCTURE, "ACK_ONLY", "INVALID_SCIENTIFIC_TERMINAL_RECEIPT"}
            # A newly implemented notification can be SCIENTIFIC or UNKNOWN; UNKNOWN
            # is not itself a contradiction so old receipts remain auditable.
            if not conflict:
                continue
            contract_path, contract_hash = self._round_contract_evidence(round_id, record, receipt)
            request = escalation_dir / f"CONTROLLER_ESCALATION_REQUEST_{next_number:03d}.json"
            next_number += 1
            controller_state = "CLOSED" if record.get("controller_cycle_complete") else "UNKNOWN"
            payload = {
                "schema_version": 1,
                "status": "ESCALATION_REQUESTED",
                "escalation_type": "SCIENTIFIC_STATE_CONFLICT",
                "round_id": round_id,
                "protocol_id": receipt.get("protocol_id") or record.get("protocol_id"),
                "expected_round_class": expected,
                "observed_round_class": observed,
                "round_config_path": contract_path,
                "round_config_sha256": contract_hash,
                "remote_receipt_path": str(path.resolve()),
                "remote_receipt_sha256": digest,
                "supervisor_notification_path": str(notification_path.resolve()) if notification_path.exists() else None,
                "supervisor_notification_sha256": canonical_hash(notification_path) if notification_path.is_file() else None,
                "current_controller_state": controller_state,
                "expected_controller_state": "WAITING_REMOTE_JOB_OR_SCIENTIFIC_CONTINUATION",
                "controller_acknowledgement_path": str(acknowledgement.resolve()) if acknowledgement.exists() else None,
                "controller_acknowledgement_sha256": canonical_hash(acknowledgement) if acknowledgement.is_file() else None,
                "controller_acknowledgement_action": acknowledgement_value.get("action"),
                "optimizer_steps": receipt.get("optimizer_steps", 0),
                "scientific_training_started": receipt.get("scientific_training_started") is True,
                "gpu_seconds_charged": receipt.get("scientific_gpu_training_seconds", 0.0),
                "exact_inconsistency": "SCIENTIFIC_SUCCESS_RECEIPT_MISSING_REQUIRED_EVIDENCE" if invalid_success else "SCIENTIFIC_ROUND_RECEIVED_DUMMY_OR_ACK_ONLY_CONTROLLER_PATH",
                "terminal_receipt_validation_issues": scientific_terminal_receipt_issues(receipt, expected),
                "requested_decision": "REQUIRE_INFRA_REPAIR_OR_SAFE_RETRY_DECISION",
                "preserve_existing_round_identity": True,
                "do_not_launch_gpu_job": True,
            }
            atomic_json(request, payload)
            record.update({
                "scientific_state_conflict_reserved": True,
                "scientific_state_conflict_request_path": str(request.resolve()),
                "scientific_state_conflict_round_class": {"expected": expected, "observed": observed},
                "controller_cycle_complete": False,
                "controller_lifecycle_state": "SCIENTIFIC_STATE_CONFLICT_WAITING_DIRECTOR",
                "director_escalation_reserved": True,
                "director_escalation_request_path": str(request.resolve()),
                "director_escalation_reason": ["SCIENTIFIC_STATE_CONFLICT"],
                "director_escalation_wakeup_count": 0,
            })
            self.save()
            prompt = (
                f"ARC2 SCIENTIFIC_STATE_CONFLICT for round {round_id}. Read {request.resolve()} and issue a structured Director directive. "
                "Do not authorize a retry or launch until the conflict is resolved."
            )
            code, error = prompt_controller(["herdr", "agent", "prompt", director_agent, prompt], herdr_timeout_seconds)
            record["director_escalation_cli_exit_code"] = code
            if code == 0:
                record["director_escalation_status"] = "SENT"
                record["director_escalation_wakeup_count"] = 1
                conflicts.append(round_id)
            else:
                record["director_escalation_status"] = "DELIVERY_UNKNOWN_FAIL_CLOSED"
                record["director_escalation_error"] = error
            self.save()
        return conflicts

    @staticmethod
    def _valid_director_response(
        path: Path, directive_id: str, directive_hash: str,
    ) -> bool:
        """Recognize a Controller acknowledgement without interpreting its science."""
        try:
            response = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        if (
            response.get("directive_id") != directive_id
            or response.get("controller_role") != "arc-controller"
            or response.get("acknowledged") is not True
            or response.get("scientific_training_started") is not False
        ):
            return False
        # DIRECTOR_RESPONSE_001--003 predate the event path.  Preserve their
        # established acknowledgement schema while requiring a digest for all
        # new directive-driven acknowledgements.
        recorded = response.get("directive_sha256")
        return recorded in (None, directive_hash)

    def reconcile_directives(
        self, directive_dir: Path, response_dir: Path,
    ) -> list[str]:
        """Discover each immutable Director directive and queue one Controller wake.

        The directive SHA256 is the idempotency key.  State is saved before any
        external prompt so a restart cannot generate a second wakeup.  A hash
        conflict fails closed and is never delivered.
        """
        pending: list[str] = []
        for path in sorted(directive_dir.glob("DIRECTOR_DIRECTIVE_*.json")):
            try:
                directive = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            directive_id = str(directive.get("directive_id", ""))
            if not directive_id or directive_id != path.stem:
                continue
            digest = canonical_hash(path)
            record = self.state["directives"].setdefault(directive_id, {})
            prior = record.get("directive_sha256")
            if prior not in (None, digest):
                record["directive_identity_conflict"] = True
                record["directive_processed"] = False
                continue
            record.update({
                "directive_sha256": digest,
                "directive_path": str(path.resolve()),
                "decision": directive.get("decision"),
            })
            response = response_dir / path.name.replace("DIRECTIVE_", "RESPONSE_", 1)
            record["controller_response_path"] = str(response.resolve())
            if self._valid_director_response(response, directive_id, digest):
                record["controller_acknowledged"] = True
                record["directive_processed"] = True
                continue
            if not record.get("controller_wakeup_reserved"):
                notification = self.notification_dir / f"CONTROLLER_DIRECTIVE_NOTIFICATION_{directive_id.removeprefix('DIRECTOR_DIRECTIVE_')}_{digest[:12]}.json"
                atomic_json(notification, {
                    "schema_version": 1,
                    "directive_id": directive_id,
                    "directive_sha256": digest,
                    "directive": str(path.resolve()),
                    "response": str(response.resolve()),
                    "scientific_training_authorized": directive.get("scientific_training_authorized") is True,
                })
                record["controller_notification_path"] = str(notification.resolve())
                pending.append(directive_id)
        self.save()
        return pending

    def wake_controller_for_directives_once(
        self,
        directive_ids: list[str],
        controller_agent: str,
        herdr_timeout_seconds: int,
    ) -> list[str]:
        """Send exactly one prompt for each durable, unacknowledged directive."""
        woken: list[str] = []
        for directive_id in directive_ids:
            record = self.state["directives"][directive_id]
            if record.get("controller_wakeup_reserved") or record.get("directive_processed"):
                continue
            digest = record["directive_sha256"]
            notification = Path(record["controller_notification_path"])
            response = Path(record["controller_response_path"])
            record["controller_wakeup_reserved"] = True
            record["controller_wakeup_status"] = "RESERVED"
            record["controller_wakeup_count"] = 0
            self.save()
            prompt = (
                f"ARC2 Director directive notification {directive_id} ({digest}). "
                f"Read {notification} and execute the complete Controller directive cycle; acknowledgement is not terminal behavior. "
                f"Use scripts/arc2_controller_directive_cycle.py with directive {record['directive_path']}, "
                f"response {response}, and durable Controller state. Preserve any matching acknowledgement. "
                "Route the Director decision exactly: remediation decisions require autonomous repair, validation, commit, brief, and resubmission; "
                "NEW_SUBPROTOCOL_REQUIRED requires a new protocol; PAUSE/STOP halt science; unknown decisions fail closed. "
                "Do not start training unless the directive itself is CONTINUE or CONTINUE_WITH_WARNING and scientific_training_authorized is true."
            )
            exit_code, error = prompt_controller(
                ["herdr", "agent", "prompt", controller_agent, prompt], herdr_timeout_seconds,
            )
            record["controller_wakeup_cli_exit_code"] = exit_code
            if exit_code == 0:
                record["controller_wakeup_status"] = "SENT"
                record["controller_wakeup_count"] = 1
                woken.append(directive_id)
            else:
                record["controller_wakeup_status"] = "DELIVERY_UNKNOWN_FAIL_CLOSED"
                record["controller_wakeup_error"] = error
            self.save()
        return woken

    def acknowledge_directives(self, response_dir: Path) -> list[str]:
        """Mark acknowledged directives processed once, and only once."""
        acknowledged: list[str] = []
        for directive_id, record in self.state["directives"].items():
            if record.get("directive_processed"):
                continue
            response = record.get("controller_response_path")
            if not response:
                continue
            path = Path(response)
            if not path.is_absolute():
                path = response_dir / path.name
            if self._valid_director_response(path, directive_id, record.get("directive_sha256", "")):
                record["controller_acknowledged"] = True
                record["directive_processed"] = True
                acknowledged.append(directive_id)
        if acknowledged:
            self.save()
        return acknowledged

    def resume_nonterminal_directive_cycles(self, cycle_state_path: Path, controller_agent: str, herdr_timeout_seconds: int, max_retries: int = 3, lease_seconds: int = 900) -> list[str]:
        """Lease continuation turns; initial delivery remains a distinct one-shot event.

        An accepted Herdr prompt is not proof of completion.  When the agent is
        idle again, the old lease is over and a further continuation is allowed
        only while the durable Controller cycle remains active.  Two successive
        wakeups with an unchanged durable progress marker are escalated instead
        of producing an infinite loop.
        """
        if not cycle_state_path.exists() or not controller_idle(controller_agent, herdr_timeout_seconds):
            return []
        try:
            cycles = json.loads(cycle_state_path.read_text(encoding="utf-8")).get("directives", {})
        except (OSError, json.JSONDecodeError):
            return []
        resumed: list[str] = []
        for directive_id, cycle in cycles.items():
            status = str(cycle.get("state", ""))
            if status in CONTROLLER_CYCLE_TERMINAL or status in CONTROLLER_CYCLE_WAITING or not status:
                continue
            if status not in CONTROLLER_CYCLE_ACTIVE:
                continue
            record = self.state["directives"].get(directive_id)
            if not record or record.get("directive_sha256") != cycle.get("directive_sha256"):
                continue
            attempts = int(record.get("failure_retry_count", 0))
            if attempts >= max_retries:
                record["controller_cycle_escalation_required"] = True
                continue
            marker_fields = {key: cycle.get(key) for key in ("state", "last_completed_step", "next_step", "last_progress_at", "commit", "brief", "evidence", "validation")}
            marker = hashlib.sha256(json.dumps(marker_fields, sort_keys=True).encode("utf-8")).hexdigest()
            previous = record.get("dispatched_progress_marker")
            no_progress = int(record.get("no_progress_wakeups", 0)) + (1 if previous == marker else 0)
            if previous != marker:
                no_progress = 0
            if no_progress >= 2:
                record.update({"controller_cycle_escalation_required": True, "controller_cycle_escalation_reason": "CONTROLLER_STALL", "no_progress_wakeups": no_progress, "last_observed_progress_marker": marker})
                self.save()
                continue
            sequence = int(record.get("continuation_sequence", 0)) + 1
            invocation = f"{directive_id}:continuation:{sequence}"
            # Herdr is idle, so any prior lease has ended. This is a new
            # continuation, not a duplicate initial notification or ACK.
            record.update({"lifecycle_state": status, "active_controller_lease": True,
                           "lease_started_at": utc_seconds(), "continuation_sequence": sequence,
                           "invocation_id": invocation, "state_before": status,
                           "wakeup_time": utc_seconds(), "last_observed_progress_marker": marker,
                           "dispatched_progress_marker": marker, "no_progress_wakeups": no_progress,
                           "next_step": cycle.get("next_step", "RESUME_UNFINISHED"),
                           "controller_cycle_complete": False})
            self.save()
            prompt = (f"ARC2 Controller recovery invocation {invocation}: resume existing directive cycle {directive_id} from durable state {status}. "
                      f"Do not create another acknowledgement. Read {cycle_state_path} and continue only unfinished steps for {cycle.get('directive_path')}. "
                      "Execute the next unfinished requirement. Do not stop after identifying or describing it. Continue until a valid WAITING or terminal state. "
                      "Do not start training without an authorizing directive.")
            code, error = prompt_controller(["herdr", "agent", "prompt", controller_agent, prompt], herdr_timeout_seconds)
            record["active_controller_lease"] = code == 0
            record["completion_time"] = utc_seconds() if code != 0 else None
            record["state_after"] = cycle.get("state")
            record["controller_resume_status"] = "SENT" if code == 0 else "DELIVERY_UNKNOWN_FAIL_CLOSED"
            if code != 0:
                record["controller_resume_error"] = error
                record["failure_retry_count"] = attempts + 1
            if code == 0: resumed.append(directive_id)
            self.save()
        return resumed

    @staticmethod
    def _seconds_since(timestamp: object) -> float | None:
        if not isinstance(timestamp, str):
            return None
        try:
            parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        except ValueError:
            return None
        return max(0.0, (datetime.now(timezone.utc) - parsed).total_seconds())

    def escalate_controller_cycles(
        self, cycle_state_path: Path, escalation_dir: Path, director_agent: str,
        herdr_timeout_seconds: int, stall_seconds: int = 900,
    ) -> list[str]:
        """Escalate durable Controller impasses once, with Supervisor as transport.

        Ordinary recoverable work stays with the Controller.  This method only
        examines Controller-authored durable state and never decides science.
        Its pre-delivery reservation makes an ambiguous Herdr result fail
        closed rather than notifying the Director twice.
        """
        if not cycle_state_path.exists():
            return []
        try:
            cycles = json.loads(cycle_state_path.read_text(encoding="utf-8")).get("directives", {})
        except (OSError, json.JSONDecodeError):
            return []
        escalation_dir.mkdir(parents=True, exist_ok=True)
        existing = sorted(escalation_dir.glob("CONTROLLER_ESCALATION_REQUEST_*.json"))
        next_number = 1
        if existing:
            numbers = []
            for path in existing:
                try:
                    numbers.append(int(path.stem.rsplit("_", 1)[1]))
                except ValueError:
                    continue
            next_number = max(numbers, default=0) + 1
        woken: list[str] = []
        for directive_id, cycle in sorted(cycles.items()):
            state = str(cycle.get("state", ""))
            record = self.state["directives"].get(directive_id)
            if not record or record.get("director_escalation_reserved"):
                continue
            reasons: list[str] = []
            if state in CONTROLLER_ESCALATION_STATES:
                reasons.append(state)
            attempts = max(int(cycle.get("repair_attempt_count", 0)), int(record.get("failure_retry_count", 0)))
            if attempts >= 2:
                reasons.append("SAME_BLOCKER_SURVIVED_TWO_BOUNDED_REPAIRS")
            if int(record.get("no_progress_wakeups", 0)) >= 2:
                reasons.append("SUPERVISOR_TWO_WAKEUPS_WITHOUT_DURABLE_PROGRESS")
            stale = self._seconds_since(cycle.get("last_progress_at"))
            if state in CONTROLLER_CYCLE_ACTIVE and stale is not None and stale >= stall_seconds:
                reasons.append("ACTIVE_CYCLE_PROGRESS_TIMEOUT")
            if cycle.get("frozen_protocol_change_required") is True or cycle.get("requires_high_level_decision") is True:
                reasons.append("FROZEN_SCIENTIFIC_BOUNDARY_CHANGE_REQUIRED")
            if not reasons:
                continue
            request = escalation_dir / f"CONTROLLER_ESCALATION_REQUEST_{next_number:03d}.json"
            next_number += 1
            payload = {
                "schema_version": 1,
                "status": "ESCALATION_REQUESTED",
                "directive_id": directive_id,
                "directive_sha256": cycle.get("directive_sha256"),
                "blocker": reasons,
                "attempts_already_made": attempts,
                "controller_state": state,
                "last_completed_step": cycle.get("last_completed_step"),
                "next_step": cycle.get("next_step"),
                "evidence": {key: cycle.get(key) for key in ("evidence", "validation", "commit", "brief", "last_progress_at")},
                "requested_decision": "STRUCTURED_DIRECTOR_DIRECTIVE_REQUIRED",
                "scientific_training_started": False,
            }
            atomic_json(request, payload)
            record.update({
                "director_escalation_reserved": True,
                "director_escalation_request_path": str(request.resolve()),
                "director_escalation_reason": reasons,
                "director_escalation_wakeup_count": 0,
            })
            self.save()
            prompt = (
                f"ARC2 Controller escalation request for {directive_id}. Read {request.resolve()} and issue a structured Director directive. "
                "The Controller is blocked; do not start scientific training from this notification."
            )
            code, error = prompt_controller(["herdr", "agent", "prompt", director_agent, prompt], herdr_timeout_seconds)
            record["director_escalation_cli_exit_code"] = code
            if code == 0:
                record["director_escalation_status"] = "SENT"
                record["director_escalation_wakeup_count"] = 1
                woken.append(directive_id)
            else:
                record["director_escalation_status"] = "DELIVERY_UNKNOWN_FAIL_CLOSED"
                record["director_escalation_error"] = error
            self.save()
        return woken


def remote_shell(target: str, script: str, identity_file: str | None = None) -> str:
    """Run a tiny metadata-only script through RunPod's forced-PTY SSH gateway.

    The endpoint presents an interactive shell even for a command request, so
    the script is base64-encoded and sent on stdin. No secrets are logged.
    """
    command = ["ssh", "-F", "NUL", "-tt", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20"]
    if identity_file:
        command.extend(["-i", identity_file])
    command.append(target)
    nonce = secrets.token_hex(16)
    completion = json.dumps({"arc2_remote_nonce": nonce}, sort_keys=True)
    encoded = base64.b64encode((script + f"\nprintf '%s\\n' '{completion}'\n").encode("utf-8")).decode("ascii")
    # RunPod exposes an interactive bash shell with bracketed paste enabled.
    # Paste one command, then submit it with CR; LF-only piped input is merely
    # echoed by that gateway.
    line = f"echo {encoded} | base64 -d | bash; printf '__ARC2_REMOTE_END__\\n'; exit"
    payload = f"\x1b[200~{line}\x1b[201~\r"
    completed = subprocess.run(command, input=payload.encode("utf-8"), capture_output=True, check=False, timeout=45)
    # The forced PTY echoes its submitted command as human terminal output.
    # Long ASCII-safe transport envelopes can be visually corrupted by that
    # terminal echo even when the remote command and its structured result are
    # intact.  Keep protocol markers as raw bytes, and decode only the human
    # portion with replacement.  Consumers of structured JSON still decode and
    # parse their individual JSON record strictly.
    raw_lines = completed.stdout.splitlines()
    # A forced-PTY gateway can echo the bracketed-paste command verbatim.
    # The marker must therefore arrive as its own output line, never merely as
    # text embedded in the echoed command, before a metadata scan is trusted.
    marker = "__ARC2_REMOTE_END__"
    marker_lines = [line.strip() for line in raw_lines]
    # The completion JSON is base64-hidden inside the submitted command, so an
    # echoed command cannot forge it.  Require both it and the terminal marker.
    if completed.returncode != 0 or marker.encode("ascii") not in marker_lines or completion.encode("ascii") not in marker_lines:
        raise RuntimeError(f"remote metadata scan failed (exit={completed.returncode})")
    # Split at the actual marker line, not the marker text embedded in the
    # terminal's echo of the submitted command.
    marker_index = max(index for index, line in enumerate(raw_lines) if line.strip() == marker.encode("ascii"))
    return decode_human_utf8(b"\n".join(raw_lines[:marker_index]))


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
    parser.add_argument("--poll-seconds", type=float)
    parser.add_argument("--controller-agent")
    parser.add_argument("--controller-ack-dir", type=Path)
    parser.add_argument("--directive-dir", type=Path)
    parser.add_argument("--director-response-dir", type=Path)
    parser.add_argument("--controller-cycle-state", type=Path)
    parser.add_argument("--director-agent")
    parser.add_argument("--controller-escalation-dir", type=Path)
    parser.add_argument("--controller-stall-seconds", type=int, default=900)
    parser.add_argument("--herdr-timeout-seconds", type=int, default=15)
    args = parser.parse_args()
    if args.poll_seconds is not None and args.poll_seconds <= 0:
        parser.error("--poll-seconds must be greater than zero")
    if bool(args.controller_agent) != bool(args.controller_ack_dir):
        parser.error("--controller-agent and --controller-ack-dir must be supplied together")
    if bool(args.directive_dir) != bool(args.director_response_dir):
        parser.error("--directive-dir and --director-response-dir must be supplied together")
    if args.directive_dir and not args.controller_agent:
        parser.error("--directive-dir requires --controller-agent")
    if bool(args.director_agent) != bool(args.controller_escalation_dir):
        parser.error("--director-agent and --controller-escalation-dir must be supplied together")
    if args.director_agent and not args.directive_dir:
        parser.error("--director-agent requires --directive-dir")
    if args.controller_stall_seconds <= 0:
        parser.error("--controller-stall-seconds must be greater than zero")
    supervisor = Supervisor(args.state, args.notification_dir)
    if (args.ssh_target or args.remote_root) and not (args.ssh_target and args.remote_root):
        parser.error("--ssh-target and --remote-root must be supplied together")

    while True:
        remote_status = "LOCAL_ONLY"
        if args.ssh_target:
            try:
                fetch_remote_receipts(args.ssh_target, args.remote_root, args.receipt_root, args.identity_file)
                remote_status = "LOW_FREQUENCY_RECONCILIATION_OK"
            except Exception as error:  # Continue local reconciliation; retry next poll.
                remote_status = f"LOW_FREQUENCY_RECONCILIATION_ERROR:{type(error).__name__}"
        notified = supervisor.reconcile(args.receipt_root)
        woken: list[str] = []
        acknowledged: list[str] = []
        directive_notifications: list[str] = []
        directive_wakeups: list[str] = []
        directive_acknowledgements: list[str] = []
        directive_resumes: list[str] = []
        controller_escalations: list[str] = []
        scientific_state_conflicts: list[str] = []
        if args.controller_agent:
            woken = supervisor.wake_controller_once(
                notified,
                args.controller_agent,
                args.controller_ack_dir,
                args.herdr_timeout_seconds,
            )
            acknowledged = supervisor.acknowledge_controller_cycles(args.controller_ack_dir)
            if args.director_agent:
                scientific_state_conflicts = supervisor.detect_scientific_state_conflicts(
                    args.receipt_root, args.controller_ack_dir, args.controller_escalation_dir,
                    args.director_agent, args.herdr_timeout_seconds,
                )
            if args.directive_dir:
                directive_notifications = supervisor.reconcile_directives(
                    args.directive_dir, args.director_response_dir,
                )
                directive_wakeups = supervisor.wake_controller_for_directives_once(
                    directive_notifications, args.controller_agent, args.herdr_timeout_seconds,
                )
                directive_acknowledgements = supervisor.acknowledge_directives(args.director_response_dir)
                cycle_state = args.controller_cycle_state or (args.notification_dir.parent / "controller_state" / "DIRECTIVE_CYCLE_STATE.json")
                directive_resumes = supervisor.resume_nonterminal_directive_cycles(cycle_state, args.controller_agent, args.herdr_timeout_seconds)
                if args.director_agent:
                    controller_escalations = supervisor.escalate_controller_cycles(
                        cycle_state, args.controller_escalation_dir, args.director_agent,
                        args.herdr_timeout_seconds, args.controller_stall_seconds,
                    )
        print(json.dumps({
            "status": "OK",
            "remote_status": remote_status,
            "controller_notifications": notified,
            "controller_wakeups": woken,
            "controller_acknowledgements": acknowledged,
            "director_notifications": directive_notifications,
            "director_wakeups": directive_wakeups,
            "director_acknowledgements": directive_acknowledgements,
            "director_resumes": directive_resumes,
            "controller_escalations": controller_escalations,
            "scientific_state_conflicts": scientific_state_conflicts,
        }, sort_keys=True), flush=True)
        if args.poll_seconds is None:
            return 0
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
