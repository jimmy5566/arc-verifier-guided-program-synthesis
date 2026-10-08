"""Durable Controller-side routing for immutable Director directives.

The Supervisor only transports a directive exactly once.  This program owns
Controller lifecycle state and never starts GPU work by itself.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REMEDIATION = {"REQUIRE_CURRICULUM_REVIEW", "REQUIRE_DIAGNOSTIC_REVIEW", "REQUIRE_INFRA_REPAIR", "REQUIRE_CHANGES"}
KNOWN = REMEDIATION | {"CONTINUE", "CONTINUE_WITH_WARNING", "NEW_SUBPROTOCOL_REQUIRED", "PAUSE_SCIENTIFIC_EXPERIMENT", "STOP_CURRENT_TRACK", "TERMINAL_SCIENTIFIC_STOP", "EMERGENCY_RECOVERY"}
STEP_ORDER = {"PROCESSING": 0, "REMEDIATION_COMPLETE": 1, "VALIDATED": 2, "COMMITTED": 3, "RESUBMITTED": 4, "WAITING_DIRECTOR": 5}
RESOLUTION_REQUIRED = REMEDIATION | {"NEW_SUBPROTOCOL_REQUIRED", "EMERGENCY_RECOVERY"}
RESOLUTION_KEYS = {"root_cause", "why_current_path_is_invalid", "artifacts_or_conditions_that_must_remain_frozen", "required_resolution", "controller_next_actions", "minimum_acceptance_evidence", "forbidden_actions", "fresh_round_id_required", "fresh_protocol_id_required", "fresh_gate_nonce_output_root_required"}


def cpu_only_continuation(directive: dict[str, Any], decision: str) -> bool:
    """Return true for a Director-approved, non-scientific work substage.

    A CONTINUE_WITH_WARNING decision can authorize bounded CPU work while
    deliberately withholding model-loading and scientific-training authority.
    Required changes in that case are work, not a terminal acknowledgement.
    """
    return (
        decision in {"CONTINUE", "CONTINUE_WITH_WARNING"}
        and directive.get("scientific_training_authorized") is not True
        and bool(directive.get("continuation_authorized_stage"))
        and bool(directive.get("required_changes"))
    )


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    temp.replace(path)


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"schema_version": 1, "directives": {}}
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != 1 or not isinstance(value.get("directives"), dict):
        raise RuntimeError("INVALID_CONTROLLER_DIRECTIVE_STATE")
    return value


def response_payload(directive: dict[str, Any], digest: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "directive_id": directive["directive_id"],
        "directive_sha256": digest,
        "controller_role": "arc-controller",
        "acknowledged": True,
        "action": "DIRECTIVE_ACKNOWLEDGED_NO_TRAINING",
        "scientific_training_started": False,
    }


def valid_response(path: Path, directive: dict[str, Any], digest: str) -> bool:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    expected = response_payload(directive, digest)
    return all(value.get(key) == item for key, item in expected.items())


def valid_resolution_plan(value: Any) -> bool:
    return isinstance(value, dict) and RESOLUTION_KEYS.issubset(value)


def transition(record: dict[str, Any], state: str, **values: Any) -> None:
    if record.get("state") == state and all(record.get(k) == v for k, v in values.items()):
        return
    record.update(values)
    record["state"] = state
    record["last_progress_at"] = now()
    if state in {"REMEDIATION_COMPLETE", "VALIDATED", "COMMITTED", "RESUBMITTED"}:
        record["last_completed_step"] = state
    record.setdefault("history", []).append({"at": now(), "state": state, **values})


def before(record: dict[str, Any], state: str) -> bool:
    return STEP_ORDER.get(record.get("state", ""), -1) < STEP_ORDER[state]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directive", type=Path, required=True)
    parser.add_argument("--response", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--complete-evidence", type=Path)
    parser.add_argument("--validated-evidence", type=Path)
    parser.add_argument("--commit", dest="commit_sha")
    parser.add_argument("--brief", type=Path)
    parser.add_argument("--dummy-remediation", action="store_true")
    parser.add_argument("--blocker-state", choices=["BLOCKED_INFRA", "BLOCKED_SCIENCE", "HARD_BLOCKED", "NEEDS_HIGH_LEVEL_DECISION"])
    parser.add_argument("--blocker-evidence", type=Path)
    parser.add_argument("--repair-attempt-count", type=int)
    parser.add_argument("--frozen-protocol-change-required", action="store_true")
    parser.add_argument("--close-no-director", action="store_true",
                        help="close a completed remediation under the single-controller policy")
    parser.add_argument("--resolution-plan", type=Path,
                        help="legacy bridge only; future Director directives carry controller_resolution_plan")
    args = parser.parse_args()
    directive_bytes = args.directive.read_bytes()
    digest = hashlib.sha256(directive_bytes).hexdigest()
    directive = json.loads(directive_bytes.decode("utf-8"))
    directive_id = str(directive.get("directive_id", ""))
    decision = str(directive.get("decision", ""))
    is_cpu_only_continuation = cpu_only_continuation(directive, decision)
    if not directive_id or directive_id != args.directive.stem:
        raise RuntimeError("DIRECTIVE_IDENTITY_INVALID")
    state = load_state(args.state)
    record = state["directives"].setdefault(directive_id, {"directive_id": directive_id, "directive_sha256": digest})
    if record.get("directive_sha256") != digest:
        transition(record, "FAILED_CLOSED", reason="DIRECTIVE_HASH_CONFLICT")
        atomic_json(args.state, state)
        raise RuntimeError("DIRECTIVE_HASH_CONFLICT")
    if not record.get("state"):
        transition(record, "RECEIVED", decision=decision, directive_path=str(args.directive.resolve()), response_path=str(args.response.resolve()))
    if not args.response.exists():
        atomic_json(args.response, response_payload(directive, digest))
    if not valid_response(args.response, directive, digest):
        transition(record, "FAILED_CLOSED", reason="ACKNOWLEDGEMENT_INVALID")
        atomic_json(args.state, state)
        raise RuntimeError("ACKNOWLEDGEMENT_INVALID")
    if not record.get("acknowledgement_sha256"):
        transition(record, "ACKNOWLEDGED", acknowledgement_sha256=hashlib.sha256(args.response.read_bytes()).hexdigest())
    if decision == "TERMINAL_SCIENTIFIC_STOP" and record.get("reason") == "UNKNOWN_DIRECTOR_DECISION":
        # Preserve the historical fail-closed transition in history, but do not
        # leave it as the current terminal reason once this decision is known.
        record.pop("reason", None)
    # Repair only the legacy acknowledgement-only closure for a non-scientific
    # continuation.  Other terminal states remain immutable.
    if (is_cpu_only_continuation and record.get("state") == "CLOSED"
            and record.get("route") == "UNAUTHORIZED_CONTINUE_FAIL_CLOSED"):
        transition(record, "PROCESSING", route="CPU_ONLY_CONTINUATION",
                   next_step="FIRST_UNFINISHED_DIRECTIVE_REQUIREMENT",
                   controller_cycle_complete=False,
                   recovery="CONTINUE_WITH_WARNING_CPU_ONLY_SUBSTAGE")
    elif not record.get("route"):
        if decision not in KNOWN:
            transition(record, "FAILED_CLOSED", reason="UNKNOWN_DIRECTOR_DECISION")
        elif decision in REMEDIATION:
            transition(record, "PROCESSING", route="REMEDIATION", next_step="FIRST_UNFINISHED_DIRECTIVE_REQUIREMENT", controller_cycle_complete=False)
        elif decision == "NEW_SUBPROTOCOL_REQUIRED":
            transition(record, "PROCESSING", route="NEW_SUBPROTOCOL")
        elif decision == "PAUSE_SCIENTIFIC_EXPERIMENT":
            transition(record, "CLOSED", route="PAUSED_NO_GPU_LAUNCH")
        elif decision in {"STOP_CURRENT_TRACK", "TERMINAL_SCIENTIFIC_STOP"}:
            transition(record, "STOPPED", route="TERMINAL_SCIENTIFIC_STOP", controller_cycle_complete=True,
                       next_step="PRESERVE_TERMINAL_CLOSURE")
        elif decision == "EMERGENCY_RECOVERY":
            transition(record, "PROCESSING", route="ISOLATED_RECOVERY")
        elif is_cpu_only_continuation:
            transition(record, "PROCESSING", route="CPU_ONLY_CONTINUATION",
                       next_step="FIRST_UNFINISHED_DIRECTIVE_REQUIREMENT",
                       controller_cycle_complete=False)
        else:
            transition(record, "CLOSED", route="AUTHORIZED" if directive.get("scientific_training_authorized") is True else "UNAUTHORIZED_CONTINUE_FAIL_CLOSED")
    if (decision in RESOLUTION_REQUIRED or is_cpu_only_continuation) and not record.get("resolution_plan_sha256"):
        plan = directive.get("controller_resolution_plan")
        if plan is None and args.resolution_plan:
            if not args.resolution_plan.is_file():
                raise RuntimeError("MISSING_LEGACY_RESOLUTION_PLAN")
            plan_bytes = args.resolution_plan.read_bytes()
            plan = json.loads(plan_bytes.decode("utf-8"))
        elif plan is not None:
            plan_bytes = json.dumps(plan, sort_keys=True, separators=(",", ":")).encode("utf-8")
        else:
            transition(record, "FAILED_CLOSED", reason="DIRECTOR_RESOLUTION_PLAN_MISSING")
            atomic_json(args.state, state)
            raise RuntimeError("DIRECTOR_RESOLUTION_PLAN_MISSING")
        if not valid_resolution_plan(plan):
            transition(record, "FAILED_CLOSED", reason="DIRECTOR_RESOLUTION_PLAN_INVALID")
            atomic_json(args.state, state)
            raise RuntimeError("DIRECTOR_RESOLUTION_PLAN_INVALID")
        record["resolution_plan_sha256"] = hashlib.sha256(plan_bytes).hexdigest()
        record["resolution_plan_source"] = str(args.resolution_plan.resolve()) if args.resolution_plan else "DIRECTIVE_EMBEDDED"
        record["last_progress_at"] = now()
    if args.repair_attempt_count is not None:
        if args.repair_attempt_count < 0:
            raise RuntimeError("INVALID_REPAIR_ATTEMPT_COUNT")
        record["repair_attempt_count"] = args.repair_attempt_count
    if args.frozen_protocol_change_required:
        record["frozen_protocol_change_required"] = True
    if args.blocker_state:
        if args.blocker_evidence and not args.blocker_evidence.is_file():
            raise RuntimeError("MISSING_BLOCKER_EVIDENCE")
        transition(record, args.blocker_state, next_step="DIRECTOR_ESCALATION_REQUIRED",
                   controller_cycle_complete=False,
                   blocker_evidence=str(args.blocker_evidence.resolve()) if args.blocker_evidence else None)
        atomic_json(args.state, state)
        print(json.dumps({"directive_id": directive_id, "decision": decision, "state": record["state"], "scientific_training_started": False}, sort_keys=True))
        return 0
    if args.dummy_remediation:
        if decision not in REMEDIATION:
            raise RuntimeError("DUMMY_REMEDIATION_REQUIRES_REMEDIATION_DECISION")
        dummy = args.state.parent / f"{directive_id}_DUMMY_REMEDIATION.json"
        if before(record, "REMEDIATION_COMPLETE"):
            atomic_json(dummy, {"directive_id": directive_id, "status": "DUMMY_REMEDIATION_COMPLETE", "gpu_training_started": False})
            transition(record, "REMEDIATION_COMPLETE", evidence=str(dummy.resolve()))
        if before(record, "VALIDATED"): transition(record, "VALIDATED", validation="DUMMY_PASS")
        if before(record, "COMMITTED"): transition(record, "COMMITTED", commit="DUMMY_NO_GIT")
        brief = args.state.parent / f"{directive_id}_DUMMY_BRIEF.json"
        if before(record, "RESUBMITTED"):
            atomic_json(brief, {"directive_id": directive_id, "status": "DUMMY_RESUBMITTED", "gpu_training_started": False})
            transition(record, "RESUBMITTED", brief=str(brief.resolve()))
    if args.complete_evidence:
        if not args.complete_evidence.is_file(): raise RuntimeError("MISSING_REMEDIATION_EVIDENCE")
        if before(record, "REMEDIATION_COMPLETE"):
            transition(record, "REMEDIATION_COMPLETE", evidence=str(args.complete_evidence.resolve()))
            record["next_step"] = "VALIDATE"
    if args.validated_evidence:
        if not args.validated_evidence.is_file(): raise RuntimeError("MISSING_VALIDATION_EVIDENCE")
        if before(record, "VALIDATED"):
            transition(record, "VALIDATED", validation=str(args.validated_evidence.resolve()))
            record["next_step"] = "COMMIT"
    if args.commit_sha:
        if before(record, "COMMITTED"):
            transition(record, "COMMITTED", commit=args.commit_sha)
            record["next_step"] = "PREPARE_AND_SUBMIT_BRIEF"
    if args.brief:
        if not args.brief.is_file(): raise RuntimeError("MISSING_DIRECTOR_BRIEF")
        if before(record, "RESUBMITTED"):
            transition(record, "RESUBMITTED", brief=str(args.brief.resolve()))
            transition(record, "WAITING_DIRECTOR", next_step="AWAIT_DIRECTOR_DIRECTIVE", controller_cycle_complete=False, waiting_reason="DIRECTOR_DECISION_REQUIRED")
    if args.close_no_director:
        if record.get("state") not in {"REMEDIATION_COMPLETE", "VALIDATED", "COMMITTED"}:
            raise RuntimeError("CLOSE_NO_DIRECTOR_REQUIRES_COMPLETED_REMEDIATION")
        transition(record, "CLOSED", route="SINGLE_PRIMARY_CONTROLLER", next_step="SCIENTIFIC_EXECUTION_GATE",
                   controller_cycle_complete=True, closure_reason="DIRECTOR_EXCEPTION_ONLY_POLICY")
    atomic_json(args.state, state)
    print(json.dumps({"directive_id": directive_id, "decision": decision, "state": record["state"], "scientific_training_started": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
