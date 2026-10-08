"""Authorization and one-time nonce consumption for a distinct reconstruction condition.

This module deliberately knows nothing about the closed Foundation-V2 V1
protocol.  A later Director directive must bind the exact V2 contract digest
before a launcher can consume its nonce.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


QUALIFYING_DECISIONS = {"CONTINUE", "CONTINUE_WITH_WARNING"}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def read_json(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    return json.loads(raw.decode("utf-8")), sha256_bytes(raw)


def validate(contract: dict[str, Any], contract_sha256: str, directive: dict[str, Any], directive_sha256: str, gate: dict[str, Any]) -> None:
    """Fail closed unless every identity in the future authorization matches."""
    auth = gate.get("authorization")
    if not isinstance(auth, dict):
        raise RuntimeError("AUTHORIZATION_BINDING_MISSING")
    if gate.get("protocol_id") != contract.get("protocol_id"):
        raise RuntimeError("GATE_PROTOCOL_MISMATCH")
    if auth.get("directive_sha256") != directive_sha256:
        raise RuntimeError("DIRECTIVE_SHA256_MISMATCH")
    if directive.get("decision") not in QUALIFYING_DECISIONS:
        raise RuntimeError("DIRECTIVE_DECISION_NOT_QUALIFYING")
    if directive.get("scientific_training_authorized") is not True:
        raise RuntimeError("DIRECTIVE_SCIENTIFIC_AUTHORIZATION_MISSING")
    if directive.get("protocol_id") != contract.get("protocol_id") or auth.get("protocol_id") != contract.get("protocol_id"):
        raise RuntimeError("DIRECTIVE_PROTOCOL_MISMATCH")
    round_id = contract.get("round_id")
    allowed = directive.get("allowed_round_ids")
    if directive.get("scope") != round_id and not (isinstance(allowed, list) and round_id in allowed):
        raise RuntimeError("DIRECTIVE_ROUND_SCOPE_MISMATCH")
    if directive.get("contract_sha256") != contract_sha256 or auth.get("contract_sha256") != contract_sha256:
        raise RuntimeError("DIRECTIVE_CONTRACT_MISMATCH")
    nonce = contract.get("launch_nonce")
    if not isinstance(nonce, str) or not nonce or auth.get("launch_nonce") != nonce:
        raise RuntimeError("DIRECTIVE_NONCE_MISMATCH")
    if gate.get("AUTO_SCIENTIFIC_EXECUTION_AUTHORIZED") is not True or gate.get("GPU_GATE_READY") is not True:
        raise RuntimeError("MACHINE_GATE_NOT_READY")


def consume_once(consumption_dir: Path, contract: dict[str, Any], contract_sha256: str, directive_sha256: str) -> Path:
    """Atomically reserve a nonce.  Existing receipt means it was already consumed."""
    nonce = str(contract["launch_nonce"])
    consumption_dir.mkdir(parents=True, exist_ok=True)
    receipt = consumption_dir / f"{nonce}.json"
    payload = {
        "schema_version": 1,
        "protocol_id": contract["protocol_id"],
        "round_id": contract["round_id"],
        "contract_sha256": contract_sha256,
        "directive_sha256": directive_sha256,
        "launch_nonce": nonce,
        "status": "CONSUMED_ONCE",
    }
    try:
        fd = os.open(receipt, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise RuntimeError("LAUNCH_NONCE_ALREADY_CONSUMED") from error
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return receipt
