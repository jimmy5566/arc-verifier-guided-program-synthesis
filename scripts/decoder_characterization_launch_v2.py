#!/usr/bin/env python3
"""Fail-closed admission checks for the target-blind decoder cohort.

This module deliberately has no model imports.  It validates the immutable
launch binding before the worker is allowed to import torch/transformers.
"""
from __future__ import annotations
import hashlib, json, os
from pathlib import Path
from typing import Any

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

def contract_identity(contract: dict[str, Any]) -> str:
    return hashlib.sha256(canonical({k: v for k, v in contract.items() if k != "contract_sha256"})).hexdigest()

def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes((json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
    os.replace(tmp, path)

def require_file_hash(binding: dict[str, Any]) -> None:
    path = Path(binding["path"])
    if not path.is_file() or sha(path) != binding["sha256"]:
        raise RuntimeError("LAUNCH_BINDING_HASH_MISMATCH")

def read_json_utf8_lf(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    if not raw.endswith(b"\n") or raw.endswith(b"\\n") or b"\r\n" in raw:
        raise RuntimeError("MALFORMED_PREFLIGHT_OR_JSON_RECEIPT")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("MALFORMED_PREFLIGHT_OR_JSON_RECEIPT") from exc
    if not isinstance(value, dict):
        raise RuntimeError("MALFORMED_PREFLIGHT_OR_JSON_RECEIPT")
    return value

def validate_cohort(contract: dict[str, Any]) -> None:
    cohort = read_json_utf8_lf(Path(contract["cohort"]["path"]))
    if sha(Path(contract["cohort"]["path"])) != contract["cohort"]["sha256"]:
        raise RuntimeError("COHORT_HASH_MISMATCH")
    ids = cohort.get("episode_ids")
    roles = cohort.get("role_counts")
    if not isinstance(ids, list) or len(ids) != 12 or len(set(ids)) != 12 or roles != {"TARGETED_EVALUATION": 4, "TARGETED_COMPOSITION": 4, "RETENTION_SENTINEL": 4}:
        raise RuntimeError("COHORT_SELECTION_MISMATCH")
    if cohort.get("selection_rule") != "for each required role, choose the four smallest SHA256(ARC2_TOKEN_CHARACTERIZATION_V1:episode_id) values; selection uses episode IDs and roles only":
        raise RuntimeError("COHORT_SELECTION_RULE_MISMATCH")
    if not all(isinstance(item, str) and item for item in ids):
        raise RuntimeError("COHORT_SELECTION_MISMATCH")

def validate_datasets(contract: dict[str, Any]) -> None:
    """Verify bytes and row counts without parsing or inspecting test targets."""
    checks = (("target_dev_path", "target_dev_sha256", 192), ("retention_path", "retention_sha256", 96))
    for path_key, hash_key, rows in checks:
        path = Path(contract["datasets"][path_key])
        if not path.is_file() or sha(path) != contract["datasets"][hash_key]:
            raise RuntimeError("DATASET_BINDING_MISMATCH")
        with path.open("rb") as handle:
            count = sum(1 for line in handle if line.strip())
        if count != rows:
            raise RuntimeError("DATASET_ROW_COUNT_MISMATCH")

def validate_checkpoint_manifest(contract: dict[str, Any]) -> None:
    """Rehash every declared mounted model/adapter file before import."""
    manifest_path = Path(contract["checkpoint_manifest_path"])
    manifest = read_json_utf8_lf(manifest_path)
    if sha(manifest_path) != contract["checkpoint_manifest_sha256"]:
        raise RuntimeError("CHECKPOINT_MANIFEST_HASH_MISMATCH")
    for kind, root_key, files_key in (("BASE", "base_path", "base_files"), ("ADAPTER", "adapter_path", "adapter_files")):
        root = Path(manifest[root_key])
        declared = manifest.get(files_key)
        if not root.is_dir() or not isinstance(declared, list) or not declared:
            raise RuntimeError(f"{kind}_MANIFEST_INVALID")
        expected = {entry.get("name"): entry for entry in declared if isinstance(entry, dict) and isinstance(entry.get("name"), str)}
        actual = {item.name: item for item in root.iterdir() if item.is_file()}
        if set(expected) != set(actual):
            raise RuntimeError(f"{kind}_MANIFEST_EXTRA_OR_MISSING_FILE")
        for name, entry in expected.items():
            item = actual[name]
            if item.stat().st_size != entry.get("bytes") or sha(item) != entry.get("sha256"):
                raise RuntimeError(f"{kind}_MANIFEST_CONTENT_MISMATCH")

def validate_preflight(contract: dict[str, Any], *, source_commit: str) -> dict[str, Any]:
    receipt_path = Path(contract["preflight_receipt_path"])
    receipt = read_json_utf8_lf(receipt_path)
    required = {"status": "PASS_NO_MODEL_IMPORT", "source_commit": contract["runtime_source_commit"], "worker_source_commit": source_commit, "launch_contract_sha256": contract["contract_sha256"], "no_target_access": True, "model_loaded": False}
    if any(receipt.get(k) != v for k, v in required.items()):
        raise RuntimeError("PREFLIGHT_BINDING_MISMATCH")
    return receipt

def validate_contract(contract_path: Path, *, argv: list[str], environment: dict[str, str], require_review: Path | None, consume_nonce: bool) -> dict[str, Any]:
    contract = read_json_utf8_lf(contract_path)
    if contract.get("contract_id") != "ARC2_DECODER_CHARACTERIZATION_LAUNCH_V2":
        raise RuntimeError("LAUNCH_CONTRACT_ID_INVALID")
    if contract.get("contract_sha256") != contract_identity(contract):
        raise RuntimeError("LAUNCH_CONTRACT_IDENTITY_MISMATCH")
    expected_argv = contract.get("argv") if require_review is not None else contract.get("preflight_argv")
    if expected_argv != argv:
        raise RuntimeError("ARGV_BINDING_MISMATCH")
    if contract.get("environment") != environment:
        raise RuntimeError("ENVIRONMENT_BINDING_MISMATCH")
    for binding in contract["immutable_files"].values():
        require_file_hash(binding)
    validate_cohort(contract)
    validate_datasets(contract)
    validate_checkpoint_manifest(contract)
    source = Path(contract["source_root"])
    head = (source / ".git").exists()
    if not head:
        raise RuntimeError("SOURCE_ROOT_INVALID")
    import subprocess
    live = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    worker_commit = contract["worker_source_commit"]
    if live != contract["runtime_source_commit"]:
        raise RuntimeError("RUNTIME_SOURCE_COMMIT_MISMATCH")
    if require_review is not None:
        if require_review.resolve() != Path(contract["governor_review_path"]).resolve():
            raise RuntimeError("GOVERNOR_REVIEW_PATH_BINDING_MISMATCH")
        review = read_json_utf8_lf(require_review)
        required = {"decision": "CONTINUE_CONTROLLER", "reviewed_brief_sha256": contract["reviewed_brief_sha256"], "authorization_request_sha256": contract["authorization_request_sha256"], "launch_contract_sha256": contract["contract_sha256"], "worker_source_commit": worker_commit, "cohort_sha256": contract["cohort"]["sha256"]}
        if any(review.get(k) != v for k, v in required.items()):
            raise RuntimeError("GOVERNOR_REVIEW_BINDING_MISMATCH")
        # The actual model invocation must never overwrite an earlier run.
        for key in ("output_path", "receipt_path", "terminal_failure_receipt_path"):
            if Path(contract[key]).exists():
                raise RuntimeError("OUTPUT_PATH_NON_OVERWRITE_REQUIRED")
    if consume_nonce:
        nonce = Path(contract["nonce_path"])
        if not nonce.is_file() or sha(nonce) != contract["nonce_sha256"]:
            raise RuntimeError("NONCE_BINDING_MISMATCH")
        consumed = Path(contract["nonce_consumed_path"])
        try:
            fd = os.open(consumed, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as exc:
            raise RuntimeError("NONCE_ALREADY_CONSUMED") from exc
        with os.fdopen(fd, "wb") as handle:
            handle.write((json.dumps({"contract_sha256": contract["contract_sha256"], "status": "CONSUMED_ONCE"}, sort_keys=True) + "\n").encode("utf-8"))
    return contract

def terminal_failure(contract: dict[str, Any] | None, reason: str) -> None:
    if contract is None: return
    path = Path(contract["terminal_failure_receipt_path"])
    if path.exists(): return
    atomic_json(path, {"schema_version": 1, "status": "ADMISSION_FAILURE", "reason": reason, "scientific_training_started": False, "model_loaded": False, "contract_sha256": contract.get("contract_sha256")})
