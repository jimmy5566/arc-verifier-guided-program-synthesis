#!/usr/bin/env python3
"""Read-only final-commit preflight for RECONSTRUCTION_FROM_AVAILABLE_FROZEN_INPUTS_V2.

The program never imports a model library.  It verifies all declared byte
identities, the sealed-boundary policy, fresh output paths, and source parity
before any future optimizer construction is even possible.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def check_file(item: dict[str, Any]) -> dict[str, Any]:
    path = Path(item["path"])
    if not path.is_file():
        raise RuntimeError(f"REQUIRED_FILE_MISSING:{path}")
    if item.get("bytes") is not None and path.stat().st_size != int(item["bytes"]):
        raise RuntimeError(f"REQUIRED_FILE_SIZE_MISMATCH:{path}")
    actual = sha256(path)
    if actual != item["sha256"]:
        raise RuntimeError(f"REQUIRED_FILE_HASH_MISMATCH:{path}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": actual}


def check_dataset(item: dict[str, Any]) -> dict[str, Any]:
    root = Path(item["root"])
    manifest = Path(item["manifest"])
    if not root.is_dir() or not manifest.is_file():
        raise RuntimeError(f"DATASET_CONTRACT_MISSING:{root}")
    manifest_value = json.loads(manifest.read_text(encoding="utf-8"))
    if canonical(manifest_value) != item["manifest_content_sha256"]:
        raise RuntimeError(f"DATASET_MANIFEST_HASH_MISMATCH:{manifest}")
    expected = {Path(str(row["logical_name"])).name: str(row["sha256"]) for row in manifest_value[item["manifest_key"]]}
    actual = {path.name: path for path in root.glob("*.parquet")}
    if set(actual) != set(expected):
        raise RuntimeError(f"DATASET_SHARD_SET_MISMATCH:{root}")
    hashes = []
    for name in sorted(expected):
        actual_hash = sha256(actual[name])
        if actual_hash != expected[name]:
            raise RuntimeError(f"DATASET_SHARD_HASH_MISMATCH:{actual[name]}")
        hashes.append({"path": str(actual[name]), "sha256": actual_hash})
    return {"root": str(root), "manifest": str(manifest), "shards": hashes}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-source-commit", required=True)
    args = parser.parse_args()
    raw = args.binding.read_bytes()
    binding = json.loads(raw.decode("utf-8"))
    evidence: dict[str, Any] = {
        "schema_version": 1,
        "status": "FAIL_CLOSED",
        "no_model_import_or_load": True,
        "no_optimizer_constructed": True,
        "binding_path": str(args.binding),
        "binding_sha256": hashlib.sha256(raw).hexdigest(),
        "expected_source_commit": args.expected_source_commit,
    }
    try:
        if binding.get("protocol_id") not in {"RECONSTRUCTION_FROM_AVAILABLE_FROZEN_INPUTS_V2", "FOUNDATION_V2_RECONSTRUCTION_AND_TARGETED_REPAIR_V2"}:
            raise RuntimeError("PROTOCOL_ID_MISMATCH")
        source_root = Path(binding["source_provenance"]["checked_out_root"])
        head = subprocess.check_output(["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True).strip()
        if head != args.expected_source_commit:
            raise RuntimeError("SOURCE_COMMIT_MISMATCH")
        executable_hashes = []
        for item in binding["source_provenance"]["executable_files"]:
            executable_hashes.append(check_file({"path": source_root / item["path"], "sha256": item["sha256"]}))
        required = [check_file(item) for item in binding["required_files"]]
        datasets = [check_dataset(item) for item in binding["dataset_contracts"]]
        replay = check_file(binding["replay_contract"])
        policy = binding["scientific_boundaries"]
        forbidden = [str(x).lower() for x in policy["forbidden_path_terms"]]
        declared_paths = [str(item["path"]) for item in binding["required_files"]] + [str(item["root"]) for item in binding["dataset_contracts"]] + [str(binding["replay_contract"]["path"])]
        if any(term in path.lower() for term in forbidden for path in declared_paths):
            raise RuntimeError("FORBIDDEN_GOLD_OR_FINAL_AUDIT_PATH")
        for name, target in binding["fresh_paths"].items():
            if Path(target).exists():
                raise RuntimeError(f"FRESH_PATH_ALREADY_EXISTS:{name}:{target}")
        ledger = binding["budget_contract"]
        if ledger["cap_seconds"] != 28800 or ledger["reservation_seconds"] != 7200 or ledger["scientific_seconds_before_launch"] != 0:
            raise RuntimeError("BUDGET_CONTRACT_INVALID")
        evidence.update({
            "status": "PASS",
            "source_head": head,
            "verified_executables": executable_hashes,
            "verified_files": required,
            "verified_datasets": datasets,
            "verified_replay": replay,
            "fresh_paths": binding["fresh_paths"],
            "budget_contract": ledger,
            "scientific_boundaries": policy,
        })
    except Exception as error:
        evidence["first_failure"] = f"{type(error).__name__}:{error}"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"status": evidence["status"], "first_failure": evidence.get("first_failure")}, sort_keys=True))
    return 0 if evidence["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
