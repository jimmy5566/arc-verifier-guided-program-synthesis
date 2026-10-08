#!/usr/bin/env python3
"""Read-only, fail-closed identity preflight for the reconstruction launcher.

The binding is committed before a Director can authorize optimizer work.  This
program only hashes declared inputs and writes an immutable evidence record; it
does not import torch or construct a model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def checked_file(item: dict[str, Any]) -> dict[str, str]:
    path = Path(item["path"])
    expected = item["sha256"]
    if not path.is_file():
        raise RuntimeError(f"REQUIRED_FILE_MISSING:{path}")
    actual = sha(path)
    if actual != expected:
        raise RuntimeError(f"REQUIRED_FILE_HASH_MISMATCH:{path}")
    return {"path": str(path.resolve()), "sha256": actual}


def checked_dataset(contract: dict[str, Any]) -> dict[str, Any]:
    """Verify every declared shard and reject both missing and extra shards."""
    root = Path(contract["root"])
    manifest = Path(contract["manifest"])
    if not root.is_dir() or not manifest.is_file():
        raise RuntimeError(f"DATASET_CONTRACT_MISSING:{root}")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if canonical(payload) != contract["manifest_content_sha256"]:
        raise RuntimeError(f"DATASET_MANIFEST_CONTENT_MISMATCH:{manifest}")
    expected = {str(row["logical_name"]).split("/")[-1]: str(row["sha256"]) for row in payload[contract["manifest_key"]]}
    actual = {p.name: p for p in root.glob("*.parquet")}
    if set(actual) != set(expected):
        raise RuntimeError(f"DATASET_SHARD_SET_MISMATCH:{root}")
    checked = []
    for name, expected_hash in sorted(expected.items()):
        actual_hash = sha(actual[name])
        if actual_hash != expected_hash:
            raise RuntimeError(f"DATASET_SHARD_HASH_MISMATCH:{actual[name]}")
        checked.append({"path": str(actual[name].resolve()), "sha256": actual_hash})
    return {"root": str(root.resolve()), "manifest": str(manifest.resolve()), "shards": checked}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--binding", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--expected-source-sha")
    args = p.parse_args()
    raw = args.binding.read_bytes()
    binding = json.loads(raw.decode("utf-8"))
    output = {"schema_version": 1, "status": "FAIL_CLOSED", "launch_binding_sha256": hashlib.sha256(raw).hexdigest(), "no_optimizer_constructed": True}
    try:
        if binding.get("schema_version") not in {1, 2} or binding.get("status") not in {"FROZEN_PENDING_PREFLIGHT", "FROZEN_PENDING_GPU_RUNTIME_PREFLIGHT"}:
            raise RuntimeError("INVALID_RECONSTRUCTION_LAUNCH_BINDING")
        source = binding["source_provenance"]
        if args.expected_source_sha:
            source_root = Path(source["checked_out_root"])
            here = subprocess.check_output(["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True).strip()
            if here != args.expected_source_sha:
                raise RuntimeError("LOCAL_SOURCE_SHA_MISMATCH")
        verified = [checked_file(item) for item in binding["required_files"]]
        datasets = [checked_dataset(item) for item in binding.get("dataset_contracts", [])]
        output_parent = Path(binding["output"]["parent"])
        if not output_parent.is_dir():
            raise RuntimeError(f"OUTPUT_PARENT_MISSING:{output_parent}")
        output.update({"status": "PASS", "source": source, "verified_files": verified, "verified_datasets": datasets, "command": binding["command"], "output": binding["output"]})
    except Exception as error:
        output["failure"] = f"{type(error).__name__}:{error}"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"status": output["status"], "launch_binding_sha256": output["launch_binding_sha256"]}, sort_keys=True))
    return 0 if output["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
