#!/usr/bin/env python3
"""No-model runtime identity preflight for the batch-sensitivity localization.

This module deliberately imports neither torch nor transformers.  It verifies
the immutable input package and hashes every model/tokenizer/adapter file that
the later worker could load, while caching shared base-model checks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "experiments/capability_repair_baseline_v1/unified_native_model_capability_baseline_v1"
AVAILABLE = ("RECONSTRUCTED_FOUNDATION_V2_V7", "TARGETED_R1", "TARGETED_R2", "ATOMIC_R1", "FAMILY_BALANCED")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"PREFLIGHT_FILE_MISSING:{path}")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def verify_entries(root: Path, entries: list[dict[str, Any]]) -> list[str]:
    checked: list[str] = []
    for entry in entries:
        path = root / entry["name"]
        if not path.is_file() or path.stat().st_size != int(entry["bytes"]):
            raise RuntimeError(f"RUNTIME_IDENTITY_MISMATCH:{path}")
        if sha(path) != str(entry["sha256"]):
            raise RuntimeError(f"RUNTIME_IDENTITY_MISMATCH:{path}")
        checked.append(str(path))
    return checked


def stable_digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def validate(package: Path, expected_source_commit: str) -> dict[str, Any]:
    if not expected_source_commit or len(expected_source_commit) != 40:
        raise RuntimeError("EXPECTED_SOURCE_COMMIT_INVALID")
    if os.environ.get("ARC2_EXPECTED_SOURCE_COMMIT") != expected_source_commit:
        raise RuntimeError("SOURCE_COMMIT_ENV_BINDING_MISMATCH")
    manifest_path = package / "SYNTHETIC_BENCHMARK_INPUT_MANIFEST_V1.json"
    discovery_path = package / "CHECKPOINT_PROVENANCE_DISCOVERY_V1.json"
    worker_path = ROOT / "scripts/run_unified_native_batch_sensitivity_localization_v1.py"
    launcher_path = ROOT / "scripts/launch_unified_native_batch_sensitivity_localization_v1.py"
    config_path = package / "UNIFIED_NATIVE_BATCH_SENSITIVITY_LOCALIZATION_V1_CONFIG_V3.json"
    manifest, discovery = load(manifest_path), load(discovery_path)
    rows = manifest.get("episodes", [])
    if manifest.get("status") != "INPUTS_FROZEN_TARGETS_SEALED_CPU_ONLY" or len(rows) != 60:
        raise RuntimeError("INPUT_MANIFEST_INVALID")
    if any("output" in row.get("observation", {}).get("task", {}).get("test", [{}])[0] for row in rows):
        raise RuntimeError("SEALED_TARGET_LEAK_IN_INPUT_MANIFEST")
    if manifest.get("content_level_audit", {}).get("gold_dgold_final_audit_accessed") is not False:
        raise RuntimeError("FINAL_AUDIT_OR_GOLD_ACCESS_FORBIDDEN")
    records = discovery.get("checkpoint_records", {})
    if records.get("HISTORICAL_FOUNDATION_V2", {}).get("status") != "UNAVAILABLE_NOT_SUBSTITUTED":
        raise RuntimeError("HISTORICAL_REFERENCE_SUBSTITUTION_FORBIDDEN")
    if any(records.get(name, {}).get("status") != "LOCAL_MANIFEST_VERIFIED_REMOTE_RUNTIME_PENDING" for name in AVAILABLE):
        raise RuntimeError("CHECKPOINT_DISCOVERY_SET_INVALID")
    base_cache: set[str] = set()
    checked_base: list[str] = []
    checked_adapters: dict[str, list[str]] = {}
    for condition in AVAILABLE:
        record = records[condition]
        manifest_file = ROOT / record["manifest_path"]
        if sha(manifest_file) != record["manifest_sha256"]:
            raise RuntimeError(f"CHECKPOINT_MANIFEST_SHA_MISMATCH:{condition}")
        checkpoint = load(manifest_file)
        base_root = Path(checkpoint["base_path"])
        base_key = stable_digest({"root": str(base_root), "entries": checkpoint["base_files"]})
        if base_key not in base_cache:
            checked_base.extend(verify_entries(base_root, checkpoint["base_files"]))
            base_cache.add(base_key)
        adapter_paths = verify_entries(Path(checkpoint["adapter_path"]), checkpoint["adapter_files"])
        expected_adapter = record.get("adapter_model_sha256")
        actual_adapter = next((entry["sha256"] for entry in checkpoint["adapter_files"] if entry["name"] == "adapter_model.safetensors"), None)
        if actual_adapter != expected_adapter:
            raise RuntimeError(f"ADAPTER_IDENTITY_MISMATCH:{condition}")
        checked_adapters[condition] = adapter_paths
    return {
        "status": "PASS_NO_MODEL_IMPORT",
        "expected_source_commit": expected_source_commit,
        "input_manifest_sha256": sha(manifest_path),
        "checkpoint_discovery_sha256": sha(discovery_path),
        "worker_sha256": sha(worker_path),
        "launcher_sha256": sha(launcher_path),
        "config_sha256": sha(config_path),
        "conditions": list(AVAILABLE),
        "base_files_verified_once": len(checked_base),
        "adapter_files_verified_by_condition": {key: len(value) for key, value in checked_adapters.items()},
        "model_imported": False,
        "gpu_used": False,
        "optimizer_steps": 0,
        "final_audit_opened": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError("OUTPUT_NON_OVERWRITE_REQUIRED")
    result = validate(PACKAGE, args.expected_source_commit)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    os.replace(temporary, args.out)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

