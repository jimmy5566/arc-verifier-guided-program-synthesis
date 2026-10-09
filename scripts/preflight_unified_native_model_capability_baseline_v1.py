#!/usr/bin/env python3
"""CPU admission checks for the unified native capability baseline.

This verifier intentionally has no transformers/torch imports.  It validates
only immutable package structure before a separately bound runtime preflight.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/capability_repair_baseline_v1/unified_native_model_capability_baseline_v1"

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))

def validate(package: Path) -> dict:
    manifest_path = package / "SYNTHETIC_BENCHMARK_INPUT_MANIFEST_V1.json"
    discovery_path = package / "CHECKPOINT_PROVENANCE_DISCOVERY_V1.json"
    contract_path = package / "EXECUTION_CONTRACT_DRAFT_V1.json"
    manifest, discovery, contract = map(load, (manifest_path, discovery_path, contract_path))
    if manifest.get("status") != "INPUTS_FROZEN_TARGETS_SEALED_CPU_ONLY":
        raise RuntimeError("INPUT_MANIFEST_STATUS_INVALID")
    rows = manifest.get("episodes")
    if not isinstance(rows, list) or len(rows) != 60:
        raise RuntimeError("BENCHMARK_DENOMINATOR_INVALID")
    ids = [row.get("episode_id") for row in rows]
    hashes = [row.get("observation_sha256") for row in rows]
    if len(set(ids)) != 60 or len(set(hashes)) != 60:
        raise RuntimeError("BENCHMARK_IDENTITY_COLLISION")
    if any("output" in row.get("observation", {}).get("task", {}).get("test", [{}])[0] for row in rows):
        raise RuntimeError("SEALED_TARGET_LEAK_IN_INPUT_MANIFEST")
    if manifest.get("content_level_audit", {}).get("gold_dgold_final_audit_accessed") is not False:
        raise RuntimeError("FINAL_AUDIT_OR_GOLD_ACCESS_FORBIDDEN")
    sidecar = manifest.get("sealed_target_sidecar", {})
    if not isinstance(sidecar.get("sha256"), str) or sidecar.get("not_committed") is not True:
        raise RuntimeError("SEALED_TARGET_SIDECAR_CONTRACT_INVALID")
    records = discovery.get("checkpoint_records", {})
    if records.get("HISTORICAL_FOUNDATION_V2", {}).get("status") != "UNAVAILABLE_NOT_SUBSTITUTED":
        raise RuntimeError("HISTORICAL_REFERENCE_SUBSTITUTION_FORBIDDEN")
    available = [key for key, value in records.items() if value.get("status") == "LOCAL_MANIFEST_VERIFIED_REMOTE_RUNTIME_PENDING"]
    if set(available) != {"RECONSTRUCTED_FOUNDATION_V2_V7", "TARGETED_R1", "TARGETED_R2", "ATOMIC_R1", "FAMILY_BALANCED"}:
        raise RuntimeError("CHECKPOINT_DISCOVERY_SET_INVALID")
    if contract.get("input_manifest_sha256") != sha(manifest_path) or contract.get("checkpoint_discovery_sha256") != sha(discovery_path):
        raise RuntimeError("CONTRACT_BINDING_MISMATCH")
    inference = contract.get("inference_contract", {})
    if inference.get("batch_size") != 1 or inference.get("precision") != "BF16" or inference.get("primary") != "GREEDY":
        raise RuntimeError("INFERENCE_CONDITION_INVALID")
    forbidden = set(contract.get("forbidden", []))
    if not {"training", "optimizer", "backward", "TTT", "augmentation", "DFS", "Beam-4", "external_selector", "Gold", "dGold", "FINAL_AUDIT"}.issubset(forbidden):
        raise RuntimeError("FORBIDDEN_SET_INCOMPLETE")
    return {"status": "CPU_PREFLIGHT_PASS_RUNTIME_IDENTITY_PENDING", "input_manifest_sha256": sha(manifest_path), "checkpoint_discovery_sha256": sha(discovery_path), "contract_sha256": sha(contract_path), "available_checkpoint_conditions": available, "historical_foundation_v2": "UNAVAILABLE_NOT_SUBSTITUTED", "runtime_model_imported": False, "gpu_used": False}

def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--package", type=Path, default=OUT); parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(); result = validate(args.package)
    args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(result, sort_keys=True)); return 0

if __name__ == "__main__": raise SystemExit(main())
