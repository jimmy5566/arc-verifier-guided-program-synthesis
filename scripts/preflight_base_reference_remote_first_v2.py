#!/usr/bin/env python3
"""Read-only identity preflight for the V2 base-reference collector."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
def sha(p: Path) -> str: return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p: Path, value: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True); t = p.with_suffix(p.suffix + ".tmp")
    t.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"); t.replace(p)
def require(p: Path, wanted: str) -> None:
    if not p.is_file(): raise RuntimeError(f"MISSING:{p}")
    if sha(p) != wanted: raise RuntimeError(f"HASH_MISMATCH:{p}")
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--binding", type=Path, required=True); p.add_argument("--contract", type=Path, required=True)
    p.add_argument("--source-root", type=Path, required=True); p.add_argument("--base", type=Path, required=True)
    p.add_argument("--target-dev", type=Path, required=True); p.add_argument("--retention", type=Path, required=True)
    p.add_argument("--ledger", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    out = {"schema_version": 1, "status": "FAIL_CLOSED", "no_model_imported": True, "no_model_loaded": True,
           "no_lora_constructed": True, "no_optimizer_constructed": True, "scientific_training_started": False}
    try:
        binding = json.loads(a.binding.read_text(encoding="utf-8")); contract = json.loads(a.contract.read_text(encoding="utf-8"))
        if binding.get("canonical_ledger") != "orchestration/budget/ARC2_CUMULATIVE_NEW_GPU_TRAINING_LEDGER_V5.jsonl": raise RuntimeError("NONCANONICAL_LEDGER")
        if binding.get("status") != "FROZEN_PENDING_DIRECTOR_BASE_REFERENCE_AUTHORIZATION": raise RuntimeError("BINDING_STATUS_INVALID")
        if contract.get("contract_id") != "ARC_NATIVE_OBSERVATION_ONLY_V1": raise RuntimeError("CONTRACT_ID_INVALID")
        if binding.get("inference_contract_sha256") != sha(a.contract): raise RuntimeError("INFERENCE_CONTRACT_HASH_MISMATCH")
        if any("FINAL_AUDIT" in str(x).upper() or "TRAIN.JSONL" in str(x).upper() for x in (a.target_dev, a.retention)): raise RuntimeError("FORBIDDEN_EVALUATION_INPUT")
        for rel, wanted in binding["source_hashes"].items(): require(a.source_root / rel, wanted)
        for name, wanted in binding["base_files"].items(): require(a.base / name, wanted)
        data = binding["data"]
        require(a.target_dev, data["TARGET_DEV"]["sha256"]); require(a.retention, data["RETENTION_SENTINEL"]["sha256"])
        if a.target_dev.stat().st_size != data["TARGET_DEV"]["bytes"] or a.retention.stat().st_size != data["RETENTION_SENTINEL"]["bytes"]: raise RuntimeError("DATA_SIZE_MISMATCH")
        ledger = [json.loads(line) for line in a.ledger.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(ledger) != 1 or ledger[0] != {"cap_seconds": 28800, "gpu_training_started": False, "record_type": "LEDGER_INITIALIZATION", "schema_version": 1}: raise RuntimeError("V5_LEDGER_CONTENT_INVALID")
        out.update({"status": "PASS_READ_ONLY_BASE_REFERENCE_PREFLIGHT", "binding_sha256": sha(a.binding), "contract_sha256": sha(a.contract),
                    "base_manifest_identity": hashlib.sha256(json.dumps(binding["base_files"], sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                    "target_dev_sha256": data["TARGET_DEV"]["sha256"], "retention_sha256": data["RETENTION_SENTINEL"]["sha256"],
                    "ledger_sha256": sha(a.ledger), "collector_launch_blocked_until_scientific_execution_gate": True})
    except Exception as exc:
        out["failure"] = f"{type(exc).__name__}:{exc}"
    write(a.output, out); print(json.dumps({"status": out["status"]}, sort_keys=True))
    return 0 if out["status"].startswith("PASS_") else 1
if __name__ == "__main__": raise SystemExit(main())
