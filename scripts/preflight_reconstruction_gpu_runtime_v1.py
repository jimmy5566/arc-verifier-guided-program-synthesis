#!/usr/bin/env python3
"""No-model, fail-closed runtime preflight for Foundation-V2 reconstruction."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    os.replace(temp, path)


def require(value: bool, reason: str) -> None:
    if not value:
        raise RuntimeError(reason)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--identity-preflight", type=Path, required=True)
    parser.add_argument("--data-policy-audit", type=Path, required=True)
    parser.add_argument("--ledger-snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.binding.read_bytes()
    output: dict[str, Any] = {
        "schema_version": 1,
        "status": "FAIL_CLOSED",
        "no_model_loaded": True,
        "no_optimizer_constructed": True,
        "scientific_gpu_training_seconds": 0.0,
        "binding_sha256": hashlib.sha256(raw).hexdigest(),
    }
    try:
        binding = json.loads(raw.decode("utf-8"))
        require(binding.get("schema_version") == 2, "INVALID_BINDING_SCHEMA")
        require(binding.get("status") == "FROZEN_PENDING_GPU_RUNTIME_PREFLIGHT", "INVALID_BINDING_STATUS")
        runtime = binding["runtime_contract"]
        require(Path(runtime["interpreter"]).resolve() == Path(sys.executable).resolve(), "WRONG_PYTHON_INTERPRETER")
        import torch  # import/version/CUDA inspection only; never construct a model
        observed = {
            "interpreter": sys.executable,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "transformers": importlib.metadata.version("transformers"),
            "peft": importlib.metadata.version("peft"),
            "bitsandbytes": importlib.metadata.version("bitsandbytes"),
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
            "bf16_supported": torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
        for key in ("python", "torch", "cuda", "transformers", "peft", "bitsandbytes", "gpu"):
            require(observed[key] == runtime[key], f"RUNTIME_IDENTITY_MISMATCH:{key}")
        require(observed["cuda_available"] and observed["device_count"] == 1 and observed["bf16_supported"], "CUDA_OR_BF16_UNAVAILABLE")
        identity = json.loads(args.identity_preflight.read_text(encoding="utf-8"))
        require(identity.get("status") == "PASS" and identity.get("no_optimizer_constructed") is True, "IDENTITY_PREFLIGHT_NOT_PASS")
        policy = json.loads(args.data_policy_audit.read_text(encoding="utf-8"))
        require(policy.get("status") == "PASS", "TARGET_BLIND_DATA_POLICY_NOT_PASS")
        checks = policy.get("checks", {})
        require(all(checks.get(key) == 0 for key in ("novel_holdout_rows", "novel_validation_rows", "hard_exclude_rows", "quarantine_rows", "retention_sentinel_rows_in_schedule")), "LEAKAGE_OR_SENTINEL_CONTAMINATION")
        ledger = json.loads(args.ledger_snapshot.read_text(encoding="utf-8"))
        require(ledger.get("status") == "PASS" and ledger.get("scientific_gpu_training_seconds") == 0.0 and ledger.get("remaining_seconds") == 28800.0, "V6_BUDGET_LEDGER_NOT_ZERO")
        root = Path(binding["source_provenance"]["checked_out_root"])
        for item in binding["source_provenance"]["files"]:
            require(sha(root / item["path"]) == item["sha256"], f"SOURCE_FILE_HASH_MISMATCH:{item['path']}")
        git_head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        output.update({
            "status": "PASS", "runtime": observed, "runtime_contract": runtime,
            "source_head": git_head, "reviewed_executable_commit": binding["source_provenance"]["reviewed_executable_commit"],
            "identity_preflight_sha256": sha(args.identity_preflight),
            "data_policy_audit_sha256": sha(args.data_policy_audit),
            "ledger_snapshot_sha256": sha(args.ledger_snapshot),
            "output": binding["output"], "reservation_seconds": binding["reservation_seconds"],
            "final_audit_accessed": False,
        })
    except Exception as exc:
        output["failure"] = f"{type(exc).__name__}:{exc}"
    atomic(args.output, output)
    print(json.dumps({"status": output["status"], "no_model_loaded": True, "no_optimizer_constructed": True}, sort_keys=True))
    return 0 if output["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
