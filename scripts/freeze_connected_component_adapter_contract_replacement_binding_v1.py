#!/usr/bin/env python3
"""Freeze the minimal all-B1 binding replacement after adapter-closure repair.

This is an engineering-only replacement: it preserves the protocol, cohort,
checkpoint weights, seeds, decoder and all 168/48 partition identities from a
predecessor binding.  It changes only the bound source closure, validates and
records the PEFT configuration bytes that were previously omitted, and assigns
a fresh one-shot nonce/output root.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, sort_keys=True, indent=2).encode("utf-8") + b"\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predecessor", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--nonce", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--control-root", required=True)
    parser.add_argument("--adapter-config-sha", action="append", default=[], metavar="CHECKPOINT=SHA256")
    args = parser.parse_args()

    predecessor = json.loads(args.predecessor.read_text(encoding="utf-8"))
    execution = dict(predecessor["execution"])
    if (execution.get("reused_rows"), execution.get("new_free_running_rows"),
            execution.get("requested_batch_size"), execution.get("teacher_forced_physical_batch_size"),
            execution.get("retry")) != (168, 48, 1, 1, False):
        raise RuntimeError("PREDECESSOR_SCIENCE_SHAPE_MISMATCH")
    if execution.get("optimizer_steps") not in (0, None) or execution.get("backward_calls") not in (0, None):
        raise RuntimeError("PREDECESSOR_NO_UPDATE_MISMATCH")
    config_hashes = {}
    for item in args.adapter_config_sha:
        checkpoint, separator, digest = item.partition("=")
        if not separator or len(digest) != 64:
            raise RuntimeError("ADAPTER_CONFIG_SHA_ARGUMENT_INVALID")
        config_hashes[checkpoint] = digest
    runtime = {key: dict(value) for key, value in predecessor["checkpoint_runtime"].items()}
    if set(runtime) != set(config_hashes):
        raise RuntimeError("ADAPTER_CONFIG_CHECKPOINT_SET_MISMATCH")
    for checkpoint, details in runtime.items():
        details["adapter_config_sha256"] = config_hashes[checkpoint]

    for relative in predecessor["source_hashes"]:
        if not (ROOT / relative).is_file():
            raise RuntimeError("SOURCE_CLOSURE_FILE_MISSING:" + relative)
    source_hashes = {relative: sha(ROOT / relative) for relative in predecessor["source_hashes"]}
    execution.update({"nonce": args.nonce, "output_root": args.output_root,
                      "control_root": args.control_root, "retry": False,
                      "fresh_nonce_required": True, "fresh_output_root_required": True})
    binding = dict(predecessor)
    binding.update({
        "status": "CPU_PREFLIGHT_READY_OWNER_ONE_SHOT_AUTHORIZATION_REQUIRED",
        "execution_source_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "runtime_source_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "source_hashes": source_hashes,
        "checkpoint_runtime": runtime,
        "execution": execution,
        "authorization": {
            "owner_one_shot_required": True,
            "not_authorized_by_this_binding": True,
            "permitted_launches": 1,
            "retry": False,
            "predecessor_binding_sha256": sha(args.predecessor),
            "predecessor_authorization_consumed": True,
        },
        "adapter_contract_repair": {
            "cause": "PEFT_REQUIRES_ADAPTER_DIRECTORY_AND_BOUND_ADAPTER_CONFIG",
            "adapter_config_hashes": config_hashes,
            "science_identity_changed": False,
        },
    })
    binding_sha = write(args.out, binding)
    receipt = {
        "status": "FROZEN_ADAPTER_CONTRACT_REPLACEMENT_BINDING",
        "binding_sha256": binding_sha,
        "predecessor_binding_sha256": sha(args.predecessor),
        "source_commit": binding["execution_source_commit"],
        "model_loaded": False,
        "gpu_used": False,
        "sidecar_opened": False,
        "science_identity_changed": False,
    }
    print(json.dumps({"receipt_sha256": write(args.out.with_name(args.out.stem + "_RECEIPT.json"), receipt), **receipt}, sort_keys=True))


if __name__ == "__main__":
    main()
