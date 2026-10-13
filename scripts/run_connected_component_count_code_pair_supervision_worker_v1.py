"""Bounded production worker for the Director-required count-code-pair pilot.

The CPU integration path deliberately shares the launcher, binding and worker
entrypoints with ``--execute``.  It stops before importing model libraries,
CUDA, the checkpoint, or any sealed target material.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from importlib import metadata
from shutil import disk_usage
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.run_connected_component_count_decision_token_supervision_pilot_v1 import (
    atomic_json,
    fail,
    sha256,
    validate_preclaim,
)


def _binding(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _execution_plan(binding: dict, output_root: Path, require_fresh: bool) -> dict:
    plan = binding.get("execution")
    required = {
        "worker_path", "worker_sha256", "launcher_path", "launcher_sha256",
        "runtime_python", "nonce", "remote_output_root", "combined_cap_seconds",
        "training_cap_seconds", "evaluation_cap_seconds", "retry",
        "terminal_receipt", "authorization_required", "cpu_integration_root",
    }
    if not isinstance(plan, dict) or not required.issubset(plan):
        fail("EXECUTION_PLAN_SCHEMA")
    if plan["worker_path"] != "scripts/run_connected_component_count_code_pair_supervision_worker_v1.py":
        fail("WORKER_PATH")
    if sha256(ROOT / plan["worker_path"]) != plan["worker_sha256"]:
        fail("WORKER_IDENTITY")
    if plan["launcher_path"] != "scripts/run_connected_component_count_decision_token_supervision_pilot_v1.py":
        fail("LAUNCHER_PATH")
    if sha256(ROOT / plan["launcher_path"]) != plan["launcher_sha256"]:
        fail("LAUNCHER_IDENTITY")
    if not isinstance(plan["nonce"], str) or len(plan["nonce"]) < 16:
        fail("NONCE")
    if plan["retry"] is not False or plan["authorization_required"] is not True:
        fail("RETRY_OR_AUTHORIZATION")
    caps = [plan["training_cap_seconds"], plan["evaluation_cap_seconds"], plan["combined_cap_seconds"]]
    if not all(isinstance(value, int) and value > 0 for value in caps):
        fail("CAPS")
    if plan["training_cap_seconds"] + plan["evaluation_cap_seconds"] != plan["combined_cap_seconds"]:
        fail("CAP_RELATION")
    if str(output_root).replace("\\", "/") != plan["remote_output_root"]:
        fail("OUTPUT_ROOT")
    if require_fresh and output_root.exists():
        fail("FRESH_OUTPUT_REQUIRED")
    return plan


def _validate_runtime_environment(binding: dict, plan: dict, output_root: Path) -> dict:
    """Production-only asset/dependency/root checks before an authorization claim.

    The test fixture deliberately switches this off; a final RunPod binding
    enables it and therefore checks the real Python, dependencies, full model
    and tokenizer byte identities, free space and still-fresh output root.
    """
    if not plan.get("enforce_runtime_environment", False):
        return {"enforced": False}
    if Path(sys.executable).resolve() != Path(plan["runtime_python"]).resolve():
        fail("RUNTIME_PYTHON")
    for name, expected in plan.get("dependency_versions", {}).items():
        if metadata.version(name) != expected:
            fail(f"DEPENDENCY:{name}")
    manifest = json.loads((ROOT / binding["v7"]["checkpoint_manifest_path"]).read_text(encoding="utf-8"))
    checked = 0
    for base_key, file_key in (("base_path", "base_files"), ("adapter_path", "adapter_files")):
        root = Path(manifest[base_key])
        for asset in manifest[file_key]:
            path = root / asset["name"]
            if not path.is_file() or path.stat().st_size != asset["bytes"] or sha256(path) != asset["sha256"]:
                fail(f"ASSET:{base_key}:{asset['name']}")
            checked += 1
    free = disk_usage(output_root.parent).free
    if free < int(plan.get("minimum_free_bytes", 0)):
        fail("DISK_FREE")
    if output_root.exists():
        fail("DUPLICATE_OR_NONFRESH_OUTPUT")
    return {"enforced": True, "asset_files_sha256_verified": checked, "free_bytes": free}


def production_preclaim(binding_path: Path, output_root: Path) -> dict:
    preclaim = validate_preclaim(binding_path)
    binding = _binding(binding_path)
    plan = _execution_plan(binding, output_root, require_fresh=True)
    runtime = _validate_runtime_environment(binding, plan, output_root)
    return {**preclaim, "execution_runtime": runtime, "output_root_fresh": True}


def cpu_integration(binding_path: Path, output_root: Path) -> dict:
    """Exercise the executable worker's real CPU-only path and receipt edge."""
    binding = _binding(binding_path)
    plan = binding.get("execution", {})
    if str(output_root).replace("\\", "/") != plan.get("cpu_integration_root"):
        fail("CPU_INTEGRATION_ROOT")
    # Validate the actual execution root too, without creating it.
    preclaim = production_preclaim(binding_path, Path(plan["remote_output_root"]))
    if output_root.exists():
        fail("FRESH_CPU_INTEGRATION_ROOT")
    # Imports remain after preclaim.  Nothing below imports a model or touches CUDA.
    import torch
    from scripts.decision_token_supervision import count_code_pair_label_indices, matched_weights
    from scripts.e04_e_equal_slot_loss import backward_weighted_slot_term
    from training_data.pipeline import task_to_sample

    schedule = json.loads((ROOT / binding["inputs"]["diverse_schedule"]["path"]).read_text(encoding="utf-8"))["episodes"]
    slots = schedule[:4]
    terms, masses, focused = [], [], 0
    torch.manual_seed(17)
    for row in slots:
        sample = task_to_sample(row["task"] | {"source_id": row["task"].get("source_id", f"SERIALIZER_VIEW:{row['canonical_base_id']}")})
        labels = torch.tensor([sample["labels"]], dtype=torch.long)
        logits = torch.randn((1, len(sample["labels"]), 16), dtype=torch.float32, requires_grad=True)
        weights = {index: 1.0 for index, label in enumerate(sample["labels"]) if label != -100}
        if row["role"] == "PRIMARY_CONNECTED_COMPONENT_COUNT":
            indices, _ = count_code_pair_label_indices(row["task"])
            _, weights = matched_weights(sample["labels"], indices, multiplier=2.0)
            focused += len(indices)
        dense = [weights.get(index, 0.0) for index in range(len(sample["labels"]))]
        term, valid_count, mass = backward_weighted_slot_term(logits, labels, dense, expected_slots=4)
        if valid_count != 60 or abs(mass - 60.0) > 1e-12 or logits.grad is None:
            fail("CPU_WORKER_WEIGHTED_LOSS")
        terms.append(float(term)); masses.append(mass)
    aggregate = sum(terms) / len(terms)
    if not (aggregate == aggregate and aggregate not in (float("inf"), float("-inf"))):
        fail("CPU_WORKER_EQUAL_SLOT_AGGREGATE")
    output_root.mkdir(parents=True)
    receipt = {
        "status": "PASS_CPU_INTEGRATION_STOP_BEFORE_MODEL_LOAD",
        "binding_sha256": sha256(binding_path),
        "nonce": plan["nonce"],
        "slots": [row["slot"] for row in slots],
        "per_slot_weight_mass": masses,
        "equal_slot_aggregate": aggregate,
        "focused_count_code_tokens_in_first_update": focused,
        "preclaim_status": preclaim["status"],
        "model_loaded": False,
        "gpu_used": False,
        "cuda_initialized": False,
        "sealed_sidecar_opened": False,
        "authorization_claimed": False,
    }
    atomic_json(output_root / "CPU_INTEGRATION_RECEIPT.json", receipt)
    atomic_json(output_root / plan["terminal_receipt"], {**receipt, "terminal": True})
    return receipt


def _validate_authorization(receipt_path: Path, binding_path: Path, plan: dict) -> None:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("binding_sha256") != sha256(binding_path) or receipt.get("nonce") != plan["nonce"]:
        fail("AUTHORIZATION_BINDING")
    if receipt.get("status") != "CLAIMED" or receipt.get("retry") is not False:
        fail("AUTHORIZATION_STATUS")


def execute(binding_path: Path, output_root: Path, authorization_receipt: Path) -> None:
    """The sole GPU path.  It is intentionally never exercised by CPU tests."""
    started = time.monotonic()
    binding = _binding(binding_path)
    plan = _execution_plan(binding, output_root, require_fresh=True)
    # The same production preclaim runs before the authorization receipt is used.
    production_preclaim(binding_path, output_root)
    _validate_authorization(authorization_receipt, binding_path, plan)
    output_root.mkdir(parents=True)
    try:
        # Deferred imports are the production model/CUDA boundary.
        from scripts import run_connected_components_object_extraction_pilot_v1 as worker
        deadline = started + plan["combined_cap_seconds"]
        atomic_json(output_root / "PREFLIGHT_RECEIPT.json", {
            "status": "PASS_PRECLAIM_BEFORE_MODEL_LOAD", "binding_sha256": sha256(binding_path),
            "nonce": plan["nonce"], "model_loaded": False, "gpu_used": False,
            "sealed_sidecar_opened": False,
        })
        raw = {}
        v7, torch, manifest = worker._load_v7(binding, trainable=False)
        v7_items = worker._datasets(binding)
        raw["V7"] = worker._generate(v7, torch, manifest, v7_items, binding, output_root, deadline, "V7")
        raw["V7_B1"] = worker._generate(v7, torch, manifest, worker._select_b1_items(v7_items, binding, "V7"), binding, output_root, deadline, "V7_B1", force_batch_size=1)
        del v7
        for arm_name, arm in (("CONTROL_NARROW_STRUCTURE", binding["arms"]["control"]), ("TREATMENT_STRUCTURE_DIVERSE", binding["arms"]["treatment"])):
            arm_binding = copy.deepcopy(binding)
            arm_binding["training"] = {**binding["training"], "loss_allocation": arm["loss_allocation"]}
            arm_root = output_root / arm_name
            model, torch, manifest, adapter = worker._train(arm_binding, arm_name, ROOT / arm["schedule_path"], arm_root, deadline)
            items = worker._datasets(arm_binding, arm_name)
            raw[arm_name] = worker._generate(model, torch, manifest, items, arm_binding, output_root, deadline, arm_name)
            raw[arm_name + "_B1"] = worker._generate(model, torch, manifest, worker._select_b1_items(items, arm_binding, arm_name), arm_binding, output_root, deadline, arm_name + "_B1", force_batch_size=1)
            atomic_json(arm_root / "ADAPTER_RECEIPT.json", {"adapter_sha256": adapter, "optimizer_steps": 96})
            del model
        atomic_json(output_root / "RAW_FREEZE_RECEIPT.json", {"status": "COMPLETE_RAW_FROZEN_NO_TARGETS", "raw_sha256": raw, "targets_accessed": False})
        atomic_json(output_root / plan["terminal_receipt"], {"status": "COMPLETE_PENDING_SEPARATE_SCORE", "raw_sha256": raw, "targets_accessed": False})
    except Exception as exc:
        output_root.mkdir(parents=True, exist_ok=True)
        atomic_json(output_root / plan["terminal_receipt"], {"status": "FAILED_OR_PARTIAL", "error": f"{type(exc).__name__}:{exc}", "targets_accessed": False, "wall_seconds": time.monotonic() - started})
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binding", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--cpu-integration", action="store_true")
    parser.add_argument("--preclaim-only", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--authorization-receipt", type=Path)
    args = parser.parse_args()
    if sum((args.cpu_integration, args.preclaim_only, args.execute)) != 1:
        raise SystemExit("choose exactly one of --preclaim-only, --cpu-integration or --execute")
    if args.preclaim_only:
        print(json.dumps(production_preclaim(args.binding, args.output_root), sort_keys=True))
        return
    if args.cpu_integration:
        print(json.dumps(cpu_integration(args.binding, args.output_root), sort_keys=True))
    else:
        if args.authorization_receipt is None:
            raise SystemExit("--authorization-receipt is required for --execute")
        execute(args.binding, args.output_root, args.authorization_receipt)


if __name__ == "__main__":
    main()
