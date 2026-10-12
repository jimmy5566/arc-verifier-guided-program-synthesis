#!/usr/bin/env python3
"""Freeze the single Director-approved all-physical-Batch1 successor.

This freezer does not load a model or deserialize sealed target content.  It
only carries forward immutable input identities and binds the corrected
all-Batch1 teacher-forced runtime contract.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "experiments/capability_repair_baseline_v1/connected_components_object_extraction_capability_pilot_v1"
BASE = PILOT / "connected_component_count_fit_vs_generation_diagnostic_v1"


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
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    previous_binding_path = BASE / "CPU_REAL_INPUTS_RUNTIME_BINDING_V1.json"
    previous_protocol_path = BASE / "SUCCESSOR_CPU_ONLY_FROZEN_PROTOCOL.json"
    inventory_path = BASE / "RUN_002_REUSE_AND_MISSING_ROW_INVENTORY.json"
    cohort_path = BASE / "COHORT_PROMPT_TASKS_V1.json"
    missing_path = BASE / "MISSING_TRAIN_FIT_PROMPT_TASKS_V1.json"
    raw_manifest_path = BASE / "RUN_002_EXACT_RAW_ARTIFACT_MANIFEST_V1.json"
    validation_path = BASE / "SUCCESSOR_CPU_ONLY_VALIDATION.json"
    for path in (previous_binding_path, previous_protocol_path, inventory_path, cohort_path, missing_path, raw_manifest_path, validation_path, args.sidecar):
        if not path.is_file():
            raise RuntimeError(f"MISSING_REQUIRED_INPUT:{path}")

    prior = json.loads(previous_binding_path.read_text(encoding="utf-8"))
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    if inventory.get("reused_immutable_run_002_rows") != 168 or inventory.get("new_free_running_rows_required") != 48:
        raise RuntimeError("REUSE_INVENTORY_IDENTITY_FAILURE")
    if len(inventory.get("inventory", [])) != 216:
        raise RuntimeError("REUSE_INVENTORY_ROW_COUNT_FAILURE")

    source_paths = [
        "scripts/connected_component_fit_generation_runtime_contract.py",
        "scripts/prepare_connected_component_fit_generation_inputs_v1.py",
        "scripts/materialize_connected_component_prompt_token_ids_v1.py",
        "scripts/connected_component_run002_raw_manifest_v1.py",
        "scripts/run_connected_component_fit_generation_diagnostic_v1.py",
        "scripts/score_connected_component_fit_generation_diagnostic_v1.py",
        "scripts/launch_connected_component_fit_generation_diagnostic_v1.py",
        "scripts/test_connected_component_fit_generation_production_v1.py",
        "scripts/freeze_connected_component_fit_generation_all_b1_successor_v1.py",
    ]
    source_hashes = {relative: sha(ROOT / relative) for relative in source_paths}
    commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    protocol = {
        "schema_version": 1,
        "protocol_id": "CONNECTED_COMPONENT_COUNT_FIT_VS_GENERATION_DIAGNOSTIC_V1_ALL_B1_SUCCESSOR",
        "status": "CPU_ONLY_FROZEN_ALL_B1_SUCCESSOR_PENDING_ACTUAL_POD_PREFLIGHT",
        "successor_of": {
            "director_result_review_response": "V7_POSIX_SUCCESSOR_RUN_002_RESULT_REVIEW_REQUIRED_RESPONSE.json",
            "previous_protocol_sha256": sha(previous_protocol_path),
            "previous_binding_sha256": sha(previous_binding_path),
            "run_002_retry": False,
        },
        "immutable_conditions": {
            "checkpoints": prior["checkpoints"],
            "sidecar_sha256": sha(args.sidecar),
            "reuse_inventory_sha256": sha(inventory_path),
            "cohort_prompt_tasks_sha256": sha(cohort_path),
            "missing_train_fit_prompt_tasks_sha256": sha(missing_path),
            "run_002_raw_manifest_sha256": sha(raw_manifest_path),
            "native_greedy_reuse_rows": 168,
            "new_native_greedy_train_fit_rows": 48,
            "structural_ood_regeneration": "FORBIDDEN",
            "optimizer_steps": 0,
            "backward_calls": 0,
            "training": False,
            "retry": False,
            "detached_jobs": 1,
        },
        "teacher_forced": {
            "physical_batch_size": 1,
            "checkpoint_record_forwards": 216,
            "records_per_checkpoint": 72,
            "batch_mixing": "FORBIDDEN",
            "metrics": ["correct_token_rank", "Top1", "Top2", "margin", "token_weighted_NLL", "sequence_normalized_NLL"],
        },
        "free_running": {
            "policy": "EXISTING_VALIDATED_NATIVE_GREEDY",
            "reused_rows": 168,
            "generate_once_missing_train_fit_rows": 48,
            "metrics": ["first_divergence", "independent_semantic_error_taxonomy"],
        },
        "runtime_estimate": {
            "basis": "V7 RUN_002 control timestamps: 120 teacher-forced records completed in 216 seconds including process startup; all-B1 three-checkpoint work has 216 teacher-forced records plus 48 already-enumerated native-Greedy generations.",
            "planning_estimate_seconds": 1800,
            "hard_cap_seconds": 7200,
            "estimate_status": "CONSERVATIVE_PRE_FLIGHT_ESTIMATE_TO_BE_MEASURED_ON_ACTUAL_POD_CPU_ONLY",
        },
        "authorization": {
            "cpu_preparation_authorized": True,
            "gpu_authorization": "FRESH_EXPLICIT_ONE_SHOT_REQUIRED_AFTER_ACTUAL_POD_CPU_PREFLIGHT_PASS",
            "consumed_run_002_authorization_reusable": False,
        },
        "forbidden": ["target release", "new cohort", "checkpoint substitution", "decoder change", "training", "optimizer", "backward", "batch greater than one for teacher-forced records", "RUN_002 retry"],
    }
    protocol_path = args.out_dir / "ALL_B1_SUCCESSOR_PROTOCOL.json"
    protocol_sha = write(protocol_path, protocol)
    execution = dict(prior["execution"])
    execution.pop("teacher_forced_b1_cohort_rows", None)
    execution.update({
        "requested_batch_size": 1,
        "teacher_forced_physical_batch_size": 1,
        "teacher_forced_forwards": 216,
        "teacher_forced_batch_mixing": "FORBIDDEN",
        "teacher_forced_batch_comparison": "NOT_APPLICABLE_ALL_PRIMARY_RECORDS_PHYSICAL_BATCH1",
        "free_running_requested_batch_size": 32,
        "free_running_fallback_ladder": [32, 16, 8, 4, 1],
        "new_free_running_rows": 48,
        "reused_rows": 168,
        "planning_estimate_seconds": 1800,
        "hard_cap_seconds": 7200,
        "retry": False,
        "maximum_detached_jobs": 1,
    })
    binding = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "status": "CPU_TESTED_ALL_B1_SUCCESSOR_ACTUAL_POD_PREFLIGHT_AND_FRESH_OWNER_AUTHORIZATION_REQUIRED",
        "execution_source_commit": commit,
        "protocol_sha256": protocol_sha,
        "source_hashes": source_hashes,
        "checkpoints": prior["checkpoints"],
        "sidecar_sha256": sha(args.sidecar),
        "reuse_inventory_sha256": sha(inventory_path),
        "cpu_validation_sha256": sha(validation_path),
        "runnable": prior["runnable"],
        "execution": execution,
        "authorization": {"owner_one_shot_required": True, "not_authorized_by_this_binding": True},
    }
    binding_path = args.out_dir / "ALL_B1_SUCCESSOR_EXECUTION_BINDING.json"
    binding_sha = write(binding_path, binding)
    receipt = {
        "status": "PASS_CPU_ONLY_ALL_B1_SUCCESSOR_FREEZE",
        "protocol_sha256": protocol_sha,
        "binding_sha256": binding_sha,
        "source_commit": commit,
        "teacher_forced_physical_batch_size": 1,
        "teacher_forced_forwards": 216,
        "reused_native_greedy_rows": 168,
        "new_native_greedy_rows": 48,
        "gpu_used": False,
        "model_loaded": False,
        "sealed_target_content_deserialized": False,
    }
    print(json.dumps({"receipt_sha256": write(args.out_dir / "ALL_B1_SUCCESSOR_FREEZE_RECEIPT.json", receipt), **receipt}, sort_keys=True))


if __name__ == "__main__":
    main()
