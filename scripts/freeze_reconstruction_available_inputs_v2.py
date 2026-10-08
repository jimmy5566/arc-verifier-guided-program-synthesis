#!/usr/bin/env python3
"""Freeze a distinct, non-continuation reconstruction condition.

Inputs are copied as independently named evidence from historical artifacts,
never by referring to the terminal V1 protocol, its gates, or its paths.
"""
from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "experiments" / "reconstruction_from_available_frozen_inputs_v2"
OLD = ROOT / "experiments" / "foundation_v2_reconstruction_and_targeted_repair_v1"
SOURCE_ROOT = "/root/arc-runtime-3090-gpu-benchmark-v1/arc2"
PROTOCOL = "RECONSTRUCTION_FROM_AVAILABLE_FROZEN_INPUTS_V2"
ROUND = "RECONSTRUCTED_AVAILABLE_INPUTS_V2_001"
RUN_ROOT = "/workspace/arc2/active_runs/reconstructed_available_inputs_v2_001"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return sha(path)


def remote(relative: str) -> str:
    return f"{SOURCE_ROOT}/{relative.replace('\\\\', '/')}"


def main() -> int:
    recipe = load(OLD / "RECONSTRUCTED_V2_RECIPE_FREEZE.json")
    assets = load(OLD / "RECONSTRUCTION_RUNTIME_ASSET_RECEIPT_V1.json")
    if assets["pool_bindings"]["POOL_REPLAY_V2_1"]["sha256"] != recipe["data"]["replay_shard_sha256"]:
        raise RuntimeError("REPLAY_EVIDENCE_IDENTITY_CONFLICT")
    nonce_path = DEST / "LAUNCH_NONCE.txt"
    if nonce_path.exists():
        nonce = nonce_path.read_text(encoding="ascii").strip()
    else:
        nonce = str(uuid.uuid4())
        nonce_path.parent.mkdir(parents=True, exist_ok=True)
        nonce_path.write_text(nonce + "\n", encoding="ascii", newline="\n")

    base_files = []
    for name, item in recipe["base"]["base_files"].items():
        base_files.append({"path": f"/workspace/arc2/models/qwen3_4b_grids15_sft139/{name}", "sha256": item["sha256"], "bytes": item["bytes"], "identity_role": "BASE_OR_TOKENIZER"})
    frozen_sources = [
        "artifacts/capability_pilot_2m_v1/TRAINING_CONFIG.json",
        "artifacts/capability_pilot_2m_v1/TRAINING_SCHEDULE_2M.json",
        "artifacts/capability_pilot_2m_v1/TRAINING_DISTRIBUTION_AUDIT.json",
        "artifacts/capability_pilot_2m_v1/PRETRAIN_DATA_POLICY_AUDIT.json",
        "artifacts/capability_pilot_2m_v1/NOVEL_VALIDATION_SENTINEL.json",
        "artifacts/capability_pilot_2m_v1/REPLAY_RETENTION_SENTINEL.json",
        "artifacts/foundation_v2_capability_diagnostic_v1/MODEL_IDENTITY_AUDIT.json",
    ]
    frozen_files = [{"path": remote(p), "sha256": sha(ROOT / p), "identity_role": "FROZEN_RECIPE_EVIDENCE"} for p in frozen_sources]
    exec_sources = [
        "scripts/run_reconstruction_available_inputs_v2.py",
        "scripts/consume_reconstruction_available_inputs_v2_authorization.py",
        "scripts/reconstruction_available_inputs_v2_authorization.py",
        "scripts/preflight_reconstruction_available_inputs_v2.py",
        "scripts/write_reconstruction_terminal_receipt_v1.py",
        "scripts/record_reconstruction_gpu_time_v6.py",
        "scripts/run_capability_pilot_2m_v1.py",
    ]
    executable_files = [{"path": p, "sha256": sha(ROOT / p)} for p in exec_sources]
    novel_train_manifest = "artifacts/novel_training_data_v1_1/NOVEL_TRAIN_SHARD_MANIFEST.json"
    novel_val_manifest = "artifacts/novel_training_data_v1_1/NOVEL_VAL_SHARD_MANIFEST.json"
    train_manifest = load(ROOT / novel_train_manifest)
    val_manifest = load(ROOT / novel_val_manifest)
    manifest_hash = lambda value: hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    dataset_contracts = [
        {"pool": "POOL_NOVEL_V1_1", "root": "/workspace/arc2/reconstruction-data_v1/POOL_NOVEL_V1_1/train/novel", "manifest": remote(novel_train_manifest), "manifest_key": "shards", "manifest_content_sha256": manifest_hash(train_manifest), "artifact_repository": assets["pool_bindings"]["POOL_NOVEL_V1_1"]["artifact_repository"], "artifact_commit": assets["pool_bindings"]["POOL_NOVEL_V1_1"]["artifact_commit"]},
        {"pool": "POOL_NOVEL_V1_1_VALIDATION_ONLY", "root": "/workspace/arc2/reconstruction-data_v1/POOL_NOVEL_V1_1/validation/novel", "manifest": remote(novel_val_manifest), "manifest_key": "shards", "manifest_content_sha256": manifest_hash(val_manifest), "artifact_repository": assets["pool_bindings"]["POOL_NOVEL_V1_1"]["artifact_repository"], "artifact_commit": assets["pool_bindings"]["POOL_NOVEL_V1_1"]["artifact_commit"]},
    ]
    replay = assets["pool_bindings"]["POOL_REPLAY_V2_1"]
    replay_contract = {"pool": "POOL_REPLAY_V2_1", "path": replay["remote_path"], "sha256": replay["sha256"], "bytes": replay["bytes"], "row_count": replay["rows"], "artifact_repository": replay["artifact_repository"], "artifact_commit": replay["artifact_commit"], "logical_training_shard": "train/replay/replay-00000.parquet"}
    fresh_paths = {
        "run_root": RUN_ROOT,
        "freeze": f"{RUN_ROOT}/freeze",
        "runtime": f"{RUN_ROOT}/runtime",
        "checkpoints": f"{RUN_ROOT}/checkpoints",
        "preflight": f"{RUN_ROOT}/preflight/FINAL_PREFLIGHT.json",
        "ledger": f"{RUN_ROOT}/ledger/ARC2_CUMULATIVE_NEW_GPU_TRAINING_LEDGER_V7.jsonl",
        "ledger_snapshot": f"{RUN_ROOT}/ledger/ARC2_CUMULATIVE_NEW_GPU_TRAINING_LEDGER_V7_SNAPSHOT.json",
        "accounting_state": f"{RUN_ROOT}/runtime/OPTIMIZER_ACCOUNTING_STATE.json",
        "nonce_consumption": f"{RUN_ROOT}/authorization/nonce_consumption",
        "terminal_receipt": f"/workspace/arc2/orchestration/rounds/ROUND_{ROUND}/{ROUND}_TERMINAL_RECEIPT.json",
    }
    binding = {
        "schema_version": 1,
        "status": "FROZEN_PENDING_EXACT_DIRECTOR_AUTHORIZATION_AND_FINAL_PREFLIGHT",
        "protocol_id": PROTOCOL,
        "round_id": ROUND,
        "scientific_condition_id": "RECONSTRUCTION_FROM_AVAILABLE_FROZEN_INPUTS_V2_CONDITION_001",
        "model_identity": "RECONSTRUCTED_FROM_AVAILABLE_FROZEN_INPUTS_V2_001",
        "identity_claim": "NEW_ADAPTER_FROM_INDEPENDENTLY_FROZEN_AVAILABLE_INPUTS",
        "not_claimed": ["ORIGINAL_FOUNDATION_V2_ADAPTER_RECOVERED", "CONTINUITY_WITH_CLOSED_FOUNDATION_V2_RECONSTRUCTION_AND_TARGETED_REPAIR_V1", "BITWISE_ADAPTER_REPRODUCTION"],
        "source_provenance": {"checked_out_root": SOURCE_ROOT, "executable_files": executable_files, "final_launch_commit_rule": "A later qualifying directive must bind the exact source commit; the final preflight verifies it after RunPod checkout."},
        "required_files": base_files + frozen_files,
        "dataset_contracts": dataset_contracts,
        "replay_contract": replay_contract,
        "recipe": {"source_recipe_sha256": sha(OLD / "RECONSTRUCTED_V2_RECIPE_FREEZE.json"), "recipe_values": recipe["adaptation"], "sampling_and_order": recipe["sampling_and_order"], "historical_reference_only": recipe["historical_execution_reference"]},
        "runtime_contract": {
            **recipe["historical_execution_reference"]["environment"],
            "interpreter": "/root/arc-runtime-3090-gpu-benchmark-v1/env/3090-ampere-env-v2/bin/python",
            "base_model_path": "/workspace/arc2/models/qwen3_4b_grids15_sft139",
            "precision": "BF16",
            "quantization": "NONE",
            "base_weights_frozen": True,
        },
        "budget_contract": {"cap_seconds": 28800, "reservation_seconds": 7200, "scientific_seconds_before_launch": 0, "charge_rule": "Only worker-declared optimizer intervals are scientific GPU time; pre-optimizer and wrapper time are separate."},
        "fresh_paths": fresh_paths,
        "scientific_boundaries": {"forbidden_path_terms": ["eval60", "gold", "final_audit"], "final_audit_accessed": False, "targeted_repair_data_used": False, "no_model_load_or_optimizer_before_authorization": True},
        "worker_command": ["/root/arc-runtime-3090-gpu-benchmark-v1/env/3090-ampere-env-v2/bin/python", remote("scripts/run_capability_pilot_2m_v1.py"), "--mode", "train", "--model-path", "/workspace/arc2/models/qwen3_4b_grids15_sft139", "--novel-train-root", dataset_contracts[0]["root"], "--novel-validation-root", dataset_contracts[1]["root"], "--replay-shard", replay_contract["path"], "--freeze", fresh_paths["freeze"], "--runtime", fresh_paths["runtime"], "--checkpoints", fresh_paths["checkpoints"]],
    }
    binding_sha = write(DEST / "RECONSTRUCTION_AVAILABLE_INPUTS_V2_LAUNCH_BINDING.json", binding)
    contract = {
        "schema_version": 1,
        "status": "PENDING_NEW_SUBPROTOCOL_AUTHORIZATION",
        "protocol_id": PROTOCOL,
        "round_id": ROUND,
        "round_class": "SCIENTIFIC",
        "scientific_condition_id": binding["scientific_condition_id"],
        "new_model_identity": binding["model_identity"],
        "launch_nonce": nonce,
        "launch_binding_path": "RECONSTRUCTION_AVAILABLE_INPUTS_V2_LAUNCH_BINDING.json",
        "launch_binding_sha256": binding_sha,
        "fresh_paths": fresh_paths,
        "authorization_required": {"decision": ["CONTINUE", "CONTINUE_WITH_WARNING"], "scientific_training_authorized": True, "future_directive_must_bind": ["protocol_id", "round_id", "contract_sha256", "launch_nonce", "exact_source_commit"]},
        "closed_predecessors": [{"protocol_id": "FOUNDATION_V2_RECONSTRUCTION_AND_TARGETED_REPAIR_V1", "round_id": "RECONSTRUCTED_FOUNDATION_V2_V1_001", "status": "TERMINAL_NON_DISPATCHABLE"}, {"round_id": "RECONSTRUCTED_FOUNDATION_V2_V1_002", "nonce": "3a52d466-e8ec-4d51-96d2-505853df2c2c", "status": "REJECTED_NON_CONSUMABLE"}],
        "scientific_training_started": False,
        "final_audit_accessed": False,
    }
    contract_sha = write(DEST / f"{ROUND}_CONTRACT.json", contract)
    gate = {"schema_version": 1, "protocol_id": PROTOCOL, "round_id": ROUND, "round_class": "SCIENTIFIC", "contract_sha256": contract_sha, "launch_nonce": nonce, "status": "PENDING_FUTURE_DIRECTOR_AUTHORIZATION", "GPU_GATE_READY": False, "AUTO_SCIENTIFIC_EXECUTION_AUTHORIZED": False, "authorization": {"directive_sha256": None, "protocol_id": PROTOCOL, "contract_sha256": contract_sha, "launch_nonce": nonce, "consumption": "ATOMIC_EXCLUSIVE_CREATE_ONLY_AFTER_QUALIFYING_DIRECTIVE"}, "fail_closed_reasons": ["No qualifying directive has bound this new protocol, round, contract SHA, nonce, and exact source commit.", "Final read-only preflight has not yet run on the exact final launch commit."]}
    gate_sha = write(DEST / "SCIENTIFIC_EXECUTION_GATE_AVAILABLE_INPUTS_V2.json", gate)
    provenance = {"schema_version": 1, "protocol_id": PROTOCOL, "condition_id": binding["scientific_condition_id"], "status": "AVAILABLE_INPUTS_FROZEN_PENDING_FINAL_PREFLIGHT", "replay": replay_contract, "novel": dataset_contracts, "base_and_tokenizer": {"path": "/workspace/arc2/models/qwen3_4b_grids15_sft139", "files": base_files}, "historical_recipe_evidence": {"path": "../foundation_v2_reconstruction_and_targeted_repair_v1/RECONSTRUCTED_V2_RECIPE_FREEZE.json", "sha256": binding["recipe"]["source_recipe_sha256"]}, "stage_compatibility_claim": "NOT_ASSUMED_FROM_CLOSED_PROTOCOL: this new condition independently requires exact byte preflight of all listed inputs before authorization consumption.", "forbidden_reuse": contract["closed_predecessors"]}
    provenance_sha = write(DEST / "AVAILABLE_TRAINING_INPUT_PROVENANCE.json", provenance)
    protocol = {"schema_version": 1, "protocol_id": PROTOCOL, "scientific_condition_id": binding["scientific_condition_id"], "model_identity": binding["model_identity"], "status": "FROZEN_PENDING_DIRECTOR_AUTHORIZATION", "condition_statement": "A new reconstruction condition using independently frozen, currently available byte-identified inputs; it is not a continuation of the terminal V1 reconstruction protocol.", "binding_sha256": binding_sha, "contract_sha256": contract_sha, "gate_sha256": gate_sha, "provenance_sha256": provenance_sha, "budget": binding["budget_contract"], "scientific_training_started": False}
    write(DEST / "RECONSTRUCTION_FROM_AVAILABLE_FROZEN_INPUTS_V2_PROTOCOL.json", protocol)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
