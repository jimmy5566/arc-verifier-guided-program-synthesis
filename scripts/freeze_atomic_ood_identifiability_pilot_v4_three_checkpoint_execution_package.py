"""Freeze the CPU-only execution package authorized by the V4 prefreeze review."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
V4 = ROOT / "experiments" / "capability_repair_baseline_v1" / "atomic_ood_identifiability_pilot_v4"
OUT = ROOT / "experiments" / "capability_repair_baseline_v1" / "atomic_ood_identifiability_pilot_v4_three_checkpoint_native_greedy_no_update"
WORKER = ROOT / "scripts" / "run_atomic_ood_identifiability_pilot_v4_three_checkpoint_native_greedy.py"
PROTOCOL = "ATOMIC_OOD_IDENTIFIABILITY_PILOT_V4_THREE_CHECKPOINT_NATIVE_GREEDY_NO_UPDATE"


class PackageFailure(RuntimeError):
    pass


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        handle.write(json.dumps(value, sort_keys=True, indent=2).encode("utf-8") + b"\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def b1_subset(prompts: list[dict[str, Any]]) -> list[str]:
    by_id = {row["episode_id"]: row for row in prompts}
    cells = ("REFERENCE", "OOD_GEOMETRY", "OOD_COLOR_MAPPING", "OOD_GRID_SIZE", "OOD_LAYOUT", "OOD_MARKER_ENCODING")
    chosen = []
    for index, cell in enumerate(cells):
        selected = ((index, index % 4), (7 - index, (index + 2) % 4))
        for unit, turn in selected:
            episode_id = f"AOODV2-{cell}-U{unit:02d}-T{turn}"
            if episode_id not in by_id:
                raise PackageFailure("B1_SELECTION_MISSING:" + episode_id)
            chosen.append(episode_id)
    if len(chosen) != 12 or len(set(chosen)) != 12:
        raise PackageFailure("B1_SELECTION_CARDINALITY")
    return chosen


def package() -> dict[str, Any]:
    manifest = json.loads((V4 / "MANIFEST.json").read_text(encoding="utf-8"))
    expected = {
        "INPUT_PROMPTS.jsonl": "9a87a336bff6e3fec0a9726b5212e0ae4fe21666c4839dc06f306229dafccec3",
        "SEALED_TARGET_SIDECAR.jsonl": "62217a9963a8ba62de46d9cc865426e82b8b6fb990eb4afbeda2734b59610903",
        "MANIFEST.json": "34fd28f9057c57961bee0e8c4cc55e77e514cc87335736a61a406e39adbe6cc1",
    }
    for name, expected_sha in expected.items():
        if sha(V4 / name) != expected_sha:
            raise PackageFailure("V4_INPUT_HASH_MISMATCH:" + name)
    if manifest.get("source_sha256") != "523b904ab3b712095009d0be3e29151aa8d8a563ac663a15ff3789a0192926bd":
        raise PackageFailure("V4_SOURCE_SHA_MISMATCH")
    prompts = rows(V4 / "INPUT_PROMPTS.jsonl")
    if len(prompts) != 192 or any("target" in row["task"]["test"][0] for row in prompts):
        raise PackageFailure("V4_INPUT_PROMPT_BOUNDARY")
    storage = json.loads((ROOT / "experiments/capability_repair_baseline_v1/e05_v3_semantically_identifiable_demonstrations_and_adapter_storage_recovery/E05_V3_ADAPTER_STORAGE_IDENTITY.json").read_text(encoding="utf-8"))
    assets = {item["filename"]: item["expected_source_sha256"] for item in storage["assets"]}
    native_identity = ROOT / "experiments/capability_repair_baseline_v1/e04_orientation_fixed_demonstration_baseline_v3/E04_V3_RUNTIME_NATIVE_CONFIG_IDENTITY_V1.json"
    base_manifest = ROOT / "experiments/capability_repair_baseline_v1/CHECKPOINT_MANIFEST_REMOTE_V1.json"
    nonce = hashlib.sha256((expected["MANIFEST.json"] + sha(WORKER) + "V4_THREE_ARM_EXECUTION").encode("ascii")).hexdigest()[:32]
    selected = b1_subset(prompts)
    return {
        "schema_version": 1,
        "protocol_id": PROTOCOL,
        "status": "CPU_ONLY_FROZEN_AWAITING_DIRECTOR_EXECUTION_AUTHORIZATION",
        "v4_source": {"manifest_sha256": expected["MANIFEST.json"], "sha256": manifest["source_sha256"], "prompt_sha256": expected["INPUT_PROMPTS.jsonl"], "sealed_target_sidecar_sha256": expected["SEALED_TARGET_SIDECAR.jsonl"]},
        "inputs": {"experiments/capability_repair_baseline_v1/atomic_ood_identifiability_pilot_v4/INPUT_PROMPTS.jsonl": expected["INPUT_PROMPTS.jsonl"], "experiments/capability_repair_baseline_v1/atomic_ood_identifiability_pilot_v4/SEALED_TARGET_SIDECAR.jsonl": expected["SEALED_TARGET_SIDECAR.jsonl"], "experiments/capability_repair_baseline_v1/atomic_ood_identifiability_pilot_v4/MANIFEST.json": expected["MANIFEST.json"]},
        "three_checkpoints": [
            {"arm_id": "V7", "role": "reference", "adapter_model_sha256": "1639bf6665ccfbb99cf547f59fcd581663c95b6bad770ae7846fee90f66be1c3", "adapter_config_sha256": "faa1a3bfb957209ca10ea58afbbf275ce5b51592dd78f289b68f7b655ede2b69", "base_manifest_path": "experiments/capability_repair_baseline_v1/CHECKPOINT_MANIFEST_REMOTE_V1.json", "base_manifest_sha256": sha(base_manifest)},
            {"arm_id": "E04_E_CONTROL", "role": "control", "adapter_model_sha256": assets["e04_e_run003/control_equal_slot/adapter_model.safetensors"], "adapter_config_sha256": assets["e04_e_run003/control_equal_slot/adapter_config.json"], "repository": storage["repository"], "repository_path": "e04_e_run003/control_equal_slot/adapter_model.safetensors"},
            {"arm_id": "E04_E_TREATMENT", "role": "treatment", "adapter_model_sha256": assets["e04_e_run003/treatment_equal_slot/adapter_model.safetensors"], "adapter_config_sha256": assets["e04_e_run003/treatment_equal_slot/adapter_config.json"], "repository": storage["repository"], "repository_path": "e04_e_run003/treatment_equal_slot/adapter_model.safetensors"},
        ],
        "native_runtime": {"model_family": "qwen3_4b_grids15_sft139", "config_identity_path": "experiments/capability_repair_baseline_v1/e04_orientation_fixed_demonstration_baseline_v3/E04_V3_RUNTIME_NATIVE_CONFIG_IDENTITY_V1.json", "config_identity_sha256": sha(native_identity), "tokenizer_loader": "checkpoint_native_tokenizer", "decoder_provider": "NVARCNativeProvider"},
        "decoder": {"name": "native_greedy", "sampling": False, "num_beams": 1, "max_new_tokens": 128},
        "batching": {"primary_requested_batch_size": 32, "fallback_ladder": [32, 16, 8, 4, 1], "fallback_cause": "CUDA_OOM_ONLY", "length_bucketing": True, "fixed_b1_subset": {"selection": "two deterministic episodes per cell; covers all 8 units and all 4 turns", "episode_ids": selected, "sha256": hashlib.sha256(canonical(selected)).hexdigest()}, "light_numerical_drift_tolerance": {"max_parse_valid_rate_delta": 1 / 12, "max_exact_grid_rate_delta": 1 / 12}, "gate": "parser, parsed-grid and post-freeze exact-grid scorer must agree for every fixed B1 episode; a larger rate delta or material disagreement is INVALID_NOT_INTERPRETABLE"},
        "target_isolation": {"generation_inputs": "INPUT_PROMPTS.jsonl only", "sealed_target_sidecar": "SEALED_TARGET_SIDECAR.jsonl", "sealed_target_sidecar_may_open_before_raw_freeze": False, "release_condition": "all three 192-row target-blind raw files and generation terminal receipt hash-verified before CPU scorer opens sidecar"},
        "execution_output": {"nonce": nonce, "remote_output_root": "/workspace/arc2/atomic_ood_identifiability_pilot_v4_three_checkpoint_native_greedy_no_update/run_001_" + nonce, "fresh_output_required": True, "duplicate_guard": "refuse existing output root, existing terminal receipt, or matching active nonce"},
        "terminal_receipt": {"atomic": True, "required": ["status", "runtime_binding_sha256", "raw_evidence", "target_sidecar_accessed", "optimizer_steps", "parameter_updates", "backward_calls", "final_audit_opened", "wall_seconds"], "partial_evidence_preserved": True},
        "runtime_estimate": {"historical_receipt": "unified_native_model_capability_baseline_v1/unified_native_model_capability_baseline_v3_batch16/run_001_5e46a8ba7794cbec09d838c84d26a32a/TERMINAL_RECEIPT.json", "historical_rows": 360, "historical_elapsed_seconds": 2402.376140123233, "planned_primary_rows": 576, "planned_b1_repeat_rows": 36, "estimated_seconds_linear": 4084.0394382094957, "safety_multiplier": 1.75, "active_hard_cap_seconds": 7200, "estimate_scope": "runtime estimate only; no outcome inference"},
        "forbidden": ["NO_TRAINING", "NO_OPTIMIZER", "NO_PARAMETER_UPDATES", "NO_BACKWARD", "NO_GRADIENT", "NO_TTT", "NO_AUGMENTATION", "NO_DFS", "NO_BEAM", "NO_GOLD", "NO_DGOLD", "NO_FINAL_AUDIT"],
        "entrypoint_sha256": sha(WORKER),
        "package_fingerprint_scope": "The subsequent Director authorization must bind this package SHA256, exact Git checkout SHA, worker SHA256, nonce, output root and one-shot cap; it must not alter frozen scientific assets.",
    }


def freeze(destination: Path = OUT) -> dict[str, Any]:
    if destination.exists():
        raise PackageFailure("REFUSE_OVERWRITE_FROZEN_V4_EXECUTION_PACKAGE")
    doc = package()
    write(destination / "EXECUTION_PACKAGE.json", doc)
    write(destination / "B1_SENSITIVITY_SUBSET.json", doc["batching"]["fixed_b1_subset"])
    brief = {"brief_type": "DIRECTOR_EXECUTION_AUTHORIZATION_REQUEST", "protocol_id": PROTOCOL, "request": "Authorize exactly one detached no-update, target-blind GPU native-Greedy collection only if all runtime identity, clean checkout, fresh-output, duplicate and cap gates pass. Do not authorize training, optimizer, backward, Gold/dGold/FINAL_AUDIT, target access during generation, or a rerun.", "package_sha256": sha(destination / "EXECUTION_PACKAGE.json"), "v4_manifest_sha256": doc["v4_source"]["manifest_sha256"], "arms": [arm["arm_id"] for arm in doc["three_checkpoints"]], "planned_primary_rows": doc["runtime_estimate"]["planned_primary_rows"], "planned_b1_repeat_rows": doc["runtime_estimate"]["planned_b1_repeat_rows"], "active_hard_cap_seconds": doc["runtime_estimate"]["active_hard_cap_seconds"], "model_loaded": False, "gpu_used": False, "target_sidecar_accessed": False, "training": False}
    write(destination / "EXECUTION_AUTHORIZATION_BRIEF.json", brief)
    return doc


if __name__ == "__main__":
    print(json.dumps(freeze(), sort_keys=True))
