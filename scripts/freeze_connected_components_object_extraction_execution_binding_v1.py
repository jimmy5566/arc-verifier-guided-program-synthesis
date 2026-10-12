"""Freeze the CPU-only, pre-authorization execution binding for E1.

The binding is intentionally not a launcher.  It permits no model import,
target access, remote process, optimizer, or GPU use.  A later explicit user
authorization must bind its SHA256 before the single allowed detached job.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/capability_repair_baseline_v1/connected_components_object_extraction_capability_pilot_v1"
MEASURE = BASE / "measurement_contract_v1"
OUT = BASE / "execution_binding_v1"
SCIENTIFIC_SOURCE_COMMIT = "dd95cb462204b3134729bbebbec3d8701d2bbf71"
V7_MANIFEST = ROOT / "experiments/capability_repair_baseline_v1/CHECKPOINT_MANIFEST_REMOTE_V1.json"
V7_MANIFEST_SHA = "1e124cc4f43530bbbc71103703d404b38df83a6998a3e39d7c1da0676c800549"


class BindingFailure(RuntimeError):
    pass


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def atomic_write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("wb", dir=path.parent, delete=False) as handle:
        handle.write(canonical(value) + b"\n")
        temp = Path(handle.name)
    os.replace(temp, path)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def check_prompt_rows() -> dict:
    ood = read_json(BASE / "OOD_INPUT_PROMPTS.json")
    retention = read_json(BASE / "RETENTION_INPUT_PROMPTS.json")
    composition = read_json(BASE / "COMPOSITION_SENTINEL_INPUT_PROMPTS.json")
    train_fit = read_json(MEASURE / "TRAIN_FIT_INPUT_PROMPTS.json")["records"]
    protected = read_json(MEASURE / "PROTECTED_CAPABILITY_INPUT_PROMPTS.json")["records"]
    all_tasks = [row["task"] for row in ood + retention + composition + train_fit]
    all_tasks += [row["prompt"] for row in protected]
    if any("output" in task["test"][0] for task in all_tasks):
        raise BindingFailure("TARGET_LEAK_IN_GENERATION_INPUT")
    if len(ood) != 48 or len(retention) != 24 or len(composition) != 16 or len(train_fit) != 48 or len(protected) != 16:
        raise BindingFailure("EVALUATION_CARDINALITY")
    # A fixed 32-row crosscheck: V7 sees eight held-out rows; each trained arm
    # sees those eight plus four of its own Train-Fit rows.
    def ids(rows, label, count):
        return [f"{label}:{i}" for i in range(count)]
    holdout = ids(ood, "OOD", 2) + ids(retention, "RETENTION", 1) + ids(composition, "COMPOSITION", 1)
    for condition in ("MARKER_BINDING_CONTROL", "FIXED_TURN_ROTATION_CONTROL"):
        chosen = [r for r in protected if r["condition"] == condition][:2]
        if len(chosen) != 2:
            raise BindingFailure("B1_PROTECTED_COVERAGE")
        holdout += [f"PROTECTED:{r['protected_id']}" for r in chosen]
    if len(holdout) != 8 or len(set(holdout)) != 8:
        raise BindingFailure("B1_HOLDOUT_SELECTION")
    b1 = {"V7": holdout}
    for arm in ("CONTROL_NARROW_STRUCTURE", "TREATMENT_STRUCTURE_DIVERSE"):
        arm_rows = [r for r in train_fit if r["arm"] == arm]
        selected = []
        for family in ("CONNECTED_COMPONENT_COUNT", "OBJECT_EXTRACTION"):
            selected.extend([r for r in arm_rows if r["family"] == family][:2])
        if len(selected) != 4 or {r["family"] for r in selected} != {"CONNECTED_COMPONENT_COUNT", "OBJECT_EXTRACTION"}:
            raise BindingFailure("B1_TRAINFIT_COVERAGE")
        b1[arm] = holdout + [f"TRAIN_FIT:{arm}:{r['source_slot']}" for r in selected]
    return {"b1_subset_by_checkpoint": b1, "primary_evaluations": 360,
            "batch1_repeat_evaluations": sum(map(len, b1.values())), "total_evaluations": 360 + sum(map(len, b1.values()))}


def binding() -> dict:
    main = read_json(BASE / "MANIFEST.json")
    amended = read_json(MEASURE / "AMENDED_MANIFEST.json")
    required = {
        BASE / "CONTROL_SCHEDULE.json": "ec16bb3d7f8824ba8e45e777f0099c405715de8eb2d7de959320c3f4e1c38263",
        BASE / "TREATMENT_SCHEDULE.json": "70addd17375437fe6371d57433281272ecb3bfed716289ff0b5cfd003f4cad10",
        MEASURE / "AMENDED_MANIFEST.json": "426f313a6df8948290031f7a97de096e4fdfc596649b4628efa65ed70fab100d",
        MEASURE / "TRAIN_FIT_INPUT_PROMPTS.json": "708596c8715c5380903ab1af89852c32f05bb9176fb45cfd6c95f33e63dff981",
        MEASURE / "PROTECTED_CAPABILITY_INPUT_PROMPTS.json": "48fe9563a02192aea29b4f0d20d862015c04b51b6742e6cf8c19f85339565247",
        MEASURE / "MEASUREMENT_ANALYSIS_CONTRACT.json": "46b460c290b842e11f131e5324981187ba68b8552b0f13bfc72168155fbb6f7f",
        MEASURE / "ORACLE_AND_SCORER_CONTRACT.json": "f8025a7533159127a6c2852d1971870232ddf429567f6ba3a201a42bcad06c40",
        V7_MANIFEST: V7_MANIFEST_SHA,
    }
    for path, expected in required.items():
        if sha(path) != expected:
            raise BindingFailure("FROZEN_IDENTITY_MISMATCH:" + path.name)
    if main["checkpoint"]["manifest_sha256"] != V7_MANIFEST_SHA or amended["status"] != "PASS_CPU_ONLY_ADDITIVE_MEASUREMENT_CONTRACT":
        raise BindingFailure("MANIFEST_BINDING")
    counts = check_prompt_rows()
    nonce = hashlib.sha256((SCIENTIFIC_SOURCE_COMMIT + required[MEASURE / "AMENDED_MANIFEST.json"] + "E1_EXECUTION_BINDING_V1").encode()).hexdigest()[:32]
    source_hashes = {str(p.relative_to(ROOT)).replace("\\", "/"): sha(p) for p in (
        ROOT / "src/training_data/pipeline.py", ROOT / "scripts/e04_e_equal_slot_loss.py",
        ROOT / "src/inference/arc_native_io.py", ROOT / "scripts/arc2_token_grid_parser.py",
        ROOT / "scripts/freeze_connected_components_object_extraction_capability_pilot_v1.py",
        ROOT / "scripts/amend_connected_components_object_extraction_capability_pilot_v1_measurement_contract.py",
    )}
    return {
        "schema_version": 1, "protocol_id": "CONNECTED_COMPONENTS_OBJECT_EXTRACTION_CAPABILITY_PILOT_V1_EXECUTION_BINDING_V1",
        "status": "CPU_PREFLIGHT_READY_EXPLICIT_USER_GPU_AUTHORIZATION_REQUIRED",
        "scientific_source_commit": SCIENTIFIC_SOURCE_COMMIT,
        "scientific_manifest_sha256": required[MEASURE / "AMENDED_MANIFEST.json"],
        "inputs": {str(p.relative_to(ROOT)).replace("\\", "/"): h for p, h in required.items() if p != V7_MANIFEST},
        "v7": {"checkpoint_manifest_path": str(V7_MANIFEST.relative_to(ROOT)).replace("\\", "/"), "checkpoint_manifest_sha256": V7_MANIFEST_SHA,
               "adapter_model_sha256": "1639bf6665ccfbb99cf547f59fcd581663c95b6bad770ae7846fee90f66be1c3", "adapter_config_sha256": "faa1a3bfb957209ca10ea58afbbf275ce5b51592dd78f289b68f7b655ede2b69"},
        "prospective_trained_arms": [{"arm_id": "CONTROL_NARROW_STRUCTURE", "schedule_sha256": required[BASE / "CONTROL_SCHEDULE.json"]}, {"arm_id": "TREATMENT_STRUCTURE_DIVERSE", "schedule_sha256": required[BASE / "TREATMENT_SCHEDULE.json"]}],
        "training": {"independent_initialization": "V7", "torch_seed": 2000031, "precision": "BF16", "optimizer": "PagedAdamW8bit", "lora": {"rank": 64, "alpha": 32, "dropout": 0.0}, "learning_rate": 5e-5, "updates_per_arm": 96, "slots_per_update": 4, "retry": False, "maximum_detached_jobs": 1},
        "evaluation": {"decoder": "NATIVE_GREEDY", "primary": "parsed_grid_exact_match", "length_bucketing": True, "requested_batch_size": 32, "fallback_ladder": [32, 16, 8, 4, 1], "fallback_cause": "CUDA_OOM_ONLY", **counts,
            "batch1_gate": {"generated_token_canonical_agreement_minimum": 11 / 12, "parsed_grid_agreement_minimum": 11 / 12, "scorer_agreement_minimum": 11 / 12, "aggregate_exact_rate_delta_maximum": 1 / 12, "failure": "INVALID_NOT_INTERPRETABLE"}},
        "target_isolation": {"generation_inputs_only": True, "raw_prediction_freeze_required_before_target_generation": True, "raw_fields": ["generated_token_ids", "parsed_grid", "effective_batch_size", "prompt_sha256"], "forbidden": ["Gold", "dGold", "FINAL_AUDIT", "E05_targets", "V4_targets"]},
        "runtime": {"training_cap_seconds": 7200, "evaluation_cap_seconds": 10800, "combined_cap_seconds": 18000, "actual_elapsed_recording_required": True},
        "output": {"nonce": nonce, "remote_output_root": "/workspace/arc2/connected_components_object_extraction_capability_pilot_v1/run_001_" + nonce, "fresh_output_required": True, "duplicate_guard": "refuse matching nonce, existing output root, terminal receipt, or active job"},
        "source_hashes": source_hashes, "preflight": {"model_loaded": False, "gpu_used": False, "optimizer_constructed": False, "targets_accessed": False},
        "authorization_requirement": "One explicit user authorization must bind this exact binding SHA256, one job, retry=false, and the three hard caps before any GPU launch.",
    }


def freeze(destination: Path = OUT) -> dict:
    package = binding()
    if destination.exists():
        raise BindingFailure("REFUSE_OVERWRITE_EXECUTION_BINDING")
    atomic_write(destination / "EXECUTION_BINDING_PREAUTH_V1.json", package)
    package_sha = sha(destination / "EXECUTION_BINDING_PREAUTH_V1.json")
    atomic_write(destination / "CPU_PREFLIGHT_V1.json", {"status": "PASS_CPU_PREAUTH_IDENTITY_PREFLIGHT", "binding_sha256": package_sha, "model_loaded": False, "gpu_used": False, "optimizer_constructed": False, "targets_accessed": False})
    atomic_write(destination / "EXPLICIT_USER_GPU_AUTHORIZATION_REQUEST.json", {"request": "Authorize exactly one detached GPU job for this binding; no retry.", "binding_sha256": package_sha, "training_cap_seconds": 7200, "evaluation_cap_seconds": 10800, "combined_cap_seconds": 18000, "model_loaded": False, "gpu_used": False, "targets_accessed": False})
    return package


if __name__ == "__main__":
    print(json.dumps(freeze(), sort_keys=True))
