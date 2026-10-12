"""Freeze the post-audit V7 execution binding.

This creates a CPU-only, non-authorizing package.  A real immutable Git commit
must be supplied; GPU launch still requires a separate fresh one-shot receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/capability_repair_baseline_v1/connected_components_object_extraction_capability_pilot_v1"
SOURCE = BASE / "execution_binding_v4/EXECUTION_BINDING_EXTERNAL_AUTH_V3.json"
SUCCESSOR = BASE / "scientific_successor_v2"


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canon(value): return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canon(value) + b"\n")
    os.replace(temporary, path)


def freeze(source_commit: str, out: Path):
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise ValueError("source_commit must be an exact 40-character Git SHA")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing overwrite: {out}")
    binding = json.loads(SOURCE.read_text())
    composition = SUCCESSOR / "COMPOSITION_SENTINEL_INPUT_PROMPTS.json"
    validation = SUCCESSOR / "SCIENTIFIC_CONSTRUCT_VALIDATION.json"
    composition_rel = str(composition.relative_to(ROOT)).replace("\\", "/")
    validation_rel = str(validation.relative_to(ROOT)).replace("\\", "/")
    nonce = hashlib.sha256((source_commit + sha(SUCCESSOR / "MANIFEST.json") + "EXECUTION_BINDING_V7").encode()).hexdigest()[:32]
    binding.update({
        "schema_version": 7,
        "protocol_id": "CONNECTED_COMPONENTS_OBJECT_EXTRACTION_CAPABILITY_PILOT_V2_EXECUTION_BINDING_V7",
        "status": "CPU_PRODUCTION_PATH_VALIDATED_PENDING_FRESH_ONE_SHOT_GPU_AUTHORIZATION",
        "execution_source_commit": source_commit,
        "dataset_paths": {"COMPOSITION": composition_rel},
        "scientific_construct_validation_path": validation_rel,
        "scientific_construct_validation_sha256": sha(validation),
        "runtime_preflight_path": "scripts/preflight_connected_components_runpod_runtime_v7.py",
    })
    binding["inputs"][composition_rel] = sha(composition)
    binding["inputs"][validation_rel] = sha(validation)
    binding["inputs"][binding["runtime_preflight_path"]] = sha(ROOT / binding["runtime_preflight_path"])
    binding["evaluation"].pop("batch1_gate_records", None)
    binding["evaluation"]["batch1_gate_records_by_checkpoint"] = {
        "V7": 8, "CONTROL_NARROW_STRUCTURE": 12, "TREATMENT_STRUCTURE_DIVERSE": 12,
    }
    binding["output"] = {
        **binding["output"], "nonce": nonce,
        "remote_output_root": f"/workspace/arc2/connected_components_object_extraction_capability_pilot_v2/run_001_{nonce}",
    }
    binding["runtime"] = {
        **binding["runtime"],
        "python_executable": "/root/arc2_r5_f0353c17/.venv-r5/bin/python",
        "pythonpath_prepend": ["/root/arc2_e1_v7_bnb_overlay"],
        "dependency_versions": {"torch": "2.8.0+cu128", "transformers": "4.55.4", "peft": "0.17.1", "bitsandbytes": "0.50.2"},
        "bitsandbytes_overlay_tree_sha256": "a291eaf4ee7c5501c91258046501f166f5bd265e612ea8c2a333727614ba9f17",
    }
    binding["authorization_required"] = {
        "mode": "FRESH_BINDING_SPECIFIC_EXTERNAL_ONE_SHOT_RECEIPT",
        "authorized": False, "permitted_detached_jobs": 1, "retry": False,
        "all_v1_v3_v4_v5_v6_authorizations_nontransferable": True,
    }
    for path_key, sha_key in (("worker_path", "worker_sha256"), ("launcher_path", "launcher_sha256"), ("score_entrypoint_path", "score_entrypoint_sha256")):
        binding[sha_key] = sha(ROOT / binding[path_key])
    out.mkdir(parents=True, exist_ok=False)
    binding_path = out / "EXECUTION_BINDING_V7.json"
    atomic(binding_path, binding)
    atomic(out / "CPU_PACKAGE_STATUS.json", {
        "status": "CPU_ONLY_PACKAGE_FROZEN_NOT_GPU_AUTHORIZED",
        "binding_sha256": sha(binding_path), "execution_source_commit": source_commit,
        "gpu_used": False, "model_loaded": False, "optimizer_constructed": False,
        "training_started": False, "targets_accessed": False,
    })
    return binding


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--source-commit", required=True); parser.add_argument("--out", type=Path, default=BASE / "execution_binding_v7")
    args = parser.parse_args(); print(json.dumps(freeze(args.source_commit, args.out), sort_keys=True))
