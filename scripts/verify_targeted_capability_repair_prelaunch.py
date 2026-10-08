#!/usr/bin/env python3
"""Freeze a read-only, fail-closed Round-1 launch identity record.

This utility never loads a model, starts a process on a GPU, or creates the
planned output directory.  It verifies the frozen local package every time and
requires explicit mounted paths before it can report ``PASS``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "targeted_capability_repair_v1"
PROTOCOL = EXPERIMENT / "protocol"
DATA = EXPERIMENT / "data"
OUT = EXPERIMENT / "prelaunch" / "PRELAUNCH_IDENTITY_RECORD_V1.json"
EXPECTED_RECIPE = {
    "precision": "BF16", "base_weights": "FROZEN", "lora_rank": 64,
    "lora_alpha": 32, "lora_dropout": 0.0, "optimizer_family": "PagedAdamW8bit",
    "quantization": "NONE", "context": 8704,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def relative_hashes(paths: list[Path]) -> dict[str, str]:
    return {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path) for path in sorted(paths)}


def base_identity(base_dir: Path) -> dict[str, Any]:
    # A model directory must include an independently hashable configuration
    # and index/weight descriptor.  Merely naming a directory is not identity.
    config = base_dir / "config.json"
    descriptors = [
        base_dir / "model.safetensors.index.json",
        base_dir / "pytorch_model.bin.index.json",
        base_dir / "model.safetensors",
        base_dir / "pytorch_model.bin",
    ]
    descriptor = next((candidate for candidate in descriptors if candidate.is_file()), None)
    if not config.is_file() or descriptor is None:
        return {"pass": False, "reason": "BASE_CONFIG_OR_WEIGHT_DESCRIPTOR_MISSING", "path": str(base_dir)}
    return {
        "pass": True,
        "path": str(base_dir),
        "config_sha256": sha256(config),
        "weight_descriptor": str(descriptor),
        "weight_descriptor_sha256": sha256(descriptor),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mounted-adapter", type=Path)
    parser.add_argument("--mounted-base-dir", type=Path)
    parser.add_argument("--resolved-command-file", type=Path)
    parser.add_argument("--resolved-config-file", type=Path)
    parser.add_argument("--output-run-dir")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    start = read_json(ROOT / "artifacts" / "foundation_round2_data_v1" / "FOUNDATION_V2_START_IDENTITY.json")
    protocol = read_json(PROTOCOL / "TARGETED_CAPABILITY_REPAIR_V1_PROTOCOL.json")
    data_policy = read_json(PROTOCOL / "TARGET_DATA_POLICY_V1.json")
    package_manifest = read_json(EXPERIMENT / "PROTOCOL_PACKAGE_MANIFEST_V2.json")
    recipe = protocol["training_recipe"]
    local_files = [
        DATA / "TRAIN.jsonl", DATA / "TARGET_DEV.jsonl", DATA / "RETENTION_SENTINEL.jsonl",
        DATA / "FINAL_AUDIT_SEALED.jsonl", *PROTOCOL.glob("*.json"), *PROTOCOL.glob("*.csv"), *PROTOCOL.glob("*.md"),
    ]
    local_hashes = relative_hashes(local_files)
    manifest_actual = {
        key: sha256(ROOT / key)
        for key in package_manifest["package_hashes"]
        if (ROOT / key).is_file()
    }
    checks: dict[str, Any] = {
        "foundation_start_frozen": start.get("status") == "FROZEN",
        "recipe_exact": recipe == {**EXPECTED_RECIPE, "base_weights": "frozen"},
        "local_package_hashes": local_hashes,
        "package_manifest_matches": manifest_actual == package_manifest["package_hashes"],
        "scientific_boundaries_preserved": data_policy["checks"] == {
            "episode_ids_disjoint": True, "exact_grid_identities_disjoint": True,
            "eval60_gold_used": False, "diagnostic_episodes_used_as_training": False,
            "final_audit_model_accessed": False, "programmatic_labels_only": True,
        },
    }
    # The user-facing protocol uses lower case frozen while the Director's
    # recipe contract uses upper case.  Normalize only for this comparison.
    normalized_recipe = {**recipe, "base_weights": str(recipe.get("base_weights", "")).upper()}
    checks["recipe_exact"] = normalized_recipe == EXPECTED_RECIPE

    adapter: dict[str, Any] = {"pass": False, "reason": "MOUNTED_ADAPTER_PATH_NOT_SUPPLIED"}
    if args.mounted_adapter:
        if args.mounted_adapter.is_file():
            actual = sha256(args.mounted_adapter)
            adapter = {
                "pass": actual == start["adapter_sha256"], "path": str(args.mounted_adapter),
                "sha256": actual, "expected_sha256": start["adapter_sha256"],
            }
        else:
            adapter = {"pass": False, "reason": "MOUNTED_ADAPTER_NOT_FOUND", "path": str(args.mounted_adapter)}
    base: dict[str, Any] = {"pass": False, "reason": "MOUNTED_BASE_PATH_NOT_SUPPLIED"}
    if args.mounted_base_dir:
        base = base_identity(args.mounted_base_dir)
        base["expected_base_label"] = start["base"]
    bindings = {
        "resolved_command": str(args.resolved_command_file) if args.resolved_command_file and args.resolved_command_file.is_file() else None,
        "resolved_command_sha256": sha256(args.resolved_command_file) if args.resolved_command_file and args.resolved_command_file.is_file() else None,
        "resolved_config": str(args.resolved_config_file) if args.resolved_config_file and args.resolved_config_file.is_file() else None,
        "resolved_config_sha256": sha256(args.resolved_config_file) if args.resolved_config_file and args.resolved_config_file.is_file() else None,
        "output_run_dir": args.output_run_dir,
    }
    bindings_pass = all(bindings[key] for key in ("resolved_command_sha256", "resolved_config_sha256", "output_run_dir"))
    passed = all((checks["foundation_start_frozen"], checks["recipe_exact"], checks["package_manifest_matches"], checks["scientific_boundaries_preserved"], adapter["pass"], base["pass"], bindings_pass))
    record = {
        "schema_version": 1,
        "purpose": "READ_ONLY_PRELAUNCH_IDENTITY_GATE",
        "status": "PASS" if passed else "BLOCKED_HARD_STOP_BASELINE_OR_LAUNCH_IDENTITY_UNVERIFIED",
        "gpu_training_started": False,
        "expected_foundation": start,
        "checks": checks,
        "mounted_adapter": adapter,
        "mounted_base": base,
        "launch_bindings": bindings,
        "pass_required_before_gpu": ["mounted_adapter", "mounted_base", "resolved_command", "resolved_config", "output_run_dir"],
        "failure_action": "HARD_STOP_NO_GPU_LAUNCH",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"status": record["status"], "out": str(args.out), "sha256": sha256(args.out)}, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
