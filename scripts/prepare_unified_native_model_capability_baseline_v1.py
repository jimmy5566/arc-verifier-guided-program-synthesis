#!/usr/bin/env python3
"""CPU-only freeze for the unified native capability/ranking baseline.

This generator creates only synthetic ARC observations and a locally sealed
test-target sidecar.  It deliberately imports no model library and constructs
no optimizer, candidate, or GPU command.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "UNIFIED_NATIVE_MODEL_CAPABILITY_BASELINE_V1"
HISTORICAL_FOUNDATION_SHA256 = "f0079dd399c1f0521c378a46bd7bbb737a0abc72ef0bb6c47a6c1ae33446afce"

CHECKPOINTS = {
    "RECONSTRUCTED_FOUNDATION_V2_V7": "experiments/capability_repair_baseline_v1/CHECKPOINT_MANIFEST_REMOTE_V1.json",
    "TARGETED_R1": "experiments/capability_repair_baseline_v1/forward_targeted_repair_v1/r1_paired_evaluation_v1/R1_CHECKPOINT_MANIFEST.json",
    "TARGETED_R2": "experiments/capability_repair_baseline_v1/forward_targeted_repair_v1/r2_paired_evaluation_v1/R2_CHECKPOINT_MANIFEST.json",
    "ATOMIC_R1": "experiments/capability_repair_baseline_v1/forward_capability_repair_v2/atomic_prerequisite_repair_successor_proposal_v1/concrete_protocol_v1/ATOMIC_PREREQUISITE_REPAIR_V1_001_FINAL_CHECKPOINT_MANIFEST.json",
    "FAMILY_BALANCED": "experiments/capability_repair_baseline_v1/family_balanced_loss_control_v1/round_001/evaluation_v1/FINAL_CHECKPOINT_MANIFEST.json",
}


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    os.replace(tmp, path)


def recorded_path(path: Path) -> str:
    """Keep repository-relative paths where possible, absolute paths in tests."""
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def move(grid: list[list[int]], dy: int, dx: int) -> list[list[int]]:
    height, width = len(grid), len(grid[0])
    out = [[0] * width for _ in range(height)]
    for y, row in enumerate(grid):
        for x, value in enumerate(row):
            if value:
                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width:
                    out[ny][nx] = value
    return out


def reflect_lr(grid: list[list[int]]) -> list[list[int]]:
    return [list(reversed(row)) for row in grid]


def recolor(grid: list[list[int]], source: int, target: int) -> list[list[int]]:
    return [[target if value == source else value for value in row] for row in grid]


def one_blob(rng: random.Random, color: int) -> list[list[int]]:
    grid = [[0] * 7 for _ in range(7)]
    y, x = rng.randrange(1, 5), rng.randrange(1, 5)
    grid[y][x] = color
    grid[y][x + 1] = color
    grid[y + 1][x] = color
    return grid


def task_for(family: str, rng: random.Random) -> tuple[dict, list[list[int]]]:
    color = rng.choice([1, 2, 3, 4, 5, 6, 7, 8, 9])
    source = rng.choice([1, 2, 3, 4])
    target = rng.choice([5, 6, 7, 8, 9])
    def transform(grid: list[list[int]]) -> list[list[int]]:
        if family == "STRUCTURAL_TRANSLATION":
            return move(grid, 1, 1)
        if family == "STRUCTURAL_REFLECTION":
            return reflect_lr(grid)
        if family == "COMPOSITION_RECOLOR_TRANSLATE":
            return move(recolor(grid, source, target), 1, 0)
        if family == "COMPOSITION_REFLECT_RECOLOR":
            return recolor(reflect_lr(grid), source, target)
        if family == "PROTECTED_SAME_COLOR":
            return recolor(grid, source, target)
        raise ValueError(f"unknown family {family}")
    trains = []
    for _ in range(2):
        grid = one_blob(rng, source if "RECOLOR" in family or family == "PROTECTED_SAME_COLOR" else color)
        trains.append({"input": grid, "output": transform(grid)})
    test_input = one_blob(rng, source if "RECOLOR" in family or family == "PROTECTED_SAME_COLOR" else color)
    return {"train": trains, "test": [{"input": test_input}]}, transform(test_input)


def make_benchmark(seed: int, per_family: int) -> tuple[list[dict], dict[str, list[list[int]]]]:
    rng = random.Random(seed)
    families = (
        "STRUCTURAL_TRANSLATION",
        "STRUCTURAL_REFLECTION",
        "COMPOSITION_RECOLOR_TRANSLATE",
        "COMPOSITION_REFLECT_RECOLOR",
        "PROTECTED_SAME_COLOR",
    )
    rows, targets = [], {}
    for family in families:
        for index in range(per_family):
            task, answer = task_for(family, rng)
            episode_id = f"{PROTOCOL}:SYNTHETIC:{family}:{index:03d}"
            observation = {"episode_id": episode_id, "family": family, "task": task}
            observation_sha256 = sha_bytes(canonical(observation))
            rows.append({
                "episode_id": episode_id,
                "family": family,
                "split": "SYNTHETIC_VALIDATION",
                "target_access": "SEALED_SIDECAR_ONLY",
                "observation": observation,
                "observation_sha256": observation_sha256,
            })
            targets[episode_id] = answer
    return rows, targets


def adapter_sha(manifest: dict) -> str | None:
    for entry in manifest.get("adapter_files", []):
        if entry.get("name") == "adapter_model.safetensors":
            return entry.get("sha256")
    return None


def discover_checkpoints() -> dict:
    records = {
        "QWEN3_4B_GRIDS15_SFT139_BASE_LORA_DISABLED": {
            "status": "BASE_IDENTITY_DERIVED_FROM_V7_MANIFEST_RUNTIME_UNVERIFIED",
            "lora_disabled": True,
            "base_model_identity_source": CHECKPOINTS["RECONSTRUCTED_FOUNDATION_V2_V7"],
        },
        "HISTORICAL_FOUNDATION_V2": {
            "status": "UNAVAILABLE_NOT_SUBSTITUTED",
            "required_adapter_model_sha256": HISTORICAL_FOUNDATION_SHA256,
        },
    }
    for checkpoint_id, relative in CHECKPOINTS.items():
        path = ROOT / relative
        if not path.is_file():
            records[checkpoint_id] = {"status": "LOCAL_MANIFEST_MISSING", "manifest_path": relative}
            continue
        manifest = json.loads(path.read_text(encoding="utf-8-sig"))
        records[checkpoint_id] = {
            "status": "LOCAL_MANIFEST_VERIFIED_REMOTE_RUNTIME_PENDING",
            "manifest_path": relative,
            "manifest_sha256": sha_file(path),
            "checkpoint_id": manifest.get("checkpoint_id"),
            "adapter_model_sha256": adapter_sha(manifest),
            "source_commit": manifest.get("source_commit"),
            "runtime_path_is_not_identity": True,
        }
    return records


def prepare(out_dir: Path, sealed_root: Path, seed: int, per_family: int) -> dict:
    if per_family < 4:
        raise ValueError("PER_FAMILY_MINIMUM_FOUR")
    rows, targets = make_benchmark(seed, per_family)
    ids = [row["episode_id"] for row in rows]
    hashes = [row["observation_sha256"] for row in rows]
    if len(ids) != len(set(ids)) or len(hashes) != len(set(hashes)):
        raise RuntimeError("SYNTHETIC_CONTENT_COLLISION")
    sidecar = sealed_root / "UNIFIED_NATIVE_MODEL_CAPABILITY_BASELINE_V1_TARGET_SIDECAR.json"
    atomic_json(sidecar, {"protocol_id": PROTOCOL, "targets": targets})
    sidecar_sha = sha_file(sidecar)
    manifest = {
        "schema_version": 1,
        "protocol_id": PROTOCOL,
        "status": "INPUTS_FROZEN_TARGETS_SEALED_CPU_ONLY",
        "generator": {"id": "UNIFIED_NATIVE_SYNTHETIC_GENERATOR_V1", "seed": seed, "per_family": per_family},
        "episodes": rows,
        "content_level_audit": {
            "episode_ids_unique": True,
            "observation_hashes_unique": True,
            "generator_namespace_is_new": True,
            "historical_arc_episode_ids_reused": False,
            "gold_dgold_final_audit_accessed": False,
            "note": "Synthetic observations are newly generated; a future launch must verify the same hashes on the runtime checkout.",
        },
        "sealed_target_sidecar": {"local_path": str(sidecar.resolve()), "sha256": sidecar_sha, "not_committed": True},
    }
    manifest_path = out_dir / "SYNTHETIC_BENCHMARK_INPUT_MANIFEST_V1.json"
    atomic_json(manifest_path, manifest)
    discovery = {"schema_version": 1, "protocol_id": PROTOCOL, "checkpoint_records": discover_checkpoints()}
    discovery_path = out_dir / "CHECKPOINT_PROVENANCE_DISCOVERY_V1.json"
    atomic_json(discovery_path, discovery)
    contract = {
        "schema_version": 1,
        "protocol_id": PROTOCOL,
        "status": "CPU_FROZEN_NOT_LAUNCHABLE_UNTIL_RUNTIME_GATES_PASS",
        "source_commit": os.environ.get("ARC2_SOURCE_COMMIT", "DISCOVER_AT_COMMIT_TIME"),
        "input_manifest_path": recorded_path(manifest_path),
        "input_manifest_sha256": sha_file(manifest_path),
        "checkpoint_discovery_path": recorded_path(discovery_path),
        "checkpoint_discovery_sha256": sha_file(discovery_path),
        "inference_contract": {
            "format": "NATIVE_ARC_OBSERVATION",
            "batch_size": 1,
            "precision": "BF16",
            "primary": "GREEDY",
            "alternative": "AT_MOST_ONE_TARGET_BLIND_FORCE_RANK2_AT_EARLIEST_MINIMUM_TOP1_TOP2_MARGIN_THEN_GREEDY",
            "shared_runtime_tokenizer_parser_eos_scorer": True,
            "runtime_cap_seconds": 9000,
        },
        "required_prelaunch_gates": [
            "LOCAL_HEAD_EQUALS_ORIGIN_EQUALS_RUNPOD_CHECKOUT",
            "all_available_checkpoint_adapter_sha256_verified_on_runtime",
            "historical_foundation_only_if_exact_expected_sha256_else_unavailable_not_substituted",
            "input_manifest_sha256_and_sealed_target_sidecar_sha256_verified",
            "fresh_output_root",
            "no_duplicate_live_job",
            "no_TTT_augmentation_DFS_beam4_external_selector_Gold_dGold_FINAL_AUDIT",
        ],
        "forbidden": ["training", "optimizer", "backward", "model_mutation", "TTT", "augmentation", "DFS", "Beam-4", "external_selector", "Gold", "dGold", "FINAL_AUDIT"],
        "outcomes": ["greedy_exact_grid_top1", "complete_output_top2_coverage", "family_results", "structural_transfer", "compositional_transfer", "protected_retention_same_color", "correct_token_rank", "logit_margin", "first_free_running_error", "critical_token_ranking_change"],
    }
    contract_path = out_dir / "EXECUTION_CONTRACT_DRAFT_V1.json"
    atomic_json(contract_path, contract)
    return {"manifest": manifest_path, "discovery": discovery_path, "contract": contract_path, "sealed_target_sidecar": sidecar}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "experiments/capability_repair_baseline_v1/unified_native_model_capability_baseline_v1")
    parser.add_argument("--sealed-root", type=Path, default=ROOT / ".arc2-local/orchestration/sealed_synthetic_targets")
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--per-family", type=int, default=12)
    args = parser.parse_args()
    produced = prepare(args.out, args.sealed_root, args.seed, args.per_family)
    print(json.dumps({key: {"path": str(value), "sha256": sha_file(value)} for key, value in produced.items()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
