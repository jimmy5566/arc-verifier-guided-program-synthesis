#!/usr/bin/env python3
"""Freeze the train-only cohort for the prospective E03 gradient study.

This program deliberately has no Torch, Transformers, model, or CUDA imports.
It selects the cohort solely from immutable TRAIN identifiers before any gradient
is observed, and writes the exact native token/label serialization used by the
historical all-assistant causal-CE objective.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from training_data.pipeline import IGNORE_INDEX, task_to_sample  # noqa: E402

OUT = ROOT / "experiments/capability_repair_baseline_v1/e03_v7_lora_gradient_interference_diagnostic_v1"
TRAIN = ROOT / "experiments/base_only_targeted_repair_remote_first_v2/remote_dataset_expected/TRAIN.jsonl"
CHECKPOINT = ROOT / "experiments/capability_repair_baseline_v1/CHECKPOINT_MANIFEST_REMOTE_V1.json"
RETENTION = ROOT / "experiments/capability_repair_baseline_v1/corrected_baseline_v1/CAPABILITY_REPAIR_BASELINE_V1_RETENTION.json"

# The canonical scientific names are held separate from source labels. The
# source mapping is explicit so a review can reject it rather than silently
# treating an old synthetic family as a new composition condition.
FAMILIES = (
    ("connected_components", "WEAK_ATOMIC", "ATOMIC_REPAIR", "connected components", "COUNT_4_CONNECTED_COMPONENTS"),
    ("inside_contains", "WEAK_ATOMIC", "ATOMIC_REPAIR", "inside/contains", "SELECT_OBJECT_INSIDE_FRAME"),
    ("width", "WEAK_ATOMIC", "ATOMIC_REPAIR", "width", "SELECT_WIDEST_OBJECT"),
    ("difference", "WEAK_ATOMIC", "ATOMIC_REPAIR", "difference", "OUTER_MASK_MINUS_INNER_MASK"),
    ("orientation", "WEAK_ATOMIC", "ATOMIC_REPAIR", "orientation", "ROTATE_BY_CONTROL_MARKER"),
    ("same_color", "PROTECTED_RETENTION", "ATOMIC_REPAIR", "same color", "SELECT_SHARED_COLOR_CLASS"),
    ("color_mapping", "PROTECTED_RETENTION", "RETENTION_TRAIN", "color mapping", "APPLY_EXPLICIT_COLOR_MAPPING"),
    ("object_selection_then_action", "COMPOSITION", "COMPOSITION_REPAIR", "relation selector action", "SELECT_OBJECT_TOUCHING_REFERENCE_COLOR"),
    ("novel_two_operation_order", "COMPOSITION", "COMPOSITION_REPAIR", "conditional action", "IF_OBJECT_COUNT_EVEN_THEN_COLOR_C_ELSE_B"),
)
SEED = "E03_V7_LORA_GRADIENT_INTERFERENCE_V1_SELECTION_SEED_20261010"


def canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_bytes(canonical(value) + b"\n")
    os.replace(temp, path)


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def token_binding(sample: dict) -> str:
    return sha_bytes(canonical({"input_ids": sample["input_ids"], "labels": sample["labels"]}))


def message_bindings(sample: dict) -> tuple[str, str]:
    prompt = [m for m in sample["messages"] if m["role"] == "user"]
    targets = [m for m in sample["messages"] if m["role"] == "assistant"]
    return sha_bytes(canonical(prompt)), sha_bytes(canonical(targets))


def stable_order(row: dict, canonical_family: str) -> str:
    return sha_bytes((SEED + "|" + canonical_family + "|" + row["episode_id"]).encode("ascii"))


def build_manifest(rows: list[dict]) -> dict:
    if not rows or any(row.get("split") != "TRAIN" for row in rows):
        raise RuntimeError("TRAIN_ONLY_SOURCE_REQUIRED")
    source = defaultdict(list)
    for row in rows:
        source[(row.get("role"), row.get("family"))].append(row)

    family_docs = []
    seen_ids: set[str] = set()
    for canonical_family, study_role, source_role, source_family, expected_program in FAMILIES:
        pool = [row for row in source[(source_role, source_family)] if row.get("program") == expected_program]
        if len(pool) < 32:
            raise RuntimeError(f"INSUFFICIENT_TRAIN_POOL:{canonical_family}:{len(pool)}")
        selected = sorted(pool, key=lambda row: stable_order(row, canonical_family))[:32]
        if len({row["episode_id"] for row in selected}) != 32 or any(row["episode_id"] in seen_ids for row in selected):
            raise RuntimeError(f"NONDISJOINT_TRAIN_SELECTION:{canonical_family}")
        seen_ids.update(row["episode_id"] for row in selected)
        batches = []
        for batch_index in range(4):
            members = []
            for row in selected[batch_index * 8:(batch_index + 1) * 8]:
                sample = task_to_sample({"source_id": row["episode_id"], **row["task"]})
                if not any(value != IGNORE_INDEX for value in sample["labels"]):
                    raise RuntimeError("ZERO_SUPERVISION")
                prompt_sha, target_sha = message_bindings(sample)
                members.append({
                    "episode_id": row["episode_id"], "source_episode_sha256": row["episode_sha256"],
                    "source_role": source_role, "source_family": source_family, "source_program": expected_program,
                    "task_sha256": sha_bytes(canonical(row["task"])),
                    "prompt_serialization_sha256": prompt_sha, "target_serialization_sha256": target_sha,
                    "native_text_sha256": sha_bytes(sample["text"].encode("utf-8")),
                    "input_ids": sample["input_ids"], "labels": sample["labels"],
                    "sequence_length": sample["sequence_length"],
                    "supervised_token_count": sample["assistant_token_count"],
                    "token_label_sha256": token_binding(sample),
                })
            if len(members) != 8:
                raise RuntimeError("MICROBATCH_SIZE_FAIL")
            batches.append({"microbatch_index": batch_index, "members": members})
        family_docs.append({
            "canonical_family": canonical_family, "study_role": study_role,
            "source_mapping": {"role": source_role, "family": source_family, "program": expected_program},
            "microbatches": batches,
        })

    # One whole, predeclared microbatch per family is recomputed B1. This is a
    # fixed sensitivity subset, not a result-selected subset.
    b1 = [{"canonical_family": family["canonical_family"], "microbatch_index": 0,
           "episode_ids": [m["episode_id"] for m in family["microbatches"][0]["members"]]} for family in family_docs]
    counts = Counter(member["episode_id"] for family in family_docs for batch in family["microbatches"] for member in batch["members"])
    if len(counts) != 288 or any(value != 1 for value in counts.values()):
        raise RuntimeError("GLOBAL_MICROBATCH_DISJOINTNESS_FAIL")
    return {
        "schema_version": 1, "protocol_id": "E03_V7_LORA_GRADIENT_INTERFERENCE_DIAGNOSTIC_V1",
        "status": "FROZEN_CPU_ONLY_TRAIN_LABELS", "selection_seed": SEED,
        "rows": 288, "families": family_docs,
        "fixed_batch1_sensitivity_subset": {"rows": 72, "selection": "microbatch_index_0_for_every_family", "families": b1},
        "objective": {
            "name": "HISTORICAL_ALL_ASSISTANT_CAUSAL_CE",
            "label_contract": "all assistant turn token IDs supervised; all user tokens use ignore_index=-100",
            "causal_shift": "cross_entropy(logits[:, :-1], labels[:, 1:])",
            "microbatch_reduction": "FP32 sum over all nonignored shifted labels divided by supervised_token_count",
            "historical_token_weighted_combination": "family gradient weighted by total supervised tokens",
            "equal_family_combination": "unit-normalized family gradients averaged equally",
        },
        "source": {"path": rel(TRAIN), "sha256": sha(TRAIN), "rows": len(rows), "split": "TRAIN"},
        "isolation": {"target_dev_used": False, "retention_sentinel_used": False, "gold_used": False, "dgold_used": False, "final_audit_opened": False},
    }


def main() -> None:
    rows = [json.loads(line) for line in TRAIN.read_text(encoding="utf-8").splitlines() if line.strip()]
    manifest = build_manifest(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / "E03_TRAIN_MICROBATCH_MANIFEST_V1.json"
    atomic_json(manifest_path, manifest)
    retention = json.loads(RETENTION.read_text(encoding="utf-8"))
    if retention["by_capability"]["color mapping"]["exact_rate"] != 1.0:
        raise RuntimeError("STRONG_RETENTION_FAMILY_SELECTION_BASIS_FAIL")
    config = {
        "schema_version": 1, "protocol_id": manifest["protocol_id"],
        "status": "CPU_FROZEN_PENDING_DIRECTOR_PRELAUNCH_REVIEW", "execution_authorized": False,
        "starting_checkpoint": "RECONSTRUCTED_FOUNDATION_V2_V7",
        "checkpoint_manifest_path": rel(CHECKPOINT), "checkpoint_manifest_sha256": sha(CHECKPOINT),
        "manifest_path": rel(manifest_path), "manifest_sha256": sha(manifest_path),
        "strong_retention_family": {"canonical_family": "color_mapping", "selection_before_gradients": True,
            "selection_basis": "Corrected V7 retention profile exact_rate=1.0; selection fixed before E03 gradients",
            "profile_path": rel(RETENTION), "profile_sha256": sha(RETENTION)},
        "composition_source_scope": "Existing legitimate synthetic TRAIN source only; canonical composition labels are explicit source mappings, not a claim of independent transfer.",
        "parameters": {"trainable": "LoRA adapters only", "base_weight_gradients_persisted": False, "optimizer_constructed": False, "optimizer_steps": 0, "parameter_updates": 0},
        "numerics": {"model_execution": "BF16", "loss_reduction_and_gradient_accumulation": "FP32", "primary_microbatch_size": 8, "batch1_sensitivity_rows": 72},
        "decision_rule": {
            "LOCAL_INTERFERENCE_SUPPORTED": "at least 3 of 5 weak families protected median cosine < -0.05 with family-stratified bootstrap upper < 0, OR historical combined protected alignment < -0.05 with upper < 0; every driving sign preserved B1",
            "GRADIENT_CONFLICT_DEPRIORITIZED": "all protected combined lower bounds nonnegative and <10% weak/protected independent pairs negative after sensitivity accounting",
            "INCONCLUSIVE": "any other valid pattern or sign-reversing B1 sensitivity",
            "INVALID_NOT_INTERPRETABLE": "identity, serialization, mapping, nonfinite, source, receipt, or partial-evidence failure",
        },
        "runtime": {"hard_cap_seconds": 1500, "jobs": 1, "automatic_retry": False, "fresh_output_required": True, "duplicate_job_forbidden": True, "atomic_terminal_receipt_required": True, "partial_evidence_preserved": True},
        "forbidden": ["NO_TRAINING", "NO_OPTIMIZER", "NO_PARAMETER_UPDATE", "NO_GENERATION", "NO_GOLD", "NO_DGOLD", "NO_FINAL_AUDIT", "NO_EXECUTION_BEFORE_DIRECTOR_PRELAUNCH_REVIEW"],
    }
    config_path = OUT / "E03_V7_LORA_GRADIENT_INTERFERENCE_DIAGNOSTIC_V1_CONFIG.json"
    atomic_json(config_path, config)
    print(json.dumps({"status": "FROZEN", "rows": manifest["rows"], "manifest_sha256": sha(manifest_path), "config_sha256": sha(config_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
