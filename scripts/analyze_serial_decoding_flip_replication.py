#!/usr/bin/env python3
"""Score the frozen Batch1 flip diagnostic against its preserved Batch32 rows."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


MODEL_TO_SOURCE = {
    "CAPABILITY_REPAIR_BASELINE_V1_V7": "v7",
    "FORWARD_TARGETED_CAPABILITY_REPAIR_V1_003_FINAL": "r1",
    "FORWARD_TARGETED_CAPABILITY_REPAIR_V1_004_FINAL": "r2",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--batch1", type=Path, required=True)
    parser.add_argument("--v7-batch32", type=Path, required=True)
    parser.add_argument("--r1-batch32", type=Path, required=True)
    parser.add_argument("--r2-batch32", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("OUTPUT_PATH_NON_OVERWRITE_REQUIRED")
    cohort = json.loads(args.cohort.read_text(encoding="utf-8"))
    batch1 = json.loads(args.batch1.read_text(encoding="utf-8"))
    if cohort.get("episode_count") != 65 or batch1.get("status") != "COLLECTED_PASS" or batch1.get("batch_size") != 1:
        raise RuntimeError("BATCH1_DIAGNOSTIC_BINDING_INVALID")
    cohort_rows = cohort["cohort"]
    ids = [item["episode_id"] for item in cohort_rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("COHORT_IDENTITY_INVALID")
    batch32_paths = {"v7": args.v7_batch32, "r1": args.r1_batch32, "r2": args.r2_batch32}
    batch32 = {key: json.loads(path.read_text(encoding="utf-8")) for key, path in batch32_paths.items()}
    for key, payload in batch32.items():
        if payload.get("status") != "COLLECTED_PASS":
            raise RuntimeError(f"BATCH32_RESULT_INVALID:{key}")
    families = {item["episode_id"]: item["family"] for item in cohort_rows}
    b1_models = {item["model_id"]: item for item in batch1.get("models", [])}
    if set(b1_models) != set(MODEL_TO_SOURCE):
        raise RuntimeError("BATCH1_MODEL_IDENTITY_INVALID")
    compared: dict[str, Any] = {}
    model_predictions: dict[str, dict[str, dict[str, Any]]] = {}
    for model_id, source in MODEL_TO_SOURCE.items():
        b1_rows = b1_models[model_id].get("predictions", [])
        if [item.get("episode_id") for item in b1_rows] != ids:
            raise RuntimeError(f"BATCH1_COHORT_ORDER_INVALID:{model_id}")
        b32_by_id = {item.get("episode_id"): item for item in batch32[source].get("predictions", [])}
        if set(ids) - set(b32_by_id):
            raise RuntimeError(f"BATCH32_COHORT_MISSING:{model_id}")
        agreement = sum(
            item.get("canonical_prediction_sha256") == b32_by_id[item["episode_id"]].get("canonical_prediction_sha256")
            for item in b1_rows
        )
        compared[model_id] = {
            "batch32_source": str(batch32_paths[source]),
            "batch32_source_sha256": sha(batch32_paths[source]),
            "cohort_episodes": len(ids),
            "canonical_agreement_count": agreement,
            "canonical_agreement": agreement / len(ids),
            "canonical_agreement_threshold": 0.9375,
            "canonical_agreement_pass": agreement / len(ids) >= 0.9375,
        }
        model_predictions[source] = {item["episode_id"]: item for item in b1_rows}
    family_rows: list[dict[str, Any]] = []
    for family in sorted(set(families.values())):
        family_ids = [item for item in ids if families[item] == family]
        exact = {label: sum(bool(model_predictions[label][item]["exact_grid_match"]) for item in family_ids) for label in ("v7", "r1", "r2")}
        family_rows.append({
            "family": family, "episodes": len(family_ids), "v7_exact_correct": exact["v7"],
            "r1_exact_correct": exact["r1"], "r2_exact_correct": exact["r2"],
            "r1_minus_v7": exact["r1"] - exact["v7"], "r2_minus_v7": exact["r2"] - exact["v7"],
        })
    by_family = {item["family"]: item for item in family_rows}
    required = {
        "conditional_action_r1_positive": by_family["conditional action"]["r1_minus_v7"] > 0,
        "conditional_action_r2_positive": by_family["conditional action"]["r2_minus_v7"] > 0,
        "inside_contains_r1_positive": by_family["inside/contains"]["r1_minus_v7"] > 0,
        "inside_contains_r2_positive": by_family["inside/contains"]["r2_minus_v7"] > 0,
        "same_color_r1_negative": by_family["same color"]["r1_minus_v7"] < 0,
        "same_color_r2_negative": by_family["same color"]["r2_minus_v7"] < 0,
        "difference_r1_negative": by_family["difference"]["r1_minus_v7"] < 0,
        "difference_r2_negative": by_family["difference"]["r2_minus_v7"] < 0,
        "overlay_r2_negative": by_family["overlay"]["r2_minus_v7"] < 0,
    }
    agreement_pass = all(item["canonical_agreement_pass"] for item in compared.values())
    signs_pass = all(required.values())
    result = {
        "schema_version": 1,
        "diagnostic_id": "V7_R1_R2_SERIAL_DECODING_FLIP_REPLICATION_V1",
        "status": "ANALYZED",
        "batch1_result_path": str(args.batch1), "batch1_result_sha256": sha(args.batch1),
        "cohort_path": str(args.cohort), "cohort_sha256": sha(args.cohort),
        "batch1_vs_batch32": compared,
        "batch1_family_exact_counts": family_rows,
        "director_required_signs": required,
        "director_rule_pass": agreement_pass and signs_pass,
        "director_rule_components": {"all_canonical_agreements_pass": agreement_pass, "all_family_signs_pass": signs_pass},
        "scientific_training_started": False, "optimizer_constructed": False, "final_audit_opened": False,
    }
    write(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
