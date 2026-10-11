#!/usr/bin/env python3
"""Freeze E05 V3 paired arm contrasts from the immutable V2 vectors only."""
from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experiments/capability_repair_baseline_v1/e05_v3_semantically_identifiable_demonstrations_and_adapter_storage_recovery/e05_v3_nonoracle_structural_residual_decomposition_v2"
OUT = ROOT / "experiments/capability_repair_baseline_v1/e05_v3_semantically_identifiable_demonstrations_and_adapter_storage_recovery/e05_v3_paired_arm_contrast_closure_v1"
RESULT = SOURCE / "RESULT_V2.json"
VECTORS = SOURCE / "PER_BASE_VECTORS_V2.json"
CONCLUSION = SOURCE / "RESULT_CONCLUSION_V2.json"
RECEIPT = SOURCE / "EXECUTION_RECEIPT_V2.json"
DIRECTOR_RESPONSE = Path(r"C:\Users\asus\Desktop\py\kaggle\ARC2\orchestration\director\responses\RESULT_REVIEW_BRIEF_V2B_RESPONSE.json")
DIRECTOR_RESPONSE_SHA256 = "bf63af2a1406f2d3c0ced63e3fcc6fa3fa4714aba398352ac00e15de8bd961ba"
TREATMENT = "E04E_RUN003_TREATMENT_EQUAL_SLOT"
REFERENCES = ("E04E_RUN003_CONTROL_EQUAL_SLOT", "RAW_V7_NO_E04E_UPDATE")
CONDITIONS = ("MARKER_BINDING_CONTROL", "MARKER_CONDITIONED_ROTATION")
CATEGORIES = (
    "EXACT_INPUT_COPY",
    "TARGET_FOREGROUND_GEOMETRY_COLOR_ONLY_ERROR",
    "TARGET_CONGRUENT_GEOMETRY_WRONG_PLACEMENT",
    "TARGET_CELL_COUNT_PRESERVED_NONCONGRUENT_GEOMETRY",
    "DIMENSION_OR_CELL_COUNT_FAILURE",
)
EXCLUDED = ("EXACT_TARGET_MAPPING", "PARSER_INVALID")
OUTCOMES = CATEGORIES + EXCLUDED


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic(path: Path, payload: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def outcome(view: dict) -> str:
    status = view["status"]
    if status == "ELIGIBLE":
        category = view.get("category")
        if category not in CATEGORIES:
            raise RuntimeError("UNKNOWN_ELIGIBLE_CATEGORY:" + repr(category))
        return category
    if status in EXCLUDED:
        return status
    raise RuntimeError("UNKNOWN_VIEW_STATUS:" + repr(status))


def indexed_vectors(vectors: list[dict]) -> dict[tuple[str, str, str, int, str], str]:
    index: dict[tuple[str, str, str, int, str], str] = {}
    for record in vectors:
        arm = record["arm"]
        condition = record["condition"]
        base = record["canonical_base_id"]
        views = record["four_view_vector"]
        if len(views) != 4:
            raise RuntimeError("NOT_FOUR_VIEWS:" + base)
        if record["eligible_count"] + record["exact_target_count"] + record["parser_invalid_count"] != 4:
            raise RuntimeError("INCONSISTENT_VECTOR_COUNTS:" + base)
        for view in views:
            key = (arm, condition, base, int(view["turn"]), view["episode_id"])
            if key in index:
                raise RuntimeError("DUPLICATE_VIEW:" + repr(key))
            if (view["arm"], view["condition"], view["canonical_base_id"]) != (arm, condition, base):
                raise RuntimeError("VIEW_METADATA_MISMATCH:" + repr(key))
            index[key] = outcome(view)
    return index


def paired_report(index: dict, reference: str, condition: str) -> dict:
    treatment = {(base, turn, eid): value for (arm, cond, base, turn, eid), value in index.items() if arm == TREATMENT and cond == condition}
    comparator = {(base, turn, eid): value for (arm, cond, base, turn, eid), value in index.items() if arm == reference and cond == condition}
    if set(treatment) != set(comparator):
        raise RuntimeError("ARM_VIEW_IDENTITY_MISMATCH:" + reference + ":" + condition)
    bases = sorted({base for base, _, _ in treatment})
    if len(bases) != 48:
        raise RuntimeError("CANONICAL_BASE_COUNT_NOT_48:" + reference + ":" + condition)
    transitions: Counter[tuple[str, str]] = Counter()
    per_outcome = {name: [] for name in OUTCOMES}
    for base in bases:
        keys = sorted((key for key in treatment if key[0] == base), key=lambda x: x[1])
        if len(keys) != 4:
            raise RuntimeError("BASE_DOES_NOT_HAVE_FOUR_VIEWS:" + base)
        for key in keys:
            transitions[(comparator[key], treatment[key])] += 1
        for name in OUTCOMES:
            ref_share = sum(comparator[key] == name for key in keys) / 4.0
            treatment_share = sum(treatment[key] == name for key in keys) / 4.0
            per_outcome[name].append({"canonical_base_id": base, "reference_share": ref_share, "treatment_share": treatment_share, "delta_treatment_minus_reference": treatment_share - ref_share})
    outcome_reports = {}
    for name, deltas in per_outcome.items():
        changed = [row["canonical_base_id"] for row in deltas if row["delta_treatment_minus_reference"] != 0.0]
        outcome_reports[name] = {
            "per_base_four_view_share_delta": deltas,
            "mean_paired_delta": sum(row["delta_treatment_minus_reference"] for row in deltas) / len(deltas),
            "changed_base_count": len(changed),
            "changed_canonical_base_ids": changed,
            "unchanged_base_count": len(deltas) - len(changed),
        }
    transition_table = [
        {"reference_outcome": ref, "treatment_outcome": trt, "view_count": transitions[(ref, trt)]}
        for ref in OUTCOMES for trt in OUTCOMES
    ]
    return {
        "comparison": "TREATMENT_MINUS_" + reference,
        "condition": condition,
        "canonical_base_count": len(bases),
        "views_per_base": 4,
        "total_paired_views": len(treatment),
        "outcomes": outcome_reports,
        "complete_paired_transition_table": transition_table,
        "transition_total_check": sum(row["view_count"] for row in transition_table),
    }


def main() -> None:
    if OUT.exists():
        raise RuntimeError("CLOSURE_OUTPUT_ALREADY_EXISTS_NO_RERUN")
    required = (RESULT, VECTORS, CONCLUSION, RECEIPT, DIRECTOR_RESPONSE)
    if not all(path.is_file() for path in required):
        raise RuntimeError("REQUIRED_FROZEN_INPUT_MISSING")
    if sha(DIRECTOR_RESPONSE) != DIRECTOR_RESPONSE_SHA256:
        raise RuntimeError("DIRECTOR_RESPONSE_IDENTITY_MISMATCH")
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    conclusion = json.loads(CONCLUSION.read_text(encoding="utf-8"))
    response = json.loads(DIRECTOR_RESPONSE.read_text(encoding="utf-8"))
    if result.get("status") != "COMPLETE_VALID_DESCRIPTIVE_ONLY" or conclusion.get("primary_prospective_decision") != "INCONCLUSIVE_MIXED_REQUIRED_CONDITIONS":
        raise RuntimeError("FROZEN_RESULT_NOT_ELIGIBLE_FOR_CLOSURE")
    if response.get("next_stage") != "CPU_ONLY_E05_V3_PAIRED_ARM_CONTRAST_CLOSURE_V1" or not response.get("execution_authorization", {}).get("cpu_only_paired_contrast_closure"):
        raise RuntimeError("DIRECTOR_RESPONSE_NOT_AUTHORIZING_CLOSURE")
    index = indexed_vectors(json.loads(VECTORS.read_text(encoding="utf-8"))["vectors"])
    expected = 3 * 2 * 48 * 4
    if len(index) != expected:
        raise RuntimeError("FROZEN_VECTOR_COUNT_MISMATCH")
    reports = [paired_report(index, reference, condition) for reference in REFERENCES for condition in CONDITIONS]
    if any(report["transition_total_check"] != 192 for report in reports):
        raise RuntimeError("TRANSITION_TABLE_NOT_EXHAUSTIVE")
    OUT.mkdir(parents=True)
    shutil.copyfile(DIRECTOR_RESPONSE, OUT / "DIRECTOR_RESULT_REVIEW_RESPONSE_V2B.json")
    inputs = {"result_v2_sha256": sha(RESULT), "per_base_vectors_v2_sha256": sha(VECTORS), "result_conclusion_v2_sha256": sha(CONCLUSION), "execution_receipt_v2_sha256": sha(RECEIPT), "director_result_review_response_v2b_sha256": sha(OUT / "DIRECTOR_RESULT_REVIEW_RESPONSE_V2B.json")}
    contrasts = {"schema_version": 1, "stage": "CPU_ONLY_E05_V3_PAIRED_ARM_CONTRAST_CLOSURE_V1", "status": "COMPLETE_VALID_DESCRIPTIVE_ONLY", "input_identities": inputs, "treatment_arm": TREATMENT, "reference_arms": list(REFERENCES), "outcomes": list(OUTCOMES), "conditions": list(CONDITIONS), "reports": reports, "preservation": "All 48 canonical bases and all four views per base are retained in every condition and comparison; no classification is rerun."}
    atomic(OUT / "PAIRED_CONTRASTS_V1.json", contrasts)
    closure = {"schema_version": 1, "stage": contrasts["stage"], "status": contrasts["status"], "input_identities": inputs, "paired_contrasts_sha256": sha(OUT / "PAIRED_CONTRASTS_V1.json"), "prospective_conclusion_preserved": conclusion["primary_prospective_decision"], "interpretation": "Condition-stratified paired arm contrasts are descriptive closure of existing frozen development evidence. They neither establish treatment-specific causality nor authorize intervention selection, checkpoint promotion, model work, or generalization claims.", "prohibitions_observed": result["prohibitions_observed"]}
    atomic(OUT / "CLOSURE_RESULT_V1.json", closure)
    receipt = {"schema_version": 1, "status": closure["status"], "closure_result_sha256": sha(OUT / "CLOSURE_RESULT_V1.json"), "paired_contrasts_sha256": sha(OUT / "PAIRED_CONTRASTS_V1.json"), "model_loaded": False, "gpu_used": False, "training": False, "inference": False, "generation": False, "gradient_work": False, "final_audit_opened": False}
    atomic(OUT / "CLOSURE_RECEIPT_V1.json", receipt)
    brief = {"brief_type": "DIRECTOR_E05_V3_PAIRED_ARM_CONTRAST_CLOSURE_RESULT_REVIEW_V1", "request": "Review this CPU-only descriptive closure of the previously frozen E05 V3 result. Do not promote a checkpoint, select an intervention, or authorize model/GPU/training/inference/generation/gradient/target-expansion work from it.", "closure_result_sha256": sha(OUT / "CLOSURE_RESULT_V1.json"), "paired_contrasts_sha256": sha(OUT / "PAIRED_CONTRASTS_V1.json"), "closure_receipt_sha256": sha(OUT / "CLOSURE_RECEIPT_V1.json"), "input_identities": inputs, "prospective_conclusion_preserved": conclusion["primary_prospective_decision"], "allowed_outcomes": ["CONTINUE_CONTROLLER", "REQUIRE_CHANGES", "PAUSED", "TERMINAL"], "boundaries": ["CPU_ONLY_FROZEN_VECTORS", "NO_RERUN_CLASSIFICATION", "NO_MODEL_LOADING", "NO_GPU_WORK", "NO_TRAINING", "NO_INFERENCE", "NO_GENERATION", "NO_GRADIENT_WORK", "NO_TARGET_EXPANSION", "NO_GOLD_DGOLD_FINAL_AUDIT"]}
    atomic(OUT / "CLOSURE_RESULT_REVIEW_BRIEF_V1.json", brief)
    print(sha(OUT / "CLOSURE_RESULT_REVIEW_BRIEF_V1.json"))


if __name__ == "__main__":
    main()
