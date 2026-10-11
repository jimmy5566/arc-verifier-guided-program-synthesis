#!/usr/bin/env python3
"""Freeze the preregistered cross-condition conclusion from E05 V2 outputs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/capability_repair_baseline_v1/e05_v3_semantically_identifiable_demonstrations_and_adapter_storage_recovery/e05_v3_nonoracle_structural_residual_decomposition_v2"
RESULT = OUT / "RESULT_V2.json"
RECEIPT = OUT / "EXECUTION_RECEIPT_V2.json"
VECTORS = OUT / "PER_BASE_VECTORS_V2.json"
CONCLUSION = OUT / "RESULT_CONCLUSION_V2.json"
BRIEF = OUT / "RESULT_REVIEW_BRIEF_V2B.json"
CONDITIONS = ("MARKER_BINDING_CONTROL", "MARKER_CONDITIONED_ROTATION")
TREATMENT = "E04E_RUN003_TREATMENT_EQUAL_SLOT"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic(path: Path, payload: dict) -> None:
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def main() -> None:
    if CONCLUSION.exists() or BRIEF.exists():
        raise RuntimeError("CONCLUSION_OUTPUT_ALREADY_EXISTS_NO_RERUN")
    result = json.loads(RESULT.read_text())
    if result.get("status") != "COMPLETE_VALID_DESCRIPTIVE_ONLY":
        raise RuntimeError("RESULT_NOT_VALID_FOR_CONCLUSION")
    qualifying = {}
    for condition in CONDITIONS:
        item = result["aggregate"].get(f"{TREATMENT}|{condition}")
        if not item or item["bootstrap"].get("status") != "VALID":
            raise RuntimeError("PRIMARY_CONDITIONAL_ESTIMAND_INCONCLUSIVE:" + condition)
        lower = {category: interval[0] for category, interval in item["bootstrap"]["category_ci95"].items()}
        qualifying[condition] = sorted(category for category, value in lower.items() if value > .5)
    shared = sorted(set(qualifying[CONDITIONS[0]]) & set(qualifying[CONDITIONS[1]]))
    conclusion = {
        "schema_version": 1,
        "status": "COMPLETE_VALID_DESCRIPTIVE_ONLY",
        "primary_prospective_decision": ("MECHANISM_DOMINANT:" + shared[0]) if len(shared) == 1 else "INCONCLUSIVE_MIXED_REQUIRED_CONDITIONS",
        "treatment_categories_with_lower_ci_above_0_50": qualifying,
        "rule": "One and only one Treatment category must have lower 95 percent bootstrap bound above 0.50 for the primary conditional eligible-residual share in both required conditions.",
        "result_sha256": sha(RESULT),
        "vectors_sha256": sha(VECTORS),
        "execution_receipt_sha256": sha(RECEIPT),
        "interpretation_limit": "Reused synthetic development evidence only; descriptive morphology, not a causal mechanism, checkpoint-promotion basis, intervention choice, or independent ARC-generalization confirmation.",
        "prohibitions_observed": result["prohibitions_observed"],
    }
    atomic(CONCLUSION, conclusion)
    brief = {
        "brief_type": "DIRECTOR_E05_V3_NONORACLE_STRUCTURAL_RESIDUAL_DECOMPOSITION_RESULT_REVIEW_V2B",
        "request": "Interpret the frozen CPU-only descriptive morphology result and its preregistered cross-condition conclusion. Do not infer causal mechanism, select intervention, promote a checkpoint, or authorize model/GPU work.",
        "result_sha256": sha(RESULT),
        "vectors_sha256": sha(VECTORS),
        "execution_receipt_sha256": sha(RECEIPT),
        "conclusion_sha256": sha(CONCLUSION),
        "primary_prospective_decision": conclusion["primary_prospective_decision"],
        "allowed_outcomes": ["CONTINUE_CONTROLLER", "REQUIRE_CHANGES", "PAUSED", "TERMINAL"],
        "boundaries": ["NO_MODEL_LOADING", "NO_GPU_WORK", "NO_TRAINING", "NO_INFERENCE", "NO_GENERATION", "NO_GRADIENT_WORK", "NO_TARGET_EXPANSION", "NO_GOLD_DGOLD_FINAL_AUDIT", "NO_ADAPTIVE_TUNING_OR_INTERVENTION_SELECTION"],
    }
    atomic(BRIEF, brief)
    print(conclusion["primary_prospective_decision"])


if __name__ == "__main__":
    main()
