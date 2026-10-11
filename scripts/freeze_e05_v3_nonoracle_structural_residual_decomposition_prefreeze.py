#!/usr/bin/env python3
"""Freeze, but do not execute, the Director-specified E05 residual decomposition."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/capability_repair_baseline_v1/e05_v3_semantically_identifiable_demonstrations_and_adapter_storage_recovery"
EVIDENCE = OUT / "RUN_004_REMOTE_EVIDENCE"
PRIOR_SPEC = OUT / "E05_V3_MARKER_RESPONSIVENESS_ERROR_TAXONOMY_PREFREEZE_SPEC_V1.json"
PRIOR_SPEC_SHA256 = "58d0bbdd8fba5f594092160f885160d69bb456fe39b6a9dd99a153c659ae3a3e"
RESULT = OUT / "E05_V3_MARKER_RESPONSIVENESS_ERROR_TAXONOMY_RESULT_V1.json"
RESULT_RECEIPT = OUT / "E05_V3_MARKER_RESPONSIVENESS_ERROR_TAXONOMY_EXECUTION_RECEIPT_V1.json"
RESULT_BRIEF = OUT / "E05_V3_MARKER_RESPONSIVENESS_ERROR_TAXONOMY_RESULT_REVIEW_BRIEF.json"
DIRECTOR_RESPONSE = OUT / "E05_V3_MARKER_RESPONSIVENESS_ERROR_TAXONOMY_RESULT_REVIEW_BRIEF_RESPONSE.json"
OUTPUT_SPEC = OUT / "E05_V3_NONORACLE_STRUCTURAL_RESIDUAL_DECOMPOSITION_PREFREEZE_SPEC_V1.json"
OUTPUT_BRIEF = OUT / "E05_V3_NONORACLE_STRUCTURAL_RESIDUAL_DECOMPOSITION_PREEXECUTION_REVIEW_BRIEF.json"
RAW_NAMES = (
    "RAW_V7_NO_E04E_UPDATE_RAW_GENERATIONS.jsonl",
    "E04E_RUN003_CONTROL_EQUAL_SLOT_RAW_GENERATIONS.jsonl",
    "E04E_RUN003_TREATMENT_EQUAL_SLOT_RAW_GENERATIONS.jsonl",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    if sha256(PRIOR_SPEC) != PRIOR_SPEC_SHA256:
        raise RuntimeError("INVALID_PREFREEZE_PRIOR_TAXONOMY_SPEC_SHA256")
    prior = json.loads(PRIOR_SPEC.read_text(encoding="utf-8-sig"))
    response = json.loads(DIRECTOR_RESPONSE.read_text(encoding="utf-8-sig"))
    if response.get("reviewed_brief_sha256") != sha256(RESULT_BRIEF):
        raise RuntimeError("INVALID_PREFREEZE_DIRECTOR_RESPONSE_BINDING")
    if response.get("decision") != "CONTINUE_CONTROLLER":
        raise RuntimeError("INVALID_PREFREEZE_DIRECTOR_DECISION")

    sidecar = read_jsonl(OUT / "TARGET_SCORER_SIDECAR.jsonl")
    prompts = read_jsonl(OUT / "INPUT_PROMPTS.jsonl")
    side_ids = [row.get("episode_id") for row in sidecar]
    prompt_ids = [row.get("episode_id") for row in prompts]
    if len(sidecar) != 768 or len(set(side_ids)) != 768:
        raise RuntimeError("INVALID_PREFREEZE_TARGET_SIDECAR_IDENTITIES")
    if len(prompts) != 768 or len(set(prompt_ids)) != 768 or set(prompt_ids) != set(side_ids):
        raise RuntimeError("INVALID_PREFREEZE_INPUT_PROMPT_IDENTITIES")
    by_condition = {}
    for row in sidecar:
        by_condition.setdefault(row.get("condition"), {}).setdefault(row.get("canonical_base_id"), []).append(row)
    if set(by_condition) != {
        "FIXED_TURN_ROTATION_CONTROL",
        "MARKER_BINDING_CONTROL",
        "NO_TRANSFORM_RETENTION_CONTROL",
        "MARKER_CONDITIONED_ROTATION",
    }:
        raise RuntimeError("INVALID_PREFREEZE_CONDITION_SET")
    for condition, bases in by_condition.items():
        if len(bases) != 48 or any(sorted(item.get("turn") for item in rows) != [0, 1, 2, 3] for rows in bases.values()):
            raise RuntimeError("INVALID_PREFREEZE_CANONICAL_BASE_GROUPING:" + str(condition))

    raw_sha256 = {name: sha256(EVIDENCE / name) for name in RAW_NAMES}
    if raw_sha256 != {name: prior["raw_evidence_sha256"][name] for name in RAW_NAMES}:
        raise RuntimeError("INVALID_PREFREEZE_RAW_EVIDENCE_BINDING")
    reused_names = (
        "MANIFEST.json",
        "INPUT_PROMPTS.jsonl",
        "TARGET_SCORER_SIDECAR.jsonl",
        PRIOR_SPEC.name,
        RESULT.name,
        RESULT_RECEIPT.name,
        RESULT_BRIEF.name,
        DIRECTOR_RESPONSE.name,
    )
    reused = {name: sha256(OUT / name) for name in reused_names}

    spec = {
        "schema_version": 1,
        "protocol_id": "E05_V3_SEMANTICALLY_IDENTIFIABLE_DEMONSTRATIONS_AND_ADAPTER_STORAGE_RECOVERY",
        "stage": "CPU_ONLY_E05_V3_NONORACLE_STRUCTURAL_RESIDUAL_DECOMPOSITION_PREFREEZE_V1",
        "status": "CPU_ONLY_PREFREEZE_NO_RESIDUAL_CLASSIFICATION_EXECUTED",
        "director_response_sha256": reused[DIRECTOR_RESPONSE.name],
        "prior_taxonomy_specification_sha256": PRIOR_SPEC_SHA256,
        "prior_taxonomy_result_sha256": reused[RESULT.name],
        "prior_taxonomy_execution_receipt_sha256": reused[RESULT_RECEIPT.name],
        "reused_artifacts_sha256": reused,
        "raw_evidence_sha256": raw_sha256,
        "source_freezer_sha256": sha256(Path(__file__)),
        "population": {
            "scope": ["MARKER_BINDING_CONTROL", "MARKER_CONDITIONED_ROTATION"],
            "arms": [
                "RAW_V7_NO_E04E_UPDATE",
                "E04E_RUN003_CONTROL_EQUAL_SLOT",
                "E04E_RUN003_TREATMENT_EQUAL_SLOT",
            ],
            "eligible_view": "parser_valid is true AND parsed_grid differs from that episode's sealed target",
            "excluded_view_reporting": [
                "PARSER_INVALID",
                "EXACT_TARGET_MAPPING",
            ],
            "unit": "canonical base with four turn views retained",
            "base_aggregation": "For each category, compute count of eligible views in the four-view canonical base divided by four; report the unweighted mean over all 48 bases. Report eligible-view coverage separately. Do not drop bases with zero eligible views.",
        },
        "task_schema_semantics": {
            "test_input": "the sole grid at prompt.test[0].input, joined by episode_id",
            "target_background": "unique modal color in the sealed target grid; ties invalidate the affected view",
            "target_foreground_support": "coordinates whose target color differs from target_background",
            "output_foreground_support": "coordinates whose parsed output color differs from target_background, evaluated only when parsed output has target dimensions",
            "translation_congruent": "there is a single nonzero integer row/column offset that maps every target foreground coordinate bijectively to output foreground support within bounds; rotations, reflections, recoloring, cropping, and resizing do not qualify",
        },
        "taxonomy_order": [
            "EXACT_INPUT_COPY",
            "TARGET_FOREGROUND_GEOMETRY_COLOR_ONLY_ERROR",
            "TARGET_CONGRUENT_GEOMETRY_WRONG_PLACEMENT",
            "TARGET_CELL_COUNT_PRESERVED_NONCONGRUENT_GEOMETRY",
            "DIMENSION_OR_CELL_COUNT_FAILURE",
        ],
        "taxonomy_precedence": {
            "EXACT_INPUT_COPY": "parsed_grid equals the full frozen test input grid exactly, before any target-shape comparison",
            "TARGET_FOREGROUND_GEOMETRY_COLOR_ONLY_ERROR": "otherwise, target and output dimensions match; output foreground support equals target foreground support; every target-background coordinate equals target_background in the output; and at least one target-foreground color differs from target",
            "TARGET_CONGRUENT_GEOMETRY_WRONG_PLACEMENT": "otherwise, dimensions match; output foreground support is a nonzero translation of target foreground support; and every output coordinate outside that support equals target_background",
            "TARGET_CELL_COUNT_PRESERVED_NONCONGRUENT_GEOMETRY": "otherwise, dimensions match and output foreground-support cardinality equals target foreground-support cardinality",
            "DIMENSION_OR_CELL_COUNT_FAILURE": "otherwise, output dimensions differ from target OR foreground-support cardinality differs from target; this is the residual catch-all and makes the taxonomy exhaustive",
        },
        "invalidity_gates": [
            "every frozen raw evidence hash, target-sidecar hash, input-prompt hash, result hash, receipt hash, result-review brief hash, and Director-response binding must match",
            "every raw arm must contain exactly 768 unique episode_id values equal to the target sidecar's set and its declared arm_id",
            "every canonical base in each required condition must retain exactly turns 0,1,2,3",
            "a parser-valid eligible view with non-rectangular grids, missing sole test input, tied target background mode, or an unclassifiable category is INVALID_NOT_INTERPRETABLE",
            "any category overlap or nonexhaustive eligible view is INVALID_NOT_INTERPRETABLE",
        ],
        "required_reports": [
            "eligible, parser-invalid, and exact-target view counts by arm and condition",
            "category counts and canonical-base equal-weight shares by arm and condition",
            "Treatment versus Control and Treatment versus V7 per-base category-share deltas as descriptive evidence",
            "all per-base four-view category vectors, with no adaptive filtering",
        ],
        "bootstrap": {
            "seed": 20261013,
            "replicates": 10000,
            "resampling": "paired canonical-base bootstrap within each arm and condition; retain all four views for each sampled base",
            "interval": "percentile 95 percent",
            "denominator_instability": "If an eligible-residual denominator is zero in any bootstrap draw, that estimand is INCONCLUSIVE for the condition rather than repaired or reweighted.",
        },
        "prospective_decision_rule": {
            "mechanism_dominant": "For Treatment, one and only one category has lower 95 percent bootstrap bound above 0.50 of eligible parser-valid nonoracle views in both required conditions, and neither condition has denominator instability.",
            "mixed_or_inconclusive": "Otherwise, including disagreement across conditions, tied or absent dominance, invalid identities, nonexclusive taxonomy, or denominator instability.",
            "interpretation_limit": "Descriptive morphology on reused synthetic development evidence only; no causal training mechanism, checkpoint promotion, intervention selection, or independent ARC generalization claim.",
        },
        "prohibitions": [
            "NO_MODEL_LOADING",
            "NO_GPU_WORK",
            "NO_TRAINING",
            "NO_INFERENCE",
            "NO_GENERATION",
            "NO_GRADIENT_WORK",
            "NO_TARGET_EXPANSION",
            "NO_GOLD_DGOLD_FINAL_AUDIT",
            "NO_ADAPTIVE_TUNING_OR_INTERVENTION_SELECTION",
        ],
        "execution_authorization": False,
        "preexecution_review_required": True,
    }
    OUTPUT_SPEC.write_text(json.dumps(spec, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    brief = {
        "brief_type": "DIRECTOR_E05_V3_NONORACLE_STRUCTURAL_RESIDUAL_DECOMPOSITION_PREEXECUTION_REVIEW",
        "protocol_id": spec["protocol_id"],
        "request": "Approve or reject exactly the frozen CPU-only residual-decomposition execution. No model loading, GPU work, training, generation, inference, gradient work, target expansion, or intervention selection is requested.",
        "specification_path": OUTPUT_SPEC.name,
        "specification_sha256": sha256(OUTPUT_SPEC),
        "director_result_response_sha256": spec["director_response_sha256"],
        "allowed_outcomes": [
            "CONTINUE_CONTROLLER only for unchanged CPU-only execution",
            "REQUIRE_CHANGES",
            "PAUSED",
            "TERMINAL",
        ],
        "boundaries": spec["prohibitions"],
    }
    OUTPUT_BRIEF.write_text(json.dumps(brief, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print("PASS_E05_V3_NONORACLE_RESIDUAL_PREFREEZE")


if __name__ == "__main__":
    main()
