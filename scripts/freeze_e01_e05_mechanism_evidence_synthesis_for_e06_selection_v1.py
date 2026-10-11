#!/usr/bin/env python3
"""Freeze a CPU-only E01--E05 evidence matrix for Director E06 selection."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/capability_repair_baseline_v1/e06_selection_evidence_synthesis_v1"
EXTERNAL_ROOT = Path(r"C:\Users\asus\Desktop\py\kaggle\ARC2")
DIRECTOR_RESPONSE = EXTERNAL_ROOT / "orchestration/director/responses/CLOSURE_RESULT_REVIEW_BRIEF_V1_RESPONSE.json"
DIRECTOR_RESPONSE_SHA256 = "d7c1d095e249d89357146e1b4c1a47f18a10d92571467daff308244a3c93f997"

SOURCES = {
    "e01_measurement_status": ROOT / "orchestration/ARC2_RESEARCH_AGENDA_E01_E07.md",
    "e02_nll_exact_alignment": ROOT / "experiments/capability_repair_baseline_v1/paired_fixed_b32_dev_nll_transfer_v1/current_stage_closure_v1/CLOSURE_RESULT_PACKAGE.json",
    "beam4_candidate_recall": ROOT / "experiments/capability_repair_baseline_v1/paired_target_blind_beam4_candidate_recall_v1/run_002_result_v1/RESULT_PACKAGE.json",
    "beam4_director_interpretation": ROOT / "orchestration/director/responses/GOVERNOR_REVIEW_PAIRED_TARGET_BLIND_BEAM4_CANDIDATE_RECALL_V1_RESULT_RESPONSE.json",
    "e03_repeatability_sensitivity": ROOT / "experiments/capability_repair_baseline_v1/e03_v7_lora_gradient_interference_diagnostic_v1/cpu_only_observed_repeatability_envelope_sensitivity_v1/RESULT_PACKAGE.json",
    "e04_local_gradient_result": ROOT / "orchestration/director/responses/E04_D_ROTATION_MARKER_LOCAL_GRADIENT_DIAGNOSTIC_V1_RUN_002_RESULT_BRIEF_RESPONSE.json",
    "e04_native_greedy_b1": ROOT / "experiments/capability_repair_baseline_v1/e04_orientation_fixed_demonstration_baseline_v3/e04_v3_full768_b1_no_update_remeasurement_v1/run_001_2ee668738df64d558f666f5989d09add/RESULT_PACKAGE.json",
    "e05_native_greedy": ROOT / "experiments/capability_repair_baseline_v1/e05_v3_semantically_identifiable_demonstrations_and_adapter_storage_recovery/E05_V3_RUN_004_CPU_EXACT_GRID_RESULT.json",
    "e05_nonoracle_result": ROOT / "experiments/capability_repair_baseline_v1/e05_v3_semantically_identifiable_demonstrations_and_adapter_storage_recovery/e05_v3_nonoracle_structural_residual_decomposition_v2/RESULT_V2.json",
    "e05_paired_closure": ROOT / "experiments/capability_repair_baseline_v1/e05_v3_semantically_identifiable_demonstrations_and_adapter_storage_recovery/e05_v3_paired_arm_contrast_closure_v1/CLOSURE_RESULT_V1.json",
    "e05_paired_contrasts": ROOT / "experiments/capability_repair_baseline_v1/e05_v3_semantically_identifiable_demonstrations_and_adapter_storage_recovery/e05_v3_paired_arm_contrast_closure_v1/PAIRED_CONTRASTS_V1.json",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic(path: Path, payload: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def source_hashes() -> dict[str, str]:
    missing = [name for name, path in SOURCES.items() if not path.is_file()]
    if missing:
        raise RuntimeError("IMMUTABLE_SOURCE_MISSING:" + ",".join(missing))
    return {name: sha(path) for name, path in SOURCES.items()}


def matrix() -> list[dict]:
    common = {
        "all_prospective_studies_require": [
            "one matched control-versus-treatment factor only, exact common starting checkpoint and matched optimizer, updates, token budget, seed policy, decoder, and compute cap",
            "legitimate synthetic TRAIN only, with a newly frozen independently generated non-Gold validation cohort structurally disjoint from TRAIN and historically exposed cohorts",
            "target-blind generation or prediction freeze before exact-grid scoring; Gold, dGold, and FINAL_AUDIT remain sealed",
            "free-running exact-grid equal-family macro primary estimand with paired family-stratified uncertainty, a protected-retention non-inferiority gate, and held-out compositional transfer as mandatory outcomes",
            "predeclared SUCCESS, FAILURE, INCONCLUSIVE, and INVALID_NOT_INTERPRETABLE rules; no validation-driven adaptation or checkpoint promotion",
        ]
    }
    return [
        {
            "mechanism_id": "SEQUENCE_OBJECTIVE_OR_RANK_MISMATCH",
            "status": "STILL_PLAUSIBLE_HIGH_INFORMATION",
            "established_observations": [
                "On the reused 60-episode TARGET_DEV cohort, Family-Balanced teacher-forced NLL was lower for all 60 episodes while Greedy exact grids declined from 9 to 7, with 2 fixes and 4 harms.",
                "Under frozen Batch1 Beam-4, Family-Balanced has 31/60 oracle Recall@4 versus V7 18/60 while rank-one exact is only 11/60 versus 10/60; equal-family Recall@4 difference is 0.2167 with 95% bootstrap interval [0.0833, 0.35].",
            ],
            "supported_hypothesis": "Token-level teacher-forced fitting may leave correct complete sequences insufficiently preferred at rank one.",
            "contradictory_or_limiting_evidence": [
                "The NLL-to-exact gap and oracle candidate reachability are descriptive reused-development evidence, not causal identification of an objective defect.",
                "Oracle Recall@4 is not deployable rank-one performance and cannot validate target-informed selection.",
            ],
            "unresolved_discriminator": "Does a predeclared sequence-level objective improve free-running rank-one exact grids over a matched final-query CE control on a newly generated, disjoint compositional validation cohort while retaining protected capabilities?",
            "minimum_prospective_evidence": common["all_prospective_studies_require"],
        },
        {
            "mechanism_id": "START_STATE_GRADIENT_ANTAGONISM",
            "status": "DEPRIORITIZED_NOT_EXCLUDED_GLOBALLY",
            "established_observations": [
                "At exact V7 start on the frozen E04 rotation-marker contrast, all 48 measured matched gradient cosines were positive; mean cosine was 0.8518 and bootstrap interval [0.8428, 0.8597].",
                "The E03 repeatability sensitivity result preserves E03 as INCONCLUSIVE_REPEATABILITY_GATE_FAILED; its post-hoc positive lower bounds are TRAIN-only sensitivity evidence.",
            ],
            "supported_hypothesis": "Start-state local rotation-marker antagonism is not supported as the primary explanation for the observed E04 regression.",
            "contradictory_or_limiting_evidence": [
                "Positive start-state alignment does not establish alignment later in training or across all families.",
                "No stable global gradient-interference conclusion follows from the E03 envelope.",
            ],
            "unresolved_discriminator": "Only a separately justified trajectory-resolved, repeatable gradient study could test later-training interference; it is not the preferred E06 training factor from current evidence.",
            "minimum_prospective_evidence": ["If ever reconsidered, first pass a prospectively specified repeatability gate and connect the trajectory measure to the same matched free-running exact-grid and retention endpoints."],
        },
        {
            "mechanism_id": "EXPLICIT_REPLAY_PROTECTION_OR_ALLOCATION",
            "status": "PLAUSIBLE_BUT_NO_TRANSFER_SUCCESS",
            "established_observations": [
                "E05 RUN_004 native-Greedy exact scoring found Treatment and equal-slot Control identical on fixed-turn rotation, marker-conditioned rotation, and marker binding; all three checkpoints were 0/192 on marker binding.",
                "Treatment was down 8/192 on no-transform retention versus V7, with paired interval [-13, -3]; the existing E05 route is valid but inconclusive and checkpoint promotion is rejected.",
            ],
            "supported_hypothesis": "The earlier replay effect, if any, was distribution-specific or noncompositional rather than a robust marker-conditioned repair.",
            "contradictory_or_limiting_evidence": [
                "Equal-slot Treatment-minus-Control is nearly indistinguishable in frozen E05 pairing, so this evidence does not establish replay as the causal remedy.",
                "Marker-binding equality at floor is not preservation evidence.",
            ],
            "unresolved_discriminator": "Can a distinct replay/allocation treatment produce an independently replicated exact-grid gain without protected-retention harm on a new structurally disjoint cohort?",
            "minimum_prospective_evidence": common["all_prospective_studies_require"],
        },
        {
            "mechanism_id": "MARKER_GROUNDING_OR_STRUCTURAL_REPRESENTATION",
            "status": "STILL_PLAUSIBLE",
            "established_observations": [
                "E05 V2 residual morphology is condition-specific: Treatment dimension-or-cell-count failure dominates marker-binding control, while foreground-geometry color-only error dominates marker-conditioned rotation.",
                "The frozen cross-condition conclusion is INCONCLUSIVE_MIXED_REQUIRED_CONDITIONS and E05 paired closure did not establish treatment specificity.",
            ],
            "supported_hypothesis": "The failure may reflect marker grounding, structural representation, or composition rather than a single replay deficit.",
            "contradictory_or_limiting_evidence": [
                "Raw V7 and equal-slot Control show the same broad condition split, so the morphology does not identify a Treatment-specific causal mechanism.",
                "The results are reused synthetic development evidence only.",
            ],
            "unresolved_discriminator": "A single representation-focused treatment must separately test marker grounding, transformation execution, and their held-out composition against a matched control, rather than infer cause from residual morphology.",
            "minimum_prospective_evidence": common["all_prospective_studies_require"],
        },
        {
            "mechanism_id": "DECODING_OR_CANDIDATE_SELECTION",
            "status": "SUPPORTED_REACHABILITY_NOT_A_TRAINING_MECHANISM",
            "established_observations": [
                "The frozen Beam-4 result shows a large correct-candidate reachability advantage for Family-Balanced, but not an equivalent rank-one advantage.",
                "The native CPU scorer establishes that frozen Greedy exact-grid and frozen-branch alternate coverage are distinct, target-blind definitions.",
            ],
            "supported_hypothesis": "A target-blind selection or sequence-rank intervention could be a relevant discriminator of rank-one failure.",
            "contradictory_or_limiting_evidence": [
                "Candidate reachability does not identify how training created it and cannot authorize a selector trained or tuned on scored TARGET_DEV candidates.",
                "The historical selector route is not an E06 intervention choice and remains outside this synthesis.",
            ],
            "unresolved_discriminator": "Does a frozen target-blind rule improve rank-one exact grid on a disjoint cohort without target-derived features or protected-family harm?",
            "minimum_prospective_evidence": ["Freeze selector features and thresholds before candidate generation and target scoring, use a disjoint non-Gold cohort, and require a positive paired rank-one gain with no protected-family regression."],
        },
        {
            "mechanism_id": "COMPOSITIONAL_TRANSFER_LIMITATION",
            "status": "STILL_PLAUSIBLE_AND_MANDATORY_TO_TEST",
            "established_observations": [
                "E05 valid native-Greedy evidence did not confirm transfer of the reused E04 effect to marker-conditioned rotation and showed a no-transform retention cost.",
                "No accepted result establishes independent ARC generalization; FINAL_AUDIT remains sealed.",
            ],
            "supported_hypothesis": "Atomic or prompt-structure improvements may fail to transfer to held-out compositions.",
            "contradictory_or_limiting_evidence": [
                "Existing cohorts are reused synthetic development evidence and cannot estimate independent ARC transfer.",
                "The evidence does not identify whether any failure is data coverage, objective, optimization, decoding, or representation.",
            ],
            "unresolved_discriminator": "Does a matched treatment improve a predeclared held-out compositional validation condition while retaining atomic and protected outcomes?",
            "minimum_prospective_evidence": common["all_prospective_studies_require"],
        },
    ]


def main() -> None:
    if OUT.exists():
        raise RuntimeError("E06_EVIDENCE_SYNTHESIS_OUTPUT_ALREADY_EXISTS_NO_RERUN")
    if not DIRECTOR_RESPONSE.is_file() or sha(DIRECTOR_RESPONSE) != DIRECTOR_RESPONSE_SHA256:
        raise RuntimeError("DIRECTOR_RESPONSE_IDENTITY_MISMATCH")
    response = json.loads(DIRECTOR_RESPONSE.read_text(encoding="utf-8"))
    if response.get("next_stage") != "CPU_ONLY_E01_E05_MECHANISM_EVIDENCE_SYNTHESIS_FOR_E06_SELECTION_V1" or not response.get("authorization", {}).get("cpu_only_existing_evidence_synthesis"):
        raise RuntimeError("DIRECTOR_CPU_ONLY_AUTHORIZATION_MISSING")
    hashes = source_hashes()
    OUT.mkdir(parents=True)
    shutil.copyfile(DIRECTOR_RESPONSE, OUT / "DIRECTOR_CLOSURE_RESPONSE_V1.json")
    report = {
        "schema_version": 1,
        "stage": "CPU_ONLY_E01_E05_MECHANISM_EVIDENCE_SYNTHESIS_FOR_E06_SELECTION_V1",
        "status": "COMPLETE_VALID_DESCRIPTIVE_ONLY",
        "director_response_sha256": sha(OUT / "DIRECTOR_CLOSURE_RESPONSE_V1.json"),
        "source_sha256": hashes,
        "e05_prospective_conclusion_preserved": "INCONCLUSIVE_MIXED_REQUIRED_CONDITIONS",
        "e05_route_disposition": "E05_CLOSED_VALID_BUT_INCONCLUSIVE",
        "matrix": matrix(),
        "selection_boundary": "This matrix is evidence for Director selection of at most one future E06 mechanism-guided pilot. It selects no intervention, changes no scientific condition, and creates no training, GPU, model, inference, generation, gradient, target, Gold, dGold, or FINAL_AUDIT authorization.",
        "prohibitions_observed": {"model_loaded": False, "gpu_used": False, "training": False, "inference": False, "generation": False, "gradient_work": False, "targets_opened": False, "final_audit_opened": False},
    }
    atomic(OUT / "E01_E05_MECHANISM_EVIDENCE_MATRIX_V1.json", report)
    receipt = {"schema_version": 1, "status": report["status"], "matrix_sha256": sha(OUT / "E01_E05_MECHANISM_EVIDENCE_MATRIX_V1.json"), "director_response_sha256": report["director_response_sha256"], **report["prohibitions_observed"]}
    atomic(OUT / "EVIDENCE_SYNTHESIS_RECEIPT_V1.json", receipt)
    brief = {
        "brief_type": "DIRECTOR_E06_MECHANISM_GUIDED_PILOT_SELECTION_REVIEW_V1",
        "request": "Select at most one small, falsifiable E06 mechanism-guided pilot for a separately frozen protocol, or stop/revise. Do not authorize execution, model loading, GPU work, training, inference, generation, gradients, target expansion, checkpoint promotion, Gold, dGold, or FINAL_AUDIT from this review.",
        "evidence_matrix_sha256": sha(OUT / "E01_E05_MECHANISM_EVIDENCE_MATRIX_V1.json"),
        "receipt_sha256": sha(OUT / "EVIDENCE_SYNTHESIS_RECEIPT_V1.json"),
        "director_response_sha256": report["director_response_sha256"],
        "e05_prospective_conclusion_preserved": report["e05_prospective_conclusion_preserved"],
        "allowed_outcomes": ["CONTINUE_CONTROLLER", "REQUIRE_CHANGES", "PAUSED", "TERMINAL"],
        "required_if_continuing": ["name exactly one mechanism", "state one matched control-versus-treatment factor", "require legitimate synthetic TRAIN and new independent non-Gold structurally disjoint validation", "freeze free-running exact-grid equal-family macro primary outcome, protected retention, and held-out composition gate", "require a separate pre-execution review before any model or GPU work"],
        "boundaries": ["E05_INCONCLUSIVE_MIXED_REQUIRED_CONDITIONS_PRESERVED", "NO_INTERVENTION_SELECTION_BY_CONTROLLER", "NO_MODEL_GPU_TRAINING_INFERENCE_GENERATION_GRADIENTS", "NO_TARGET_EXPANSION", "NO_GOLD_DGOLD_FINAL_AUDIT"],
    }
    atomic(OUT / "DIRECTOR_E06_SELECTION_REVIEW_BRIEF_V1.json", brief)
    print(sha(OUT / "DIRECTOR_E06_SELECTION_REVIEW_BRIEF_V1.json"))


if __name__ == "__main__":
    main()
