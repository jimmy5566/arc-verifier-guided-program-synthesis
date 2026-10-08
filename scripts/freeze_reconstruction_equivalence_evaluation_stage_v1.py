"""Freeze the single stage-level Round-009 equivalence evaluation package.

This registers prior immutable evidence by hash.  It does not rerun prior
audits, access target payloads, load a model, or execute an evaluation.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "experiments" / "foundation_v2_reconstruction_and_targeted_repair_v2" / "round_009_equivalence_evaluation"
BRIEFS = ROOT / "orchestration" / "director" / "briefs"
ESCALATIONS = ROOT / "orchestration" / "director" / "escalations"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def main() -> int:
    files = [
        "ROUND_009_SEALED_EQUIVALENCE_EVALUATION_V1_PROTOCOL.json",
        "ROUND_009_SEALED_REFERENCE_PROVENANCE_AUDIT.json",
        "ROUND_009_SEALED_EQUIVALENCE_PHASE_A_CONTRACT.json",
        "ROUND_009_SEALED_EQUIVALENCE_PHASE_B_SCORER_CONTRACT.json",
        "ROUND_009_SEALED_EQUIVALENCE_METRIC_SEMANTICS.json",
        "ROUND_009_SEALED_EQUIVALENCE_CPU_PREFLIGHT.json",
        "NOVEL_GENERATION_INPUT_MANIFEST.json",
        "REPLAY_GENERATION_INPUT_MANIFEST.json",
        "PROTECTED_CAPABILITY_INPUT_MANIFEST.json",
    ]
    evidence = {name: {"path": str((PKG / name).relative_to(ROOT)).replace("\\", "/"), "sha256": sha(PKG / name), "reused": name.startswith(("NOVEL_", "REPLAY_", "PROTECTED_"))} for name in files}
    registry = {
        "schema_version": 1,
        "registry_id": "ARC2_FROZEN_EVIDENCE_REGISTRY_V1",
        "reuse_rule": "REUSE_ONLY_WHEN_CONTENT_SHA256_GIT_PROVENANCE_AND_SCOPE_ARE_UNCHANGED",
        "scope": "RECONSTRUCTION_EQUIVALENCE_EVALUATION_STAGE_V1",
        "reused_evidence": evidence,
        "new_evidence": {name: item for name, item in evidence.items() if not item["reused"]},
        "provenance_commit": "dce13d29f09925aa1f9412d0195c6ef30ca0617d",
        "status": "FROZEN_INCREMENTAL_EVIDENCE"
    }
    write(PKG / "ARC2_FROZEN_EVIDENCE_REGISTRY_V1.json", registry)
    stage = {
        "schema_version": 1,
        "stage_id": "RECONSTRUCTION_EQUIVALENCE_EVALUATION_STAGE_V1",
        "parent_protocol_id": "FOUNDATION_V2_RECONSTRUCTION_AND_TARGETED_REPAIR_V2",
        "round_id": "RECONSTRUCTED_FOUNDATION_V2_V2_009",
        "checkpoint_identity": {"adapter_model_sha256": "1639bf6665ccfbb99cf547f59fcd581663c95b6bad770ae7846fee90f66be1c3", "adapter_config_sha256": "faa1a3bfb957209ca10ea58afbbf275ce5b51592dd78f289b68f7b655ede2b69"},
        "evaluation_dataset_identity": {key: evidence[key] for key in evidence if key.endswith("INPUT_MANIFEST.json")},
        "decoding_contract": evidence["ROUND_009_SEALED_EQUIVALENCE_PHASE_A_CONTRACT.json"],
        "scorer_parser_identity": {"phase_b": evidence["ROUND_009_SEALED_EQUIVALENCE_PHASE_B_SCORER_CONTRACT.json"], "metrics": evidence["ROUND_009_SEALED_EQUIVALENCE_METRIC_SEMANTICS.json"]},
        "protected_capability_reference": {"status": "MISSING", "audit": evidence["ROUND_009_SEALED_REFERENCE_PROVENANCE_AUDIT.json"]},
        "equivalence_thresholds": {"novel_generation_exact_min": 0.80, "replay_generation_exact_min": 0.78, "protected_capability_major_regressions_max": 0},
        "provenance": {"evidence_registry": {"path": str((PKG / "ARC2_FROZEN_EVIDENCE_REGISTRY_V1.json").relative_to(ROOT)).replace("\\", "/"), "sha256": sha(PKG / "ARC2_FROZEN_EVIDENCE_REGISTRY_V1.json")}, "source_commit": registry["provenance_commit"]},
        "output_contract": {"phase_a_immutable_before_phase_b": True, "phase_roots_distinct": True, "terminal_receipts_required": True, "no_adaptive_feedback": True},
        "stage_execution_authorization": "PENDING_DIRECTOR_STAGE_DECISION",
        "stage_blocker": "EXACT_SEALED_REFERENCE_AND_PROTECTED_BASELINE_PROVENANCE_MISSING",
        "forbidden": ["model_loading", "generation", "scoring", "GPU_inference", "targeted_repair", "FINAL_AUDIT"],
    }
    write(PKG / "RECONSTRUCTION_EQUIVALENCE_EVALUATION_STAGE_V1.json", stage)
    stage_hash = sha(PKG / "RECONSTRUCTION_EQUIVALENCE_EVALUATION_STAGE_V1.json")
    brief = {
        "schema_version": 1, "brief_id": "DIRECTOR_BRIEF_021", "stage_id": stage["stage_id"], "directive_id": "DIRECTOR_DIRECTIVE_020", "directive_sha256": "a352eb4d48d7affebee20e81f64eaeb666cfca6a06eea021a7c43579a4e91837",
        "stage_package": {"path": str((PKG / "RECONSTRUCTION_EQUIVALENCE_EVALUATION_STAGE_V1.json").relative_to(ROOT)).replace("\\", "/"), "sha256": stage_hash},
        "evidence_registry": {"path": str((PKG / "ARC2_FROZEN_EVIDENCE_REGISTRY_V1.json").relative_to(ROOT)).replace("\\", "/"), "sha256": sha(PKG / "ARC2_FROZEN_EVIDENCE_REGISTRY_V1.json")},
        "completed_cpu_only_work": ["target-blind manifests reused by hash", "two phase output-isolation contracts frozen", "metric semantics frozen", "no-model preflight passed to fail-closed boundary"],
        "blocker": stage["stage_blocker"], "scientific_training_started": False, "model_loading_started": False, "evaluation_execution_started": False, "final_audit_accessed": False,
        "requested_decision": "Authorize this complete stage only if exact pre-existing sealed references, protected baseline, and major-regression definition are bound; otherwise provide the smallest scientifically valid replacement route."
    }
    write(BRIEFS / "DIRECTOR_BRIEF_021.json", brief)
    escalation = {
        "schema_version": 1, "status": "ESCALATION_REQUESTED", "escalation_type": "BLOCKED_SCIENCE", "directive_id": "DIRECTOR_DIRECTIVE_020", "directive_sha256": brief["directive_sha256"],
        "blocker": [stage["stage_blocker"]], "attempts_already_made": 1, "evidence": {"stage_package": str((PKG / "RECONSTRUCTION_EQUIVALENCE_EVALUATION_STAGE_V1.json").resolve()), "brief": str((BRIEFS / "DIRECTOR_BRIEF_021.json").resolve())},
        "requested_decision": "STAGE_LEVEL_SEALED_REFERENCE_PROVENANCE_DECISION_REQUIRED", "scientific_training_started": False, "evaluation_execution_started": False
    }
    write(ESCALATIONS / "CONTROLLER_ESCALATION_REQUEST_006.json", escalation)
    print("RECONSTRUCTION_EQUIVALENCE_EVALUATION_STAGE_V1=FROZEN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
