"""Freeze, but never execute, the sealed Round-009 equivalence evaluation.

This tool is intentionally CPU-only.  It reads only target-blind sentinel
metadata and file names/hashes.  It never opens a candidate reference payload,
loads a model, or invokes a scorer.
"""
from __future__ import annotations

import hashlib
import json
import secrets
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "experiments" / "foundation_v2_reconstruction_and_targeted_repair_v2" / "round_009_equivalence_evaluation"
ARTIFACTS = ROOT / "artifacts" / "capability_pilot_2m_v1"
DIRECTIVES = ROOT / "orchestration" / "director" / "directives"
ROUND_RECEIPT = ROOT / ".arc2-local" / "orchestration" / "receipts" / "RECONSTRUCTED_FOUNDATION_V2_V2_009_TERMINAL_RECEIPT.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def manifest_info(name: str) -> dict[str, object]:
    path = PACKAGE / name
    value = json.loads(path.read_text(encoding="utf-8"))
    items = value.get("episodes", [])
    safe_keys = {"sample_id", "family", "source", "sequence_length", "supervised_token_count", "token_label_sha256"}
    if not isinstance(items, list) or any(not isinstance(x, dict) or set(x) != safe_keys for x in items):
        raise RuntimeError(f"TARGET_BLIND_MANIFEST_INVALID:{name}")
    return {"path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": digest(path), "item_count": len(items), "target_payloads_present": False}


def main() -> int:
    PACKAGE.mkdir(parents=True, exist_ok=True)
    manifests = {name: manifest_info(name) for name in (
        "NOVEL_GENERATION_INPUT_MANIFEST.json", "REPLAY_GENERATION_INPUT_MANIFEST.json", "PROTECTED_CAPABILITY_INPUT_MANIFEST.json"
    )}
    nonce_path = PACKAGE / "ROUND_009_SEALED_EQUIVALENCE_EVALUATION_V1_NONCES.json"
    if nonce_path.exists():
        nonces = json.loads(nonce_path.read_text(encoding="utf-8"))
    else:
        nonces = {"schema_version": 1, "phase_a_nonce": secrets.token_hex(24), "phase_b_nonce": secrets.token_hex(24)}
        if nonces["phase_a_nonce"] == nonces["phase_b_nonce"]:
            raise RuntimeError("NONCE_COLLISION")
        write(nonce_path, nonces)

    sentinel_sources = []
    for filename, surface in (("NOVEL_VALIDATION_SENTINEL.json", "NOVEL"), ("REPLAY_RETENTION_SENTINEL.json", "REPLAY")):
        path = ARTIFACTS / filename
        value = json.loads(path.read_text(encoding="utf-8"))
        sentinel_sources.append({"surface": surface, "path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": digest(path), "episode_count": value["episode_count"], "contains_canonical_generation_outputs": False})

    protocol = {
        "schema_version": 1,
        "protocol_id": "ROUND_009_SEALED_EQUIVALENCE_EVALUATION_V1",
        "kind": "ADDITIVE_EVALUATION_SUBSTAGE_ONLY",
        "round_id": "RECONSTRUCTED_FOUNDATION_V2_V2_009",
        "round_terminal_receipt_sha256": "4f558ece5665c8e7b859ceafa8ea806187380619292aa662f7743b3df3ec9ba0",
        "selected_checkpoint": {"adapter_model_sha256": "1639bf6665ccfbb99cf547f59fcd581663c95b6bad770ae7846fee90f66be1c3", "adapter_config_sha256": "faa1a3bfb957209ca10ea58afbbf275ce5b51592dd78f289b68f7b655ede2b69", "path": "/workspace/arc2/active_runs/reconstructed_foundation_v2_v2_009/checkpoints/tokens_2000000"},
        "directive_sha256": {"DIRECTOR_DIRECTIVE_019": digest(DIRECTIVES / "DIRECTOR_DIRECTIVE_019.json"), "DIRECTOR_DIRECTIVE_020": digest(DIRECTIVES / "DIRECTOR_DIRECTIVE_020.json")},
        "input_manifests": manifests,
        "thresholds": {"novel_generation_exact_min": 0.80, "replay_generation_exact_min": 0.78, "protected_capability_major_regressions_max": 0},
        "execution_authorization": {"model_loading": False, "generation": False, "scoring": False, "gpu_inference": False, "scientific_training": False, "final_audit": False},
        "status": "FROZEN_PENDING_SEALED_REFERENCE_PROVENANCE_AND_LATER_EXECUTION_DIRECTIVE"
    }
    write(PACKAGE / "ROUND_009_SEALED_EQUIVALENCE_EVALUATION_V1_PROTOCOL.json", protocol)

    audit = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "method": "READ_ONLY_CPU_METADATA_AND_FILENAME_DISCOVERY",
        "forbidden_inputs_not_opened": ["Eval60 Gold", "diagnostic Gold", "FINAL_AUDIT", "exposed Novel holdout targets", "targeted-repair data"],
        "known_target_blind_sources": sentinel_sources,
        "manifest_bindings": manifests,
        "searched_roots": ["artifacts/capability_pilot_2m_v1", "artifacts/novel_training_data_v1_1", "artifacts/foundation_v2_capability_diagnostic_v1", "experiments/foundation_v2_reconstruction_and_targeted_repair_v2"],
        "candidate_reference_payloads_opened": 0,
        "canonical_reference_mapping": {"NOVEL": "MISSING", "REPLAY": "MISSING", "PROTECTED_CAPABILITY": "MISSING"},
        "canonical_baseline_outputs": "MISSING",
        "major_regression_definition": "MISSING",
        "conclusion": "FAIL_CLOSED_REFERENCE_PROVENANCE_UNAVAILABLE",
        "next_action": "DIRECTOR_ESCALATION_REQUIRED_BEFORE_ANY_PHASE_A_OR_PHASE_B_EXECUTION"
    }
    write(PACKAGE / "ROUND_009_SEALED_REFERENCE_PROVENANCE_AUDIT.json", audit)

    phase_a = {
        "schema_version": 1, "protocol_id": protocol["protocol_id"], "phase": "A_TARGET_BLIND_PREDICTION",
        "nonce": nonces["phase_a_nonce"], "execution_authorized": False,
        "model_binding": protocol["selected_checkpoint"],
        "base_and_tokenizer_binding": "INHERIT_EXACT_FROZEN_ROUND_009_RUNTIME_IDENTITY_ONLY",
        "inputs": manifests, "prompts": "INPUT_GRIDS_ONLY; NO_TARGET_OR_REFERENCE_ACCESS",
        "decode": "DETERMINISTIC_PARAMETERS_AND_SEED_MUST_BE_BOUND_BY_A_LATER_EXECUTION_DIRECTIVE",
        "required_outputs_before_scoring": ["raw_predictions.jsonl", "parse_records.jsonl", "item_order_manifest.json", "prediction_hashes.json", "terminal_receipt.json"],
        "output_root": "/workspace/arc2/active_runs/reconstructed_foundation_v2_v2_009/sealed_equivalence_phase_a_" + nonces["phase_a_nonce"],
        "prohibitions": ["reference payload access", "scorer launch", "optimizer", "checkpoint writes", "adaptive feedback"],
        "status": "FROZEN_BLOCKED_BY_MISSING_SEALED_REFERENCE_PROVENANCE_AND_EXECUTION_AUTHORIZATION"
    }
    write(PACKAGE / "ROUND_009_SEALED_EQUIVALENCE_PHASE_A_CONTRACT.json", phase_a)
    phase_b = {
        "schema_version": 1, "protocol_id": protocol["protocol_id"], "phase": "B_SEALED_CPU_SCORER",
        "nonce": nonces["phase_b_nonce"], "execution_authorized": False,
        "starts_only_after": "PHASE_A_TERMINAL_IMMUTABLE_PREDICTION_MANIFEST",
        "inputs_allowed": ["phase_a_prediction_manifest", "exact_reference_manifest", "exact_protected_baseline_manifest"],
        "model_loading_forbidden": True, "phase_a_write_forbidden": True, "adaptive_feedback_forbidden": True,
        "output_root": "/workspace/arc2/active_runs/reconstructed_foundation_v2_v2_009/sealed_equivalence_phase_b_" + nonces["phase_b_nonce"],
        "reference_provenance_status": audit["conclusion"],
        "status": "FROZEN_BLOCKED_BY_MISSING_REFERENCE_AND_BASELINE_IDENTITIES"
    }
    write(PACKAGE / "ROUND_009_SEALED_EQUIVALENCE_PHASE_B_SCORER_CONTRACT.json", phase_b)
    metrics = {
        "schema_version": 1, "protocol_id": protocol["protocol_id"],
        "canonical_parse": "ONE_GRID_PER_UNIQUE_MANIFEST_ITEM; MISSING_DUPLICATE_OR_UNPARSABLE_FAIL_CLOSED",
        "normalization": "ARC_NATIVE_GRID_CANONICAL_JSON_EXACT_MATCH",
        "surfaces": {"NOVEL": {"denominator": 128, "minimum_exact_rate": 0.80}, "REPLAY": {"denominator": 64, "minimum_exact_rate": 0.78}, "PROTECTED_CAPABILITY": {"maximum_major_regressions": 0}},
        "required_result_fields": ["denominator", "exact_count", "exact_rate", "missing_count", "invalid_count", "pass", "raw_predictions_sha256", "reference_manifest_sha256", "scorer_sha256", "logs_sha256", "result_sha256"],
        "status": "FROZEN_PENDING_CANONICAL_REFERENCE_AND_BASELINE_IDENTITY"
    }
    write(PACKAGE / "ROUND_009_SEALED_EQUIVALENCE_METRIC_SEMANTICS.json", metrics)
    print("ROUND_009_SEALED_EQUIVALENCE_FREEZE=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
