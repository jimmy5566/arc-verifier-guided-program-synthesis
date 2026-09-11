"""Freeze P1--P4 parameter-grounding conditions before any new inference."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm.parameter_grounding import parameter_contract_inventory, prompt_hashes


OUTPUT = ROOT / "configs" / "PARAMETER_GROUNDING_REPAIR_V1_FROZEN_CONFIG.json"
S2_CHECKPOINTS = ROOT / "artifacts" / "program_selection_repair_output_v17" / "checkpoints" / "program_selection_repair_v1" / "S2_FAMILY_FIRST_SELECTION"


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def stable_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"frozen config already exists: {OUTPUT}")
    benchmark = read(ROOT / "configs" / "macro_api_comprehension_benchmark_v1.json")
    s2_result = read(ROOT / "experiments" / "results" / "PROGRAM_SELECTION_REPAIR_V1.json")
    forensics = read(ROOT / "experiments" / "results" / "PARAMETER_GROUNDING_FORENSICS_V1.json")
    if benchmark["case_count"] != 60 or s2_result["conditions"]["S2_FAMILY_FIRST_SELECTION"]["semantic_success"] != 31 or s2_result["conditions"]["S2_FAMILY_FIRST_SELECTION"]["wrong_parameter"] != 17:
        raise ValueError("requires frozen S2 31 success / 17 wrong-parameter baseline")
    if forensics["s2_wrong_parameter_cases"] != 17 or forensics["pure_parameter_failure_count"] + forensics["non_parameter_structural_failure_count"] != 17:
        raise ValueError("forensics must fully partition 17 S2 parameter failures")
    files = sorted(S2_CHECKPOINTS.glob("case_*.json"))
    if len(files) != 60:
        raise ValueError("requires exactly 60 frozen S2 checkpoints")
    baseline_programs: dict[str, dict] = {}; case_outcomes: dict[str, str] = {}
    for path in files:
        record = read(path); case_id = str(record["case_id"])
        structured = json.loads(str(record["structured_raw_response"]))
        baseline_programs[case_id] = {"family": str(record["selection"]["family_prediction"]), "skeleton_id": str(structured["skeleton_id"]), "program": json.loads(str(record["materialized_program"]))}
        case_outcomes[case_id] = str(record["semantic"]["outcome"])
    if len(baseline_programs) != 60 or sum(value in {"SEMANTIC_EXACT_CANONICAL", "SEMANTIC_EQUIVALENT_ALTERNATE"} for value in case_outcomes.values()) != 31:
        raise ValueError("frozen S2 checkpoint coverage/outcomes mismatch")
    fields = {item["case_id"]: list(item["parameter_fields"]) for item in forensics["case_forensics"]}
    contract = parameter_contract_inventory(); hashes = prompt_hashes()
    payload = {
        "experiment_id": "PARAMETER_GROUNDING_REPAIR_V1", "status": "FROZEN_BEFORE_ONE_FORMAL_PASS_PER_CONDITION",
        "benchmark_config": "configs/macro_api_comprehension_benchmark_v1.json", "benchmark_hash": benchmark["benchmark_hash"], "registry_hash": benchmark["registry_hash"], "schema_hash": benchmark["schema_hash"], "case_count": 60, "category_counts": benchmark["category_counts"],
        "model": s2_result["model"], "sampling": {"temperature": 0, "top_p": 1, "seed": 0, "thinking": False}, "generation": {"candidate_budget": 1, "max_new_tokens": 256, "context_window": 12288},
        "semantic_scorer": {"module": "llm.c3_semantic_intent", "source_sha256": hashlib.sha256((ROOT / "src" / "llm" / "c3_semantic_intent.py").read_bytes()).hexdigest(), "llm_judge_used": False},
        "parameter_contract": {"source": "existing Macro registry + compiler-aware literal domains", "sha256": stable_hash(contract), "direct_literal_mode": True}, "prompt_hashes": hashes,
        "prompt_versions": {"P1_NORMALIZED_SLOT_EXTRACTION": "parameter_grounding_v1.normalized", "P2_CONSTRAINED_PARAMETER_CHOICE": "parameter_grounding_v1.constrained", "P3_SLOTWISE_PARAMETER_SELECTION": "parameter_grounding_v1.slotwise", "P4_SELECTIVE_PARAMETER_REPAIR": "parameter_grounding_v1.selective"},
        "p0_s2_baseline": {"source": "PROGRAM_SELECTION_REPAIR_V1/S2_FAMILY_FIRST_SELECTION", "result_sha256": stable_hash(s2_result), "semantic_success": 31, "semantic_success_rate": 31 / 60, "compile_valid": 60, "wrong_parameter": 17, "wrong_operation": 3, "case_outcomes": case_outcomes},
        "s2_baseline_programs": baseline_programs,
        "forensics": {"result_sha256": stable_hash(forensics), "s2_wrong_parameter_cases": 17, "pure_parameter_failure_count": forensics["pure_parameter_failure_count"], "non_parameter_structural_failure_count": forensics["non_parameter_structural_failure_count"], "c3_static_slot_interface_confounds": forensics["c3_static_slot_interface_confounds"], "top_parameter_fields": forensics["top_parameter_fields"], "top_error_subtypes": forensics["top_error_subtypes"], "wrong_parameter_fields_by_case": fields},
        "conditions": {"P0_S2_FAMILY_FIRST": {"formal_passes": 0, "source": "frozen historical S2"}, "P1_NORMALIZED_SLOT_EXTRACTION": {"formal_passes": 1, "family_and_skeleton": "frozen S2", "parameter_mode": "normalized semantic slots"}, "P2_CONSTRAINED_PARAMETER_CHOICE": {"formal_passes": 1, "family_and_skeleton": "frozen S2", "parameter_mode": "contract candidate choice"}, "P3_SLOTWISE_PARAMETER_SELECTION": {"formal_passes": 1, "family_and_skeleton": "frozen S2", "parameter_mode": "one constrained choice per slot"}, "P4_SELECTIVE_PARAMETER_REPAIR": {"formal_passes": 1, "family_and_skeleton": "frozen S2", "parameter_mode": "constrained extraction plus local patch gate"}},
        "protocol": {"arc_data_used": False, "arc_solutions_used": False, "new_arc_inference": False, "no_p0_rerun": True, "one_formal_pass_per_new_condition": True, "no_retry_failed_slots": True, "worker_gpu_assignment": {"P1_NORMALIZED_SLOT_EXTRACTION": 0, "P2_CONSTRAINED_PARAMETER_CHOICE": 1, "P3_SLOTWISE_PARAMETER_SELECTION": 2, "P4_SELECTIVE_PARAMETER_REPAIR": 3}, "startup": "sequential artifact warm-up, staggered worker initialization, READY barrier", "canonical_program_sent_to_model": False, "semantic_label_used_for_repair": False, "raw_responses_public": False},
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "FROZEN", "path": str(OUTPUT), "benchmark_hash": benchmark["benchmark_hash"], "parameter_contract_hash": payload["parameter_contract"]["sha256"]}))


if __name__ == "__main__":
    main()
