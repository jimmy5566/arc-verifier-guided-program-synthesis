"""Freeze R1--R3 before any contextual likelihood scoring occurs."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm.contextual_parameter_classifier import classifier_config_hash
from llm.contextual_parameter_ir import contextual_ir_schema, contextual_ir_schema_hash
from llm.parameter_semantic_ontology import ParameterSemanticOntologyV1, stable_hash
from llm.semantic_relation_normalizer import extractor_hash, relation_mapping_hash, relation_ontology_hash


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    output = ROOT / "configs/CONTEXTUAL_PARAMETER_REASONING_V1_FROZEN_CONFIG.json"
    if output.exists():
        raise FileExistsError(output)
    retrieval = read(ROOT / "configs/PARAMETER_SEMANTIC_RETRIEVAL_V1_FROZEN_CONFIG.json")
    benchmark = read(ROOT / "configs/macro_api_comprehension_benchmark_v1.json")
    forensics = read(ROOT / "experiments/results/CONTEXTUAL_PARAMETER_FORENSICS_V1.json")
    root = ROOT / "artifacts/parameter_semantic_retrieval_output_v22_full/checkpoints/parameter_semantic_retrieval_v1/Q1_DETERMINISTIC_ONTOLOGY_REPAIR"
    q1_programs = {item["case_id"]: {"family": retrieval["s2_baseline_programs"][item["case_id"]]["family"], "skeleton_id": retrieval["s2_baseline_programs"][item["case_id"]]["skeleton_id"], "program": item["program"]} for path in sorted(root.glob("case_*.json")) for item in [read(path)]}
    if len(q1_programs) != 60 or forensics["q1_baseline"]["remaining_pure_parameter_failures"] != 12:
        raise ValueError("requires complete frozen Q1 checkpoint and 12-case forensic")
    ontology = ParameterSemanticOntologyV1()
    payload = {
        "experiment_id": "CONTEXTUAL_PARAMETER_REASONING_V1",
        "status": "FROZEN_BEFORE_ONE_FORMAL_PASS_PER_R1_R2_R3",
        "benchmark_config": "configs/macro_api_comprehension_benchmark_v1.json",
        "benchmark_hash": benchmark["benchmark_hash"], "registry_hash": benchmark["registry_hash"], "schema_hash": benchmark["schema_hash"], "case_count": 60,
        "q1_baseline": {"semantic_success": 36, "remaining_pure_parameter_failures": 12, "programs_sha256": stable_hash(q1_programs), "checkpoint_source": "Q1_DETERMINISTIC_ONTOLOGY_REPAIR frozen checkpoint"},
        "q1_baseline_programs": q1_programs,
        "model": retrieval["model"], "tokenizer_version": retrieval["tokenizer_version"],
        "sampling": {"temperature": 0, "thinking": False, "generation_calls": 0, "candidate_budget": 1},
        "parameter_ontology": {"sha256": ontology.sha256},
        "relation_ontology": {"sha256": relation_ontology_hash()}, "semantic_feature_extractor": {"sha256": extractor_hash()},
        "contextual_ir": {"schema": contextual_ir_schema(), "sha256": contextual_ir_schema_hash()},
        "r1": {"mapping_sha256": relation_mapping_hash(), "rule": "global relation extraction + contract-scoped normalization only"},
        "r2": {"config_sha256": classifier_config_hash(), "minimum_margin": 0.05, "rule": "teacher-forced likelihood over supplied contextual IR candidate IDs"},
        "r3": {"gate_sha256": stable_hash({"priority": "R1_UNAMBIGUOUS_ELSE_R2_HIGH_DIFFERENT", "minimum_margin": 0.05, "default": "KEEP_Q1"}), "minimum_margin": 0.05, "rule": "R1 priority; otherwise R2 only when high margin and different; otherwise keep Q1"},
        "conditions": {"R0_Q1_HISTORICAL": {"formal_passes": 0}, "R1_RELATIONAL_NORMALIZER": {"formal_passes": 1}, "R2_CONTEXTUAL_CANDIDATE_CLASSIFIER": {"formal_passes": 1}, "R3_CONSERVATIVE_CONTEXTUAL_REPAIR": {"formal_passes": 1}},
        "protocol": {"arc_data_used": False, "arc_solutions_used": False, "new_arc_inference": False, "canonical_program_sent_to_model": False, "semantic_label_used_for_repair": False, "all_60_cases_same_logic": True, "one_formal_pass_per_condition": True, "no_retry": True, "no_threshold_sweep": True, "hardware": "exactly 4x NVIDIA L4", "startup": "sequential model artifact warm-up, staggered MODEL_READY barrier", "raw_completions": "none", "raw_scores_public": False},
    }
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "sha256": hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}))


if __name__ == "__main__":
    main()
