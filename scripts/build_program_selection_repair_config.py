"""Freeze S1--S4 before any new program-selection inference."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from llm.compiler_reachability import reachability_inventory
from llm.semantic_program_selector import MAX_CANDIDATES, prompt_hashes


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "configs" / "PROGRAM_SELECTION_REPAIR_V1_FROZEN_CONFIG.json"


def stable_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"frozen config already exists: {OUTPUT}")
    benchmark = read(ROOT / "configs" / "macro_api_comprehension_benchmark_v1.json")
    c3 = read(ROOT / "experiments" / "results" / "MACRO_API_COMPILER_AWARE_ABLATION_V1.json")
    audit = read(ROOT / "experiments" / "results" / "C3_SEMANTIC_INTENT_AUDIT_V1.json")
    if benchmark["case_count"] != 60 or sorted(benchmark["category_counts"].values()) != [15, 15, 15, 15]:
        raise ValueError("requires frozen 60-case API benchmark")
    if audit["metrics"]["semantic_success"] != 31 or c3["conditions"]["C3_STRUCTURED_COMPILER_CONSTRAINED"]["funnel"]["compile_valid"] != 60:
        raise ValueError("S0 must preserve audited C3 baseline")
    outcomes = {item["case_id"]: item["outcome"] for item in audit["case_assessments"]}
    if len(outcomes) != 60:
        raise ValueError("S0 semantic labels must cover exactly 60 cases")
    inventory = reachability_inventory(max_depth=3)
    payload = {
        "experiment_id": "PROGRAM_SELECTION_REPAIR_V1", "status": "FROZEN_BEFORE_ONE_FORMAL_PASS_PER_CONDITION",
        "benchmark_config": "configs/macro_api_comprehension_benchmark_v1.json", "benchmark_hash": benchmark["benchmark_hash"], "registry_hash": benchmark["registry_hash"], "schema_hash": benchmark["schema_hash"], "case_count": 60, "category_counts": benchmark["category_counts"],
        "model": c3["model"], "sampling": {"temperature": 0, "top_p": 1, "seed": 0, "thinking": False}, "generation": {"candidate_budget": 1, "max_new_tokens": 256, "context_window": 12288},
        "prompt_versions": {"S1_MINIMALITY_AWARE_SELECTION": "semantic_program_selector_v1.minimality", "S2_FAMILY_FIRST_SELECTION": "semantic_program_selector_v1.family_first", "S3_CONTRASTIVE_SKELETON_RANKING": "semantic_program_selector_v1.contrastive", "S4_FAMILY_FILTERED_CONTRASTIVE_MINIMALITY": "semantic_program_selector_v1.family_contrastive_minimality"}, "prompt_hashes": prompt_hashes(),
        "semantic_scorer": {"module": "llm.c3_semantic_intent", "source_sha256": hashlib.sha256((ROOT / "src" / "llm" / "c3_semantic_intent.py").read_bytes()).hexdigest(), "llm_judge_used": False},
        "compiler_inventory": {"sha256": stable_hash(inventory), "compiler_supported_macro_count": inventory["compiler_supported_macro_count"], "static_compiler_valid_skeleton_count": sum(item["compile_valid_chain_count"] for item in inventory["chain_inventory"].values())},
        "s0_frozen_baseline": {"source": "C3_SEMANTIC_INTENT_AUDIT_V1", "semantic_success": 31, "semantic_success_rate": 31 / 60, "compile_valid": 60, "over_transformed": 19, "wrong_operation": 8, "wrong_parameter": 2, "case_outcomes": outcomes, "audit_result_sha256": stable_hash(audit)},
        "conditions": {"S0_C3_STRUCTURED_COMPILER_CONSTRAINED": {"formal_passes": 0, "source": "frozen audit"}, "S1_MINIMALITY_AWARE_SELECTION": {"formal_passes": 1, "selection_principle": "minimal sufficient semantic operations"}, "S2_FAMILY_FIRST_SELECTION": {"formal_passes": 1, "stages": ["family enum", "family-filtered structured selection"]}, "S3_CONTRASTIVE_SKELETON_RANKING": {"formal_passes": 1, "stages": ["global deterministic K candidates", "rank", "structured parameter fill"]}, "S4_FAMILY_FILTERED_CONTRASTIVE_MINIMALITY": {"formal_passes": 1, "stages": ["family enum", "filtered K candidates", "minimality-aware rank", "structured parameter fill"]}},
        "protocol": {"arc_data_used": False, "arc_solutions_used": False, "new_arc_inference": False, "one_formal_pass_per_new_condition": True, "no_s0_rerun": True, "candidate_k": MAX_CANDIDATES, "no_canonical_prompt": True, "no_expected_family_prompt": True, "worker_gpu_assignment": {"S1_MINIMALITY_AWARE_SELECTION": 0, "S2_FAMILY_FIRST_SELECTION": 1, "S3_CONTRASTIVE_SKELETON_RANKING": 2, "S4_FAMILY_FILTERED_CONTRASTIVE_MINIMALITY": 3}, "startup": "sequential artifact warm-up, staggered worker initialization, READY barrier", "raw_responses_public": False},
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "FROZEN", "path": str(OUTPUT), "benchmark_hash": benchmark["benchmark_hash"], "candidate_k": MAX_CANDIDATES}))


if __name__ == "__main__":
    main()
