"""Freeze Q1--Q4 finite semantic retrieval before forward scoring begins."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm.parameter_candidate_scorer import scorer_config_hash
from llm.parameter_semantic_ontology import ParameterSemanticOntologyV1, stable_hash


OUTPUT = ROOT / "configs" / "PARAMETER_SEMANTIC_RETRIEVAL_V1_FROZEN_CONFIG.json"
INVENTORY = ROOT / "experiments" / "results" / "PARAMETER_CANDIDATE_INVENTORY_V1.json"


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"frozen config already exists: {OUTPUT}")
    if not INVENTORY.exists():
        raise FileNotFoundError("candidate inventory must be completed before configuration freeze")
    previous = read(ROOT / "configs" / "PARAMETER_GROUNDING_REPAIR_V1_FROZEN_CONFIG.json")
    benchmark = read(ROOT / "configs" / "macro_api_comprehension_benchmark_v1.json")
    inventory = read(INVENTORY); ontology = ParameterSemanticOntologyV1()
    if previous["p0_s2_baseline"]["semantic_success"] != 31 or previous["p0_s2_baseline"]["wrong_parameter"] != 17:
        raise ValueError("requires frozen S2 31/60 with 17 wrong-parameter cases")
    if inventory.get("case_count") != 17 or inventory.get("ontology_sha256") != ontology.sha256:
        raise ValueError("candidate inventory/ontology mismatch")
    aliases = {name: [candidate["aliases"] for candidate in values] for name, values in ontology.public()["parameters"].items()}
    alias_hash = stable_hash(aliases)
    semantic_scorer = ROOT / "src" / "llm" / "c3_semantic_intent.py"
    payload = {
        "experiment_id": "PARAMETER_SEMANTIC_RETRIEVAL_V1", "status": "FROZEN_BEFORE_ONE_FORMAL_PASS_PER_CONDITION",
        "benchmark_config": "configs/macro_api_comprehension_benchmark_v1.json", "benchmark_hash": benchmark["benchmark_hash"], "registry_hash": benchmark["registry_hash"], "schema_hash": benchmark["schema_hash"], "case_count": 60,
        "model": previous["model"], "tokenizer_version": "Qwen3-8B attached local tokenizer revision 1", "sampling": {"temperature": 0, "thinking": False, "generation_calls": 0},
        "scoring": {"method": "teacher-forced forward likelihood only", "normalization": "mean log probability per continuation token", "minimum_margin": 0.05, "source_b_config_sha256": scorer_config_hash(), "source_c_config_sha256": scorer_config_hash(), "source_d_rule": "Q3 requires B/C agreement and no conflict with a unique A; Q4 prioritizes unique A else B/C agreement", "source_d_config_sha256": stable_hash({"q3": "three_way_agreement", "q4": "ontology_priority_else_bc_agreement", "minimum_margin": 0.05})},
        "ontology": {"id": ontology.ontology_id, "sha256": ontology.sha256, "alias_map_sha256": alias_hash, "candidate_ids": "parameter.kind.value canonical IDs from registry/compiler choices", "frozen_public_ontology": ontology.public()},
        "candidate_inventory": {"path": "experiments/results/PARAMETER_CANDIDATE_INVENTORY_V1.json", "sha256": inventory["sha256"], "case_count": 17},
        "p0_s2_baseline": previous["p0_s2_baseline"], "s2_baseline_programs": previous["s2_baseline_programs"], "forensics": previous["forensics"],
        "semantic_scorer": {"module": "llm.c3_semantic_intent", "source_sha256": hashlib.sha256(semantic_scorer.read_bytes()).hexdigest(), "used_only_after_all_predictions_frozen": True, "llm_judge_used": False},
        "conditions": {"Q0_S2_FROZEN_BASELINE": {"formal_passes": 0}, "Q1_DETERMINISTIC_ONTOLOGY_REPAIR": {"formal_passes": 1, "source": "A"}, "Q2_LIKELIHOOD_PARAMETER_REPAIR": {"formal_passes": 1, "source": "B"}, "Q3_CONSENSUS_PARAMETER_REPAIR": {"formal_passes": 1, "source": "A+B+C"}, "Q4_CONSERVATIVE_HYBRID_REPAIR": {"formal_passes": 1, "source": "A priority, B/C consensus fallback"}},
        "protocol": {"arc_data_used": False, "arc_solutions_used": False, "new_arc_inference": False, "canonical_program_sent_to_model": False, "semantic_label_used_for_repair": False, "all_60_cases_same_logic": True, "one_formal_pass_per_condition": True, "no_retry": True, "no_threshold_sweep": True, "worker_gpu_assignment": {"Q1_DETERMINISTIC_ONTOLOGY_REPAIR": 0, "Q2_LIKELIHOOD_PARAMETER_REPAIR": 1, "Q3_CONSENSUS_PARAMETER_REPAIR": 2, "Q4_CONSERVATIVE_HYBRID_REPAIR": 3}, "startup": "sequential artifact warm-up, staggered worker initialization, READY barrier", "raw_completions": "none", "raw_scores_public": False},
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "FROZEN", "path": str(OUTPUT), "ontology_hash": ontology.sha256, "candidate_inventory_hash": inventory["sha256"]}))


if __name__ == "__main__":
    main()
