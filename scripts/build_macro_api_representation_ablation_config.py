"""Freeze R1--R4 API representation ablation before any model call."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "src"))

from llm.macro_interface_v2_1 import prompt_hashes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "configs" / "MACRO_API_REPRESENTATION_ABLATION_V1_FROZEN_CONFIG.json")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to alter frozen config: {args.output}")
    benchmark = json.loads((ROOT / "configs" / "macro_api_comprehension_benchmark_v1.json").read_text(encoding="utf-8"))
    config = {
        "experiment_id": "MACRO_API_REPRESENTATION_ABLATION_V1",
        "status": "FROZEN_BEFORE_ONE_FORMAL_PASS_PER_CONDITION",
        "benchmark_config": "configs/macro_api_comprehension_benchmark_v1.json",
        "benchmark_hash": benchmark["benchmark_hash"], "registry_hash": benchmark["registry_hash"], "schema_hash": benchmark["schema_hash"],
        "case_count": 60, "category_counts": benchmark["category_counts"],
        "model": {"backend": "transformers_local", "model_source": "qwen-lm/qwen-3/Transformers/8b/1", "architecture": "Qwen3ForCausalLM", "dtype": "bfloat16", "local_files_only": True, "internet": False},
        "sampling": {"temperature": 0, "top_p": 1, "seed": 0, "thinking": False},
        "generation": {"candidate_budget": 1, "max_new_tokens": 256, "context_window": 12288},
        "prompt_versions": {"R1_TYPED_COMPACT": "macro_interface_v2_1.typed_compact", "R2_TYPED_EXAMPLES": "macro_interface_v2_1.typed_examples", "R3_STAGE1_SKELETON": "macro_interface_v2_1.two_stage_skeleton", "R3_STAGE2_PARAMETER_FILL": "macro_interface_v2_1.two_stage_parameter_fill", "R4_ONE_REPAIR": "verifier_guided_api_repair_v1.one_repair"},
        "prompt_hashes": prompt_hashes(),
        "conditions": {"R1_TYPED_COMPACT": {"formal_passes": 1}, "R2_TYPED_EXAMPLES": {"formal_passes": 1}, "R3_TWO_STAGE_TYPED": {"formal_passes": 1, "stage1": "skeleton_only", "stage2": "frozen_skeleton_parameter_fill"}, "R4_VERIFIER_GUIDED_API_REPAIR": {"formal_passes": 1, "base": "R2 raw output", "max_repairs": 1, "feedback": "first_blocking_error_plus_relevant_contract"}},
        "r0_frozen_baseline": {"name": "CURRENT_V2_REPRESENTATION", "schema_valid_rate": 0.20, "type_valid_rate": 0.05, "compile_valid_rate": 0.0, "formal_runs": 0, "source": "MACRO_API_COMPREHENSION_BENCHMARK_V1"},
        "protocol": {"arc_data_used": False, "arc_solutions_used": False, "task_prediction": False, "one_formal_pass_per_condition": True, "max_repairs": 1, "worker_gpu_assignment": {"R1_TYPED_COMPACT": 0, "R2_TYPED_EXAMPLES": 1, "R3_TWO_STAGE_TYPED": 2, "R4_VERIFIER_GUIDED_API_REPAIR": 3}, "startup": "sequential model artifact warm-up, staggered worker load, then barrier", "raw_responses_public": False},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "benchmark_hash": config["benchmark_hash"], "prompt_hashes": config["prompt_hashes"]}))


if __name__ == "__main__":
    main()
