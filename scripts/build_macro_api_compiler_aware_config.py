"""Freeze the single-pass C1--C3 compiler-aware API experiment."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from llm.compiler_aware_interface import compiler_valid_skeletons, prompt_hashes
from llm.compiler_reachability import reachability_inventory


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "configs" / "MACRO_API_COMPILER_AWARE_ABLATION_V1_FROZEN_CONFIG.json"
BENCHMARK = ROOT / "configs" / "macro_api_comprehension_benchmark_v1.json"
R2_RESULT = ROOT / "experiments" / "results" / "MACRO_API_REPRESENTATION_ABLATION_V1.json"


def stable_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"frozen config already exists: {OUTPUT}")
    benchmark, r2_result = read(BENCHMARK), read(R2_RESULT)
    r2 = r2_result["conditions"]["R2_TYPED_EXAMPLES"]
    if benchmark["case_count"] != 60 or sorted(benchmark["category_counts"].values()) != [15, 15, 15, 15]:
        raise ValueError("requires the frozen 60-case 15/15/15/15 API benchmark")
    if r2["funnel"]["type_valid"] != 50 or r2["funnel"]["compile_valid"] != 14:
        raise ValueError("C0 must reference the frozen R2 50 type-valid / 14 compile-valid result")
    base_model = r2_result["model"]
    inventory = reachability_inventory(max_depth=3)
    payload = {
        "experiment_id": "MACRO_API_COMPILER_AWARE_ABLATION_V1",
        "status": "FROZEN_BEFORE_ONE_FORMAL_PASS_PER_CONDITION",
        "benchmark_config": "configs/macro_api_comprehension_benchmark_v1.json",
        "benchmark_hash": benchmark["benchmark_hash"], "registry_hash": benchmark["registry_hash"], "schema_hash": benchmark["schema_hash"],
        "case_count": 60, "category_counts": benchmark["category_counts"],
        "model": base_model,
        "sampling": {"temperature": 0, "top_p": 1, "seed": 0, "thinking": False},
        "generation": {"candidate_budget": 1, "max_new_tokens": 256, "context_window": 12288},
        "prompt_versions": {
            "C1_COMPILER_AWARE_CATALOGUE": "compiler_aware_interface_v1.catalogue",
            "C2_COMPILER_VALID_SKELETON": "compiler_aware_interface_v1.skeleton_selection",
            "C2_PARAMETER_FILL": "compiler_aware_interface_v1.skeleton_parameter_fill",
            "C3_STRUCTURED_COMPILER_CONSTRAINED": "compiler_aware_interface_v1.structured_constraints",
        },
        "prompt_hashes": prompt_hashes(),
        "compiler_reachability": {
            "static_scope": "direct literal witnesses with zero ARC train pairs",
            "inventory_hash": stable_hash(inventory),
            "compiler_supported_macro_count": inventory["compiler_supported_macro_count"],
            "type_declared_macro_count": inventory["type_declared_macro_count"],
            "compiler_valid_skeleton_count": len(compiler_valid_skeletons()),
            "max_skeleton_depth": 3,
        },
        "c0_reference": {
            "condition": "R2_TYPED_EXAMPLES", "source_experiment": "MACRO_API_REPRESENTATION_ABLATION_V1",
            "source_result_sha256": stable_hash(r2_result), "summary": r2,
        },
        "conditions": {
            "C0_R2_TYPED_EXAMPLES": {"formal_passes": 0, "source": "frozen historical R2 aggregate"},
            "C1_COMPILER_AWARE_CATALOGUE": {"formal_passes": 1, "catalogue": "compiler-supported Macro contracts and static direct-literal forms only"},
            "C2_COMPILER_VALID_SKELETON": {"formal_passes": 1, "stage1": "select one globally enumerated compiler-valid skeleton", "stage2": "fill parameters without changing skeleton"},
            "C3_STRUCTURED_COMPILER_CONSTRAINED": {"formal_passes": 1, "generation": "one structured skeleton/slot JSON response", "materialization": "deterministic literal wrappers only after constraint validation"},
        },
        "protocol": {
            "arc_data_used": False, "arc_solutions_used": False, "task_prediction": False,
            "one_formal_pass_per_new_condition": True, "no_r2_rerun": True,
            "worker_gpu_assignment": {"C1_COMPILER_AWARE_CATALOGUE": [0], "C2_COMPILER_VALID_SKELETON": [1], "C3_STRUCTURED_COMPILER_CONSTRAINED": [2, 3]},
            "startup": "sequential model artifact warm-up, staggered worker load, then barrier",
            "raw_responses_public": False,
        },
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "FROZEN", "path": str(OUTPUT), "benchmark_hash": payload["benchmark_hash"], "skeleton_count": len(compiler_valid_skeletons())}))


if __name__ == "__main__":
    main()
