"""Frozen-boundary tests for the ARC-SFT checkpoint-only ablation."""
from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COHORT_HASH = "a2f8fb66af4b83ec09bd2f3f2bf3fc5e948a6f7a6839898b5dbf26c6f77d1fc8"


def test_specialized_config_reuses_the_exact_direct_v1_cohort_and_d0_artifact() -> None:
    config = json.loads((ROOT / "configs/ARC_SPECIALIZED_MODEL_ABLATION_V1_FROZEN_CONFIG.json").read_text(encoding="utf-8"))
    d0 = json.loads((ROOT / "configs/DIRECT_GRID_SOLVE_BASELINE_V1_FROZEN_CONFIG.json").read_text(encoding="utf-8"))
    assert config["cohort"]["task_ids_hash"] == d0["cohort"]["task_ids_hash"] == COHORT_HASH
    assert config["D0_reference"]["status"] == "REUSED_NOT_RERUN"
    assert config["D0_reference"]["prediction_freeze_sha256"] == "cdb70eccd744f1153c5dfb3c98d086442eb8f9426487008dd30619557486cfe0"


def test_d1_preserves_d0_decoding_contract_and_changes_only_checkpoint() -> None:
    config = json.loads((ROOT / "configs/ARC_SPECIALIZED_MODEL_ABLATION_V1_FROZEN_CONFIG.json").read_text(encoding="utf-8"))
    d0 = json.loads((ROOT / "configs/DIRECT_GRID_SOLVE_BASELINE_V1_FROZEN_CONFIG.json").read_text(encoding="utf-8"))
    for name in ("temperature", "top_p", "seed", "thinking", "max_new_tokens", "context_window"):
        assert config["generation"][name] == d0["generation"][name]
    assert config["generation"]["calls_per_task"] == 1 and config["generation"]["maximum_attempts"] == 2
    assert "byte-identical D0 raw-grid prompt" in config["D1_ARC_SFT_CHECKPOINT_ONLY"]["prompt_interface"]
    assert config["model"]["model_source"] == "sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1"


def test_native_interface_is_explicitly_unresolved_not_an_invented_recipe() -> None:
    config = json.loads((ROOT / "configs/ARC_SPECIALIZED_MODEL_ABLATION_V1_FROZEN_CONFIG.json").read_text(encoding="utf-8"))
    assert config["D2_ARC_SFT_NATIVE_INTERFACE"]["status"] == "NOT_RUN_NATIVE_INTERFACE_UNRESOLVED"
    assert "No native ARC serialization" in config["D2_ARC_SFT_NATIVE_INTERFACE"]["reason"]


def test_d1_runner_keeps_gold_blind_import_boundary_with_local_template_fallback_only() -> None:
    source = (ROOT / "scripts/run_direct_grid_solve_baseline_v1.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    forbidden = ("evaluation", "recognition", "semantic", "capability", "compiler", "executor", "verifier", "search", "oracle")
    assert not any(any(word in name.lower() for word in forbidden) for name in imports)
    assert "chat-template-fallback-model-path" in source
    assert "load_solutions" not in source


def test_specialized_finalizer_checks_frozen_artifacts_before_target_import() -> None:
    source = (ROOT / "scripts/finalize_arc_specialized_model_ablation_v1.py").read_text(encoding="utf-8")
    assert source.index("d0, d0_hash = _frozen") < source.index("from arc.io import load_challenges, load_solutions")
    assert "PREDICTIONS_FROZEN_BEFORE_EXACT_SCORING" in source
    assert "NOT_RUN_DEPENDS_ON_D2_NATIVE_INTERFACE" in source


def test_specialized_aggregate_records_unresolved_dependent_conditions_without_fabricated_metrics() -> None:
    result = json.loads((ROOT / "experiments/results/ARC_SPECIALIZED_MODEL_ABLATION_V1.json").read_text(encoding="utf-8"))
    assert result["D1"]["metrics"]["any_of_2_exact"] == 0.0
    assert result["D2"]["status"] == "NOT_RUN_NATIVE_INTERFACE_UNRESOLVED"
    assert result["D3"]["status"] == "NOT_RUN_DEPENDS_ON_D2_NATIVE_INTERFACE"
    assert result["D4"]["status"] == "NOT_RUN_DEPENDS_ON_D3"
    assert result["D5"]["status"] == "NOT_RUN_NO_D4_CANDIDATE_POOL"
