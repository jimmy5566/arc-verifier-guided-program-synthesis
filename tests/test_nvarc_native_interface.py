"""Pure, offline checks for the verified NVARC native representation."""
from __future__ import annotations

import ast
import json
from pathlib import Path

from inference.nvarc_native import native_messages, parse_native_grid, serialize_grid
from arc.task import ARCExample, ARCGrid, ARCTask


ROOT = Path(__file__).resolve().parents[1]


def _task() -> ARCTask:
    return ARCTask("hidden-id", (ARCExample(ARCGrid([[0, 1], [2, 3]]), ARCGrid([[3, 2], [1, 0]])),), (ARCExample(ARCGrid([[4, 5], [6, 7]])), ARCExample(ARCGrid([[8]]))))


def test_nvarc_serialization_is_exact_digit_rows_and_roundtrips() -> None:
    assert serialize_grid([[0, 1, 2], [3, 0, 4]]) == "012\n304"
    assert parse_native_grid("012\n304") == [[0, 1, 2], [3, 0, 4]]
    assert parse_native_grid("0 1\n23") is None
    assert parse_native_grid("[0,1]") is None


def test_native_parser_requires_rectangular_1_to_30_digit_rows() -> None:
    assert parse_native_grid("01\n2") is None
    assert parse_native_grid("") is None
    assert parse_native_grid("a1") is None
    assert parse_native_grid("0" * 31) is None


def test_native_messages_follow_official_alternating_train_and_independent_test_structure() -> None:
    task = _task()
    first, second = native_messages(task, 0), native_messages(task, 1)
    assert first == [{"role": "user", "content": "01\n23"}, {"role": "assistant", "content": "32\n10"}, {"role": "user", "content": "45\n67"}]
    assert second[-1] == {"role": "user", "content": "8"}
    assert "hidden-id" not in str(first)


def test_official_provenance_freezes_commit_and_required_hashes() -> None:
    manifest = json.loads((ROOT / "configs/NVARC_NATIVE_INTERFACE_846D0198_PROVENANCE.json").read_text(encoding="utf-8"))
    assert manifest["source_commit"] == "846d0198efa752534594e321fc3289fc0a06c657"
    assert set(manifest["files"]) == {"ARChitects/qwen3_configs/chat_template.j2", "ARChitects/qwen3_configs/tokenizer_config.json", "ARChitects/qwen3_configs/vocab.json", "SDG/scripts/build_datasets.py"}
    assert manifest["formal_inference_network"] == "OFFLINE"


def test_d2_config_corrects_d1_and_freezes_native_one_candidate_no_search_condition() -> None:
    config = json.loads((ROOT / "configs/ARC_SFT_NATIVE_PIPELINE_ABLATION_V1_FROZEN_CONFIG.json").read_text(encoding="utf-8"))
    assert config["D1"]["status"] == "INTERFACE_INCOMPATIBLE_CONTROL"
    assert config["D2"]["candidates_per_test_input"] == 1
    assert config["D2"]["no_ttt"] and config["D2"]["no_augmentation"] and config["D2"]["no_search"]


def test_native_runner_is_gold_blind_and_gate_prevents_search_before_d2_validity() -> None:
    source = (ROOT / "scripts/run_nvarc_native_d2.py").read_text(encoding="utf-8")
    imports = [node.module or "" for node in ast.walk(ast.parse(source)) if isinstance(node, ast.ImportFrom)]
    assert not any(any(word in module.lower() for word in ("evaluation", "recognition", "semantic", "capability", "compiler", "executor", "verifier", "search", "oracle")) for module in imports)
    assert "load_solutions" not in source and "PREDICTIONS_FROZEN_BEFORE_EXACT_SCORING" in source


def test_native_provider_uses_checkpoint_tokenizer_and_only_official_template_when_missing() -> None:
    source = (ROOT / "src/inference/nvarc_native.py").read_text(encoding="utf-8")
    assert "AutoTokenizer.from_pretrained(str(model_path), local_files_only=True" in source
    assert "tokenizer.add_special_tokens" not in source and "resize_token_embeddings" not in source
    assert "nvarc_official_for_missing_checkpoint_template" in source
    assert "token_mapping != _EXPECTED_TOKENS" in source


def test_native_scorer_opens_solutions_only_after_complete_prediction_freeze_check() -> None:
    source = (ROOT / "scripts/score_qwen4b_native_upstream_v1.py").read_text(encoding="utf-8")
    assert source.index("PREDICTIONS_FROZEN_BEFORE_EXACT_SCORING") < source.index("from arc.io import load_challenges, load_solutions")
    assert "v3." not in source and "executor" not in source and "verifier" not in source


def test_native_attachment_builder_isolated_from_v3_and_solution_data() -> None:
    source = (ROOT / "scripts/prepare_qwen4b_native_upstream_source.py").read_text(encoding="utf-8")
    assert '"src/v3"' not in source and '"*solutions*.json"' in source
    assert "native upstream source is not isolated" in source
