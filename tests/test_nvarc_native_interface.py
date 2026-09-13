"""Pure, offline checks for the verified NVARC native representation."""
from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

from inference.arc_native_io import ARCNativeInputAdapter, ARCNativeOutputParser
from inference.nvarc_native_augmentation import NativeAugmentation, bounded_native_augmentations
from inference.nvarc_native_candidates import NativeGridCandidate, deduplicate_candidates, rank_candidates
from inference.native_ranker import CandidateRankingFeatures, rank_indices, select_method_from_pseudovalidation
from inference.native_train_verifier import rank_with_verifier, train_verifier_scores
from inference.native_multiview_likelihood import aggregate, calibrated, ranks
from inference.native_strategy_hypotheses import infer_strategies, score_predictions
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
    assert ARCNativeOutputParser.parse("01\n23\n") == [[0, 1], [2, 3]]
    for invalid in (" 01\n23", "01\n23\ntext", "[[0,1]]", "<|im_end|>", "01\r\n23"):
        assert ARCNativeOutputParser.parse(invalid) is None


def test_native_messages_follow_official_alternating_train_and_independent_test_structure() -> None:
    task = _task()
    first, second = native_messages(task, 0), native_messages(task, 1)
    assert first == [{"role": "user", "content": "01\n23"}, {"role": "assistant", "content": "32\n10"}, {"role": "user", "content": "45\n67"}]
    assert second[-1] == {"role": "user", "content": "8"}
    assert "hidden-id" not in str(first)
    assert ARCNativeInputAdapter().messages(task, 0) == first


def test_native_adapter_and_parser_have_no_solving_or_downstream_dependencies() -> None:
    source = (ROOT / "src/inference/arc_native_io.py").read_text(encoding="utf-8").lower()
    forbidden = ("rulespec", "solver", "heuristic", "semantic", "executor", "verifier", "candidate", "infer")
    assert not any(term in source for term in forbidden)


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


def test_native_augmentations_are_task_agnostic_reversible_and_bounded() -> None:
    grid = [[0, 1, 2], [3, 4, 5]]
    task = _task()
    pool = bounded_native_augmentations()
    assert len(pool) == 32
    for augmentation in pool:
        assert augmentation.inverse_grid(augmentation.transform_grid(grid)) == grid
        transformed = augmentation.transform_task(task)
        assert transformed.task_id == task.task_id
        assert len(transformed.train) == len(task.train) and len(transformed.test) == len(task.test)
    reversed_task = NativeAugmentation(pair_order="reversed").transform_task(task)
    assert reversed_task.train[0].input.to_list() == task.train[-1].input.to_list()


def test_native_candidates_deduplicate_and_rank_without_targets() -> None:
    identity = NativeAugmentation()
    shifted = NativeAugmentation(color_offset=1)
    first = NativeGridCandidate(identity, (((0, 1), (2, 3)),), 4, 0.1)
    duplicate = NativeGridCandidate(shifted, (((0, 1), (2, 3)),), 4, 0.1)
    second = NativeGridCandidate(shifted, (((3, 2), (1, 0)),), 4, 0.1)

    class Provider:
        def continuation_log_likelihood(self, messages, continuation, *, context_window):
            assert messages == [{"role": "user", "content": "01"}] and context_window == 99
            return {"01\n23": -3.0, "32\n10": -1.0}[continuation]

    unique = deduplicate_candidates([first, duplicate, second])
    assert [item.prediction for item in unique] == [first.prediction, second.prediction]
    assert len(unique[0].support_augmentations) == 2 and len(unique[1].support_augmentations) == 1
    ranked = rank_candidates(Provider(), unique, [[{"role": "user", "content": "01"}]], context_window=99)
    assert ranked[0][0].prediction == second.prediction and ranked[0][1] == -1.0


def test_native_capability_push_stays_native_and_gold_blind_until_scorer() -> None:
    runner = (ROOT / "scripts/run_qwen4b_native_augmentation_search.py").read_text(encoding="utf-8")
    scorer = (ROOT / "scripts/score_qwen4b_native_augmentation_search.py").read_text(encoding="utf-8")
    assert "load_solutions" not in runner and "ARCNativeInputAdapter" not in runner
    assert "RuleSpec" not in runner and "HardVerifier" not in runner
    assert scorer.index("CANDIDATES_AND_RANKED_PREDICTIONS_FROZEN_BEFORE_EXACT_SCORING") < scorer.index("from arc.io import load_challenges, load_solutions")
    assert "CANDIDATE_HEARTBEAT" in runner and "RUNNER_HEARTBEAT" in runner
    assert "gpu_utilization_pct" in runner and "nvidia-smi" in runner
    assert 'choices=("smoke", "pilot", "full", "external")' in runner


def test_native_ttt_uses_only_train_pairs_and_resets_per_task() -> None:
    source = (ROOT / "src/inference/nvarc_native_ttt.py").read_text(encoding="utf-8").lower()
    assert "augmented.train" in source and "task.test" not in source
    assert "self.reset(seed=self.config.seed)" in source and "training_pairs_only" in source
    assert "rulespec" not in source and "heuristic" not in source and "solver" not in source
    assert "parent._modules[part]" in source


def test_native_ttt_memory_hardening_freezes_base_and_records_phase_memory() -> None:
    source = (ROOT / "src/inference/nvarc_native_ttt.py").read_text(encoding="utf-8").lower()
    smoke = (ROOT / "scripts/run_native_ttt_memory_smoke.py").read_text(encoding="utf-8").lower()
    assert "for parameter in model.parameters(): parameter.requires_grad_(false)" in source
    assert "_assert_trainable_boundary" in source
    assert "gradient_checkpointing_enable" in source and "use_cache = false" in source
    assert "before_forward" in source and "after_backward" in source and "after_optimizer" in source
    assert "step_seed = self.config.seed + step" in source
    assert "foreach=false, fused=false" in source and "base_model_unchanged" in source
    assert "memory_safe" in smoke and "deterministic_reset" in smoke
    assert "no test candidate generation" in smoke and "task.test" not in smoke


def test_native_capability_stages_are_fixed_nested_scale_gates() -> None:
    config = json.loads((ROOT / "configs/QWEN4B_MAX_NATIVE_CAPABILITY_PUSH_V1.json").read_text(encoding="utf-8"))
    assert config["stages"] == {
        "smoke": {"task_count": 2, "augmentation_count": 4, "worker_count": 1},
        "pilot": {"task_count": 5, "augmentation_count": 8, "worker_count": 2},
        "full": {"task_count": 30, "augmentation_count": 32, "worker_count": 4},
    }


def test_label_free_native_ranker_features_and_pseudovalidation_tie_break() -> None:
    rows = [
        CandidateRankingFeatures(0, -1.0, 3, 2, 2, 2),
        CandidateRankingFeatures(1, -0.1, 1, 1, 1, 1),
    ]
    ranks = rank_indices(rows)
    assert ranks["likelihood"] == [1, 0]
    assert ranks["consensus_frequency"] == [0, 1]
    assert ranks["augmentation_diversity_consensus"] == [0, 1]
    assert select_method_from_pseudovalidation({"likelihood": 2, "consensus_frequency": 2}) == "likelihood"


def test_ranker_forensics_and_pseudo_scorer_keep_oracle_boundaries() -> None:
    forensics = (ROOT / "scripts/analyze_native_ranker_frozen.py").read_text(encoding="utf-8")
    pseudo = (ROOT / "scripts/score_native_ranker_pseudovalidation.py").read_text(encoding="utf-8")
    assert forensics.index("CANDIDATES_AND_RANKED_PREDICTIONS_FROZEN_BEFORE_EXACT_SCORING") < forensics.index("from arc.io import load_challenges, load_solutions")
    assert "load_solutions" not in pseudo and "solutions-path" not in pseudo
    assert "train_pair" in pseudo.lower() and "selected_ranker" in pseudo


def test_train_only_verifier_scores_generic_pair_relations_and_fuses_likelihood() -> None:
    pairs = [
        ([[0, 1], [0, 0]], [[1, 0], [0, 0]]),
        ([[0, 2], [0, 0]], [[2, 0], [0, 0]]),
    ]
    scores = train_verifier_scores(pairs, [[0, 3], [0, 0]], [[[3, 0], [0, 0]], [[3, 3], [0, 0]]])
    assert len(scores) == 2 and scores[0] > scores[1]
    ranks = rank_with_verifier([-2.0, -0.1], scores)
    assert ranks["likelihood"] == [1, 0] and ranks["verifier"] == [0, 1]


def test_train_verifier_oracle_boundary_is_after_rerank_freeze() -> None:
    rerank = (ROOT / "scripts/rerank_native_with_train_verifier.py").read_text(encoding="utf-8")
    scorer = (ROOT / "scripts/score_native_train_verifier.py").read_text(encoding="utf-8")
    assert "load_solutions" not in rerank
    assert scorer.index("CANDIDATES_RERANKED_BY_TRAIN_ONLY_VERIFIER_FROZEN_BEFORE_EXACT_SCORING") < scorer.index("from arc.io import load_solutions")


def test_combined_native_pool_is_target_blind_and_deduplicates_predictions(tmp_path: Path) -> None:
    def artifact(candidates: list[list[list[int]]], scores: list[float]) -> dict[str, object]:
        return {
            "status": "CANDIDATES_AND_RANKED_PREDICTIONS_FROZEN_BEFORE_EXACT_SCORING",
            "records": {
                "unseen": {
                    "candidates": [{"prediction": [grid]} for grid in candidates],
                    "ranked_candidate_indices": list(range(len(candidates))),
                    "candidate_scores": scores,
                }
            },
        }
    baseline, ttt, output = tmp_path / "baseline.json", tmp_path / "ttt.json", tmp_path / "combined.json"
    baseline.write_text(json.dumps(artifact([[[0]], [[1]]], [-2.0, -1.0])), encoding="utf-8")
    ttt.write_text(json.dumps(artifact([[[1]], [[2]]], [-3.0, -0.1])), encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts/combine_native_candidate_artifacts.py"), "--baseline", str(baseline), "--ttt", str(ttt), "--output", str(output)],
        check=True, capture_output=True, text=True,
    )
    combined = json.loads(output.read_text(encoding="utf-8")); record = combined["records"]["unseen"]
    assert "load_solutions" not in (ROOT / "scripts/combine_native_candidate_artifacts.py").read_text(encoding="utf-8")
    assert combined["status"] == "CANDIDATES_COMBINED_FROZEN_BEFORE_TRAIN_VERIFIER_RERANK"
    assert len(record["candidates"]) == 3 and record["candidates"][1]["combined_provenance"] == ["baseline", "ttt"]
    assert "candidate_count" in completed.stdout


def test_multiview_robust_aggregation_and_calibrated_ranking_are_deterministic() -> None:
    values = [-4.0, -1.0, -1.0, -40.0]
    assert aggregate(values, "mean") < aggregate(values, "median")
    assert aggregate(values, "trimmed_mean") == -2.5
    scores = {"original_likelihood": [-2.0, -1.0], "calibrated_likelihood": [calibrated([-1.0, -8.0], [0.9, 0.1]), calibrated([-2.0, -1.0], [0.9, 0.1])]} 
    assert ranks(scores)["original_likelihood"] == [1, 0]
    assert ranks(scores)["calibrated_likelihood"] == [0, 1]


def test_strategy_hypotheses_are_train_only_and_score_generic_transform_and_recolor() -> None:
    transform = ARCTask("not-used", (ARCExample(ARCGrid([[1, 2], [3, 4]]), ARCGrid([[3, 1], [4, 2]])), ARCExample(ARCGrid([[5, 6], [7, 8]]), ARCGrid([[7, 5], [8, 6]]))), (ARCExample(ARCGrid([[1, 0], [2, 3]])),))
    hypothesis = infer_strategies(transform)
    assert "rot270" in hypothesis.global_transforms
    assert score_predictions(hypothesis, transform, [[[2, 1], [3, 0]]]) > score_predictions(hypothesis, transform, [[[1, 0], [2, 3]]])
    recolor = ARCTask("not-used", (ARCExample(ARCGrid([[1, 2]]), ARCGrid([[3, 4]])), ARCExample(ARCGrid([[2, 1]]), ARCGrid([[4, 3]]))), (ARCExample(ARCGrid([[1, 2]])),))
    recolor_hypothesis = infer_strategies(recolor)
    assert recolor_hypothesis.color_map == ((1, 3), (2, 4))
    assert score_predictions(recolor_hypothesis, recolor, [[[3, 4]]]) > score_predictions(recolor_hypothesis, recolor, [[[1, 2]]])
    source = (ROOT / "src/inference/native_strategy_hypotheses.py").read_text(encoding="utf-8").lower()
    assert "solutions" not in source and "task_id" not in source
