from __future__ import annotations

from scripts.build_eval60_reference_ttt_4gpu_kaggle import _frozen_inputs


def test_eval60_reference_ttt_cohort_and_config_are_frozen() -> None:
    manifest, baseline, config = _frozen_inputs()
    assert len(manifest["task_ids"]) == 60
    assert set(baseline["records"]) == set(manifest["task_ids"])
    assert manifest["baseline"] == {"any_of_k": 2, "top1": 2, "top2": 2, "pool_miss_count": 58}
    assert config["rank"] == 256 and config["alpha"] == 32 and config["ttt_steps"] == 24
    assert config["ptxas_path"] == "/usr/local/cuda-12.5/bin/ptxas"


def test_eval60_runner_is_target_blind_and_uses_dynamic_four_worker_queue() -> None:
    source = open("scripts/run_eval60_reference_ttt_4gpu.py", encoding="utf-8").read().lower()
    assert "solutions_path" not in source and "load_solutions" not in source
    assert "get_context(\"spawn\")" in source and "for worker_id in range(4)" in source
    assert "work.get()" in source and "atomic_write_json(checkpoint" in source
    assert "adapter_reset_success" in source and "candidate_view_scores_many" in source
    assert "beam" not in source and "dfs" not in source


def test_eval60_scorer_persists_predictions_before_solution_boundary() -> None:
    source = open("scripts/score_eval60_reference_ttt_4gpu.py", encoding="utf-8").read()
    assert 'atomic(args.output_dir / "predictions_frozen.json", prediction_artifact)' in source
    assert "solutions = _read(args.solutions_path)" in source
    assert source.index('atomic(args.output_dir / "predictions_frozen.json", prediction_artifact)') < source.index("solutions = _read(args.solutions_path)")
    assert "grouped_public_reference_ranking" in source


def test_eval60_scorer_keeps_empty_candidate_pool_as_a_target_blind_pool_miss() -> None:
    from scripts.score_eval60_reference_ttt_4gpu import _select

    selection, first, second = _select({"task_id": "empty", "status": "NO_VALID_NATIVE_CANDIDATE", "candidates": []})
    assert selection["status"] == "NO_VALID_NATIVE_CANDIDATE"
    assert selection["attempt_candidate_indices"] == []
    assert first is None and second is None


def test_eval60_conservative_projection_does_not_apply_a_minutes_multiplier() -> None:
    source = open("scripts/score_eval60_reference_ttt_4gpu.py", encoding="utf-8").read()
    assert "max_worker_load * 4.0" in source
    assert "* 60.0" not in source
