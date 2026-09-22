from __future__ import annotations

from pathlib import Path

from scripts.build_eval60_per_output_rescore_kaggle import _inputs, _notebook
from scripts.score_eval60_per_output_rescore import _select


def test_frozen_per_output_inputs_are_complete() -> None:
    inputs = _inputs()
    assert set(inputs) == {"eval60_manifest.json", "ttt24_candidates_frozen.json", "ttt24_predictions_frozen.json", "ttt48_candidates_frozen.json", "ttt48_predictions_frozen.json"}


def test_per_output_selector_preserves_fixed_b_support_formula() -> None:
    entries = [
        {"candidate_index": 0, "original_log_likelihood": -2.0, "view_negative_log_likelihoods": [2.0] * 8, "support_count": 1},
        {"candidate_index": 1, "original_log_likelihood": -1.0, "view_negative_log_likelihoods": [1.0] * 8, "support_count": 1},
    ]
    result, first, second = _select(entries, [{"prediction": [[1]]}, {"prediction": [[2]]}])
    assert result["attempt_candidate_indices"] == [1, 0]
    assert first == [[2]] and second == [[1]]


def test_runner_remains_target_blind_and_has_no_generation() -> None:
    source = (Path(__file__).resolve().parents[1] / "scripts/run_eval60_per_output_rescore.py").read_text(encoding="utf-8")
    assert "solutions_path" not in source
    assert "_generate_aug8" not in source
    assert 'get_context("spawn")' in source
    assert "per_output_evidence" in source


def test_notebook_enforces_evidence_freeze_before_scoring() -> None:
    source = _notebook("jimmy5566/example")
    assert source.index("per_output_evidence_frozen.json") < source.index("score=[sys.executable")
    assert "TRITON_PTXAS_PATH" in source
    assert "KAGGLE_KERNEL_INTERNET_ENABLED" in source
