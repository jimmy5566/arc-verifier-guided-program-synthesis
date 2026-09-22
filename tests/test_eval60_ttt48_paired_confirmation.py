from __future__ import annotations

from pathlib import Path

from scripts.build_eval60_ttt48_paired_confirmation_kaggle import _frozen_inputs, _notebook
from scripts.score_eval60_ttt48_paired_confirmation import _bucket


def test_ttt48_frozen_contract_changes_only_steps() -> None:
    manifest, config, ttt24_candidates, ttt24_predictions, ttt24_report = _frozen_inputs()
    assert manifest["status"] == "EVAL60_TTT48_PAIRED_CONFIRMATION_COHORT_FROZEN"
    assert len(manifest["task_ids"]) == 60
    assert config["ttt_steps"] == 48
    assert manifest["ttt24_baseline"]["top2"] == 12
    assert ttt24_candidates["reference_config"]["ttt_steps"] == 24
    assert len(ttt24_predictions["records"]) == 60
    assert ttt24_report["TTT_TOP2"] == "12/60"


def test_paired_bucket_is_exhaustive() -> None:
    assert _bucket(True, True) == "BOTH"
    assert _bucket(False, True) == "NEW_48"
    assert _bucket(True, False) == "LOST_48"
    assert _bucket(False, False) == "UNSOLVED_BOTH"


def test_target_blind_runner_and_scoring_freeze_order() -> None:
    root = Path(__file__).resolve().parents[1]
    runner_source = (root / "scripts/run_eval60_ttt48_paired_confirmation.py").read_text(encoding="utf-8")
    score_source = (root / "scripts/score_eval60_ttt48_paired_confirmation.py").read_text(encoding="utf-8")
    assert "solutions_path" not in runner_source
    assert 'get_context("spawn")' in runner_source
    assert '"ttt_steps": 48' in runner_source
    assert score_source.index('atomic_write_json(args.output_dir / "predictions_frozen.json"') < score_source.index("solutions = _read(args.solutions_path)")


def test_notebook_requires_candidate_freeze_before_target_access() -> None:
    source = _notebook("jimmy5566/example")
    assert source.index("candidates=json.loads") < source.index("score=[sys.executable")
    assert "TRITON_PTXAS_PATH" in source
    assert "KAGGLE_KERNEL_INTERNET_ENABLED" in source
