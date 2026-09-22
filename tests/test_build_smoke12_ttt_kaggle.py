from __future__ import annotations

from scripts.build_smoke12_ttt_kaggle import _CONDITIONS, _runner_config


def test_smoke12_package_conditions_only_change_declared_ablation_variable() -> None:
    reference = {
        "rank": 256,
        "alpha": 32,
        "ttt_steps": 24,
        "generation_augmentation_count": 8,
        "target_modules": ["q_proj"],
    }
    s1, s2 = _runner_config("ttt24_beam2", reference), _runner_config("ttt48_greedy", reference)
    assert s1["reference_ttt_base"] == s2["reference_ttt_base"] == reference
    assert s1["conditions"] == {"ttt24_beam2": {"ttt_steps": 24, "decode": "beam", "beam_width": 2}}
    assert s2["conditions"] == {"ttt48_greedy": {"ttt_steps": 48, "decode": "greedy", "beam_width": 1}}
    assert set(_CONDITIONS) == {"ttt24_beam2", "ttt48_greedy"}
