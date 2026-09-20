from __future__ import annotations

import hashlib

from scripts.build_eval6_reference_ttt_kaggle import _frozen_inputs
from scripts.run_eval6_reference_ttt import _assistant_labels, _reference_variants


def test_reference_eval6_is_deterministic_and_uses_documented_ttt_settings() -> None:
    manifest, baseline, config = _frozen_inputs()
    ids = manifest["task_ids"]
    assert len(ids) == 6 and ids == sorted(ids, key=lambda value: (hashlib.sha256(value.encode()).hexdigest(), value))
    assert set(baseline["records"]) == set(ids)
    assert config["rank"] == 256 and config["alpha"] == 32 and config["learning_rate"] == 5e-5
    assert config["augmentation_colour_permutations"] == 16 and config["generation"] == "unchanged greedy Aug8"


def test_reference_assistant_mask_keeps_only_assistant_grid_and_eos() -> None:
    import torch
    ids = torch.tensor([11, 10, 1, 15, 12, 10, 2, 15])
    labels = _assistant_labels(ids)
    assert labels.tolist() == [-100, -100, -100, -100, -100, -100, 2, 15]


def test_reference_runner_isolated_from_targets_and_search() -> None:
    source = open("scripts/run_eval6_reference_ttt.py", encoding="utf-8").read()
    assert "load_solutions" not in source and "beam" not in source.lower() and "dfs" not in source.lower()
