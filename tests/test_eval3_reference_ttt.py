from __future__ import annotations

import hashlib

from scripts.build_eval3_reference_ttt_kaggle import _frozen_inputs
from scripts.run_eval3_reference_ttt import _assistant_labels, _full_dialogue, _reference_variants, _training_pair


def test_eval3_uses_exactly_three_deterministic_pool_misses_and_fixed_ttt() -> None:
    manifest, baseline, config = _frozen_inputs()
    ids = manifest["task_ids"]
    assert len(ids) == 3
    assert ids == sorted(ids, key=lambda value: (hashlib.sha256(value.encode()).hexdigest(), value))
    assert set(baseline["records"]) == set(ids)
    assert config["rank"] == 256 and config["alpha"] == 32 and config["ttt_steps"] == 24
    assert config["generation"] == "unchanged greedy Aug8; beam=1"


def test_eval3_assistant_mask_preserves_only_assistant_completion() -> None:
    import torch

    token_ids = torch.tensor([11, 10, 1, 15, 12, 10, 2, 15])
    assert _assistant_labels(token_ids).tolist() == [-100, -100, -100, -100, -100, -100, 2, 15]


def test_eval3_ttt_dialogue_uses_the_existing_trusted_internal_serializer() -> None:
    from pathlib import Path

    from arc.io import load_dataset
    from inference.arc_native_io import ARCNativeInputAdapter

    task = load_dataset(Path("data/raw/arc-agi_evaluation_challenges.json"))["5dbc8537"]
    dialogue = _full_dialogue(_reference_variants(task)[0], ARCNativeInputAdapter.serialize_trusted_grid)
    assert dialogue.startswith("<|im_start|>user\n") and "<|im_start|>assistant\n" in dialogue


def test_eval3_runner_is_target_blind_and_does_not_search() -> None:
    source = open("scripts/run_eval3_reference_ttt.py", encoding="utf-8").read().lower()
    assert "solutions_path" not in source
    assert "beam" not in source
    assert "dfs" not in source
    assert "tokenizer = native_tokenizer" in source


def test_eval3_scorer_can_run_as_a_script_from_the_project_root() -> None:
    source = open("scripts/score_eval3_reference_ttt.py", encoding="utf-8").read()
    assert "sys.path.insert(0, str(ROOT))" in source
    assert "sum(ttt[\"step_seconds\"])" in source


def test_filtered_training_inputs_and_labels_keep_same_sequence() -> None:
    import torch
    original = [torch.tensor([11, 10, 1, 15, 12, 10, 2, 15]), torch.ones(20, dtype=torch.long), torch.tensor([11, 10, 3, 15, 12, 10, 4, 15])]
    kept = [item for item in original if len(item) <= 8]
    labels = [_assistant_labels(item) for item in kept]
    for step in range(5):
        ids, target = _training_pair(kept, labels, step)
        assert ids is kept[step % 2] and target is labels[step % 2]
    all_labels = [_assistant_labels(item) for item in (original[0], original[2])]
    assert _training_pair([original[0], original[2]], all_labels, 1)[0] is original[2]
