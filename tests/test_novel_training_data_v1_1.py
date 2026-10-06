from __future__ import annotations

import pytest

from novel_training_data_v1_1.pipeline import (
    HierarchicalCurriculumSampler,
    aggregate_validation_losses,
    composition_signature,
    digest,
    family_probabilities,
    official_systematicity_protocol_status,
    _context_plan,
    official_episode_key,
)


def _rows():
    return [
        {"source": "a", "family": "a:small", "raw_episode_count": 100},
        {"source": "a", "family": "a:large", "raw_episode_count": 10_000},
        {"source": "b", "family": "b:one", "raw_episode_count": 500},
    ]


def test_systematicity_signature_is_parameter_independent() -> None:
    first = {
        "x": {"type": "translation", "kwargs": {"direction": "x"}},
        "y": {"type": "mirror", "kwargs": {"direction": "y"}},
        "z": {"type": "grow", "kwargs": {"direction": "-x"}},
    }
    second = {
        "a": {"type": "grow", "kwargs": {"direction": "-y"}},
        "b": {"type": "translation", "kwargs": {"direction": "y"}},
        "c": {"type": "mirror", "kwargs": {"direction": "x"}},
    }
    assert composition_signature(first) == composition_signature(second) == "grow+mirror+translation"
    assert official_episode_key(first, [1] * 11) == official_episode_key(first, [1] * 10)


def test_family_uniform_differs_from_row_uniform() -> None:
    baseline = family_probabilities(_rows(), "ROW_UNIFORM_BASELINE")
    balanced = family_probabilities(_rows(), "FAMILY_UNIFORM_GLOBAL")
    assert baseline["a:large"] / baseline["a:small"] == 100
    assert balanced["a:large"] == balanced["a:small"]


def test_source_weighting_and_sqrt_policy() -> None:
    source_uniform = family_probabilities(_rows(), "SOURCE_UNIFORM_THEN_FAMILY_UNIFORM")
    sqrt_policy = family_probabilities(_rows(), "SQRT_FAMILY_SOURCE_WEIGHT")
    assert source_uniform["b:one"] == 0.5
    assert source_uniform["a:small"] == source_uniform["a:large"] == 0.25
    assert sqrt_policy["a:small"] == sqrt_policy["a:large"]
    assert sum(sqrt_policy.values()) == pytest.approx(1.0)


def test_deterministic_hierarchical_sampling_and_holdout_impossible() -> None:
    episodes = {"POOL_NOVEL_V1_1": {"a": {"a:one": ["train-1", "train-2"]}}}
    first = HierarchicalCurriculumSampler(episodes, {"POOL_NOVEL_V1_1": 1.0}, "FAMILY_UNIFORM_GLOBAL", 7)
    second = HierarchicalCurriculumSampler(episodes, {"POOL_NOVEL_V1_1": 1.0}, "FAMILY_UNIFORM_GLOBAL", 7)
    a = [first.draw() for _ in range(100)]
    b = [second.draw() for _ in range(100)]
    assert a == b
    assert not {row[2] for row in a} & {"holdout-1"}


def test_macro_family_validation_is_not_micro_average() -> None:
    result = aggregate_validation_losses([
        {"source": "a", "family": "a:small", "loss": 1},
        {"source": "a", "family": "a:large", "loss": 3},
        {"source": "a", "family": "a:large", "loss": 5},
        {"source": "a", "family": "a:large", "loss": 7},
    ])
    assert result["micro_average_loss"] == 4.0
    assert result["macro_family_average_loss"] == 3.0


def test_portable_fingerprint_input_has_no_absolute_path_dependency() -> None:
    logical = {"schema": "v1", "ordered_shards": [{"logical_name": "train/a.parquet", "sha256": "x"}]}
    first = digest(logical)
    moved = dict(logical)
    assert digest(moved) == first


def test_context_length_audit_never_silently_truncates() -> None:
    plan = _context_plan({"pools": {"train": {
        "mean": 8500.0,
        "max": 8512,
        "overflow_8192": {"samples_affected": 99, "supervised_token_loss": 1234, "supervised_token_loss_rate": 0.04},
        "overflow_8704": {"samples_affected": 0, "supervised_token_loss": 0, "supervised_token_loss_rate": 0.0},
    }}})
    assert not plan["silent_truncation_allowed"]
    assert plan["configs"][0]["samples_affected"] == 99
    assert plan["configs"][1]["supervised_token_loss"] == 0


def test_official_systematicity_allows_shared_validation_holdout_ood_templates() -> None:
    systematicity = {
        "higher_level_composition_signature_overlap": {
            "train_validation": 0,
            "train_holdout": 0,
            "validation_holdout": 2,
        },
        "episode_identity_overlap": {
            "train_validation": 0,
            "train_holdout": 0,
            "validation_holdout": 0,
        },
        "transformation_composition_templates": {
            "train": ["train-a", "train-b"],
            "validation": ["ood-a", "ood-b"],
            "holdout": ["ood-a", "ood-b"],
        },
    }
    result = official_systematicity_protocol_status(systematicity)
    assert result["OFFICIAL_SYSTEMATICITY_PROTOCOL_READY"] is True
    assert result["STRICT_THREE_WAY_FAMILY_ISOLATION"] is False
    assert result["checks"]["validation_holdout_episode_identity_overlap_zero"] is True
    assert result["checks"]["validation_holdout_high_level_composition_overlap"] == "UPSTREAM_EXPECTED_SHARED_OOD_TEMPLATES"


def test_official_systematicity_rejects_train_ood_template_overlap() -> None:
    systematicity = {
        "higher_level_composition_signature_overlap": {
            "train_validation": 1,
            "train_holdout": 0,
            "validation_holdout": 1,
        },
        "episode_identity_overlap": {
            "train_validation": 0,
            "train_holdout": 0,
            "validation_holdout": 0,
        },
        "transformation_composition_templates": {
            "train": ["train-a", "ood-a"],
            "validation": ["ood-a"],
            "holdout": ["ood-a"],
        },
    }
    result = official_systematicity_protocol_status(systematicity)
    assert result["OFFICIAL_SYSTEMATICITY_PROTOCOL_READY"] is False


def test_official_systematicity_rejects_validation_holdout_episode_overlap() -> None:
    systematicity = {
        "higher_level_composition_signature_overlap": {
            "train_validation": 0,
            "train_holdout": 0,
            "validation_holdout": 2,
        },
        "episode_identity_overlap": {
            "train_validation": 0,
            "train_holdout": 0,
            "validation_holdout": 1,
        },
        "transformation_composition_templates": {
            "train": ["train-a"],
            "validation": ["ood-a", "ood-b"],
            "holdout": ["ood-a", "ood-b"],
        },
    }
    result = official_systematicity_protocol_status(systematicity)
    assert result["OFFICIAL_SYSTEMATICITY_PROTOCOL_READY"] is False
