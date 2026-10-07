from __future__ import annotations

import json

import pytest

from novel_holdout_confirmation_v1.audit import (
    FAMILIES,
    composition_family,
    mcnemar_exact_p,
    paired_counts,
    prompt_pairs,
    select_sentinel,
    stratified_bootstrap,
    target_output,
    transfer_classification,
)


def test_exact_frozen_family_identities() -> None:
    assert FAMILIES == (
        "1d_arc:1d_mirror",
        "1d_arc:1d_move_2p",
        "compositional_arc:systematicity:grow+mirror+translation",
        "compositional_arc:systematicity:grow+rotation+translation",
    )


def test_prompt_extraction_does_not_decode_final_gold(monkeypatch: pytest.MonkeyPatch) -> None:
    marker = [[9, 9, 9]]
    payload = json.dumps({
        "train": [{"input": [[1]], "output": [[2]]}],
        "test": [{"input": [[3]], "output": marker}],
    }, separators=(",", ":"))
    original = json.loads

    def guarded(value, *args, **kwargs):
        assert json.dumps(marker, separators=(",", ":")) not in value
        return original(value, *args, **kwargs)

    monkeypatch.setattr(json, "loads", guarded)
    demos, final_input = prompt_pairs(payload, "1d_arc")
    assert demos == [{"input": [[1]], "output": [[2]]}]
    assert final_input == [[3]]


def test_gold_output_is_available_only_to_explicit_post_freeze_path() -> None:
    payload = json.dumps({"train": [], "test": [[[[3]], [[8]]]]}, separators=(",", ":"))
    assert target_output(payload, "1d_arc") == [[8]]


def test_compositional_metadata_family_uses_only_meta_data() -> None:
    payload = json.dumps({
        "meta_data": {"b": {"type": "translation"}, "a": {"type": "grow"}, "c": {"type": "mirror"}},
        "queries": [[[[1]], [[9]]]],
    }, separators=(",", ":"))
    assert composition_family(payload) == "compositional_arc:systematicity:grow+mirror+translation"


def test_sentinel_is_exactly_50_per_family_and_deterministic() -> None:
    candidates = []
    for family in FAMILIES:
        count = 50 if family.startswith("1d_arc:") else 60
        for index in range(count):
            candidates.append({"sample_id": f"{family}:{index}", "family": family, "source": family.split(":", 1)[0], "split": "holdout", "sequence_length": index + 1})
    left = select_sentinel(candidates)
    right = select_sentinel(reversed(candidates))
    assert [row["sample_id"] for row in left] == [row["sample_id"] for row in right]
    assert len(left) == 200
    assert {family: sum(row["family"] == family for row in left) for family in FAMILIES} == {family: 50 for family in FAMILIES}


def test_paired_counts_bootstrap_and_classification_are_frozen() -> None:
    rows = []
    for family in FAMILIES:
        for index in range(50):
            rows.append({"family": family, "base_exact": index < 10, "trained_exact": index < 20})
    counts = paired_counts([row["base_exact"] for row in rows], [row["trained_exact"] for row in rows])
    assert counts == {"n00": 120, "n01": 40, "n10": 0, "n11": 40}
    assert mcnemar_exact_p(40, 0) < 1e-10
    first = stratified_bootstrap(rows)
    second = stratified_bootstrap(rows)
    assert first == second
    assert first["point_estimate"] == pytest.approx(0.2)
    assert transfer_classification(0.2, 0.1, 0.3) == "STRONG_TRANSFER_CONFIRMED"
    assert transfer_classification(0.04, 0.01, 0.08) == "TRANSFER_CONFIRMED"
    assert transfer_classification(0.02, -0.01, 0.05) == "POSITIVE_BUT_INCONCLUSIVE"
    assert transfer_classification(-0.01, -0.04, 0.02) == "REGRESSION_WARNING"


def test_runner_has_zero_training_and_exact_generation_semantics() -> None:
    source = open("scripts/run_novel_holdout_confirmation_v1.py", encoding="utf-8").read()
    assert "do_sample=False" in source and "num_beams=1" in source
    assert "MAX_NEW_TOKENS = 932" in source and "REQUESTED_BATCH = 32" in source
    assert "prompt_ids[-3:] != [14, 12, 10]" in source
    assert ".backward(" not in source and "torch.optim" not in source
    assert '"optimizer_steps": 0' in source and '"backward_calls": 0' in source
