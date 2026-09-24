from __future__ import annotations

from scripts.run_eval3_runtime_opt import _left_pad, _record_fingerprint, compare_target_blind


def _record(*, task_id: str, loss: list[float], raw: list[dict], candidates: list[dict], invalid: int = 0) -> dict:
    return {
        "task_id": task_id,
        "ttt": {"loss_curve": loss},
        "raw_views": raw,
        "candidates": candidates,
        "invalid_candidate_count": invalid,
        "unique_candidate_count": len(candidates),
    }


def test_left_padding_preserves_each_prompt_suffix() -> None:
    import torch

    packed, lengths = _left_pad(
        [
            {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.tensor([[1, 1, 1]])},
            {"input_ids": torch.tensor([[4, 5]]), "attention_mask": torch.tensor([[1, 1]])},
        ],
        pad_token_id=0,
    )
    assert lengths == [3, 2]
    assert packed["input_ids"].tolist() == [[1, 2, 3], [0, 4, 5]]
    assert packed["attention_mask"].tolist() == [[1, 1, 1], [0, 1, 1]]


def test_strict_target_blind_parity_compares_loss_raw_views_and_dedup_pool() -> None:
    first = _record(task_id="a", loss=[1.0, 0.9], raw=[{"augmentation_index": 0, "grid": [[1]]}], candidates=[{"prediction": [[[1]]]}])
    same = _record(task_id="a", loss=[1.0, 0.9], raw=[{"augmentation_index": 0, "grid": [[1]]}], candidates=[{"prediction": [[[1]]]}])
    different = _record(task_id="a", loss=[1.0, 0.91], raw=[{"augmentation_index": 0, "grid": [[2]]}], candidates=[{"prediction": [[[2]]]}])
    serial = {"identity": "serial", "task_ids": ["a"], "records": {"a": first}}
    assert compare_target_blind(serial, {"identity": "same", "task_ids": ["a"], "records": {"a": same}})["PARITY_SAFE"]
    evidence = compare_target_blind(serial, {"identity": "different", "task_ids": ["a"], "records": {"a": different}})
    assert not evidence["PARITY_SAFE"]
    assert not evidence["tasks"]["a"]["ttt_loss_agreement"]
    assert not evidence["tasks"]["a"]["raw_view_candidate_agreement"]
    assert not evidence["tasks"]["a"]["candidate_pool_agreement"]


def test_fingerprint_exposes_only_target_blind_generation_evidence() -> None:
    record = _record(task_id="x", loss=[0.5], raw=[], candidates=[], invalid=8)
    evidence = _record_fingerprint(record)
    assert evidence["invalid_candidate_count"] == 8
    assert set(evidence) == {"raw_views_sha256", "candidate_pool_sha256", "invalid_candidate_count", "unique_candidate_count", "loss_curve_sha256"}
