from scripts.run_serial_batch_consistency_probe import parse_candidates, strict_prediction_difference


def _row(**changes):
    row = {
        "episode_id": "e1", "generated_token_ids": [1, 2, 15], "generated_length": 3,
        "termination_status": "TERMINAL_EOS", "eos_observed": True, "trailing_pad_count": 0,
        "content_token_ids": [1, 2], "parse_reason": "VALID", "canonical_prediction_sha256": "x",
        "parse_valid": True, "exact_grid_match": False,
    }
    row.update(changes)
    return row


def test_strict_difference_requires_raw_token_identity():
    assert strict_prediction_difference(_row(), _row()) == []
    assert strict_prediction_difference(_row(), _row(generated_token_ids=[1, 2, 15, 13])) == ["generated_token_ids"]
    assert strict_prediction_difference(_row(), _row(canonical_prediction_sha256="y")) == ["canonical_prediction_sha256"]


def test_candidates_are_doubling_and_bounded():
    assert parse_candidates("2,4,8,16,32") == (2, 4, 8, 16, 32)
    for invalid in ("", "4,2", "2,2", "64"):
        try:
            parse_candidates(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(invalid)
