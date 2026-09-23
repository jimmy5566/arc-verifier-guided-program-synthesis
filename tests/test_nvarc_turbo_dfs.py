from inference.nvarc_turbo_dfs import TurboDFSConfig, allowed_arc_tokens


def test_turbo_dfs_config_requires_finite_positive_bounds() -> None:
    config = TurboDFSConfig(901, 1.6094379124341003, 45.0, 64, 4)
    assert config.max_complete_candidates == 4
    try:
        TurboDFSConfig(1, 1.0, 1.0, 1, 1)
    except ValueError as error:
        assert "max_new_tokens" in str(error)
    else:
        raise AssertionError("invalid max_new_tokens accepted")


def test_public_arc_token_set_is_digits_newline_and_native_eos() -> None:
    assert allowed_arc_tokens(eos_token_id=15) == tuple(range(11)) + (15,)
