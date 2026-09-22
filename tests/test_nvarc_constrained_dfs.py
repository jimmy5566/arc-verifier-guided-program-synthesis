from __future__ import annotations

from inference.nvarc_constrained_dfs import ConstrainedDFSConfig, GridPrefix


def test_grid_prefix_accepts_rectangular_grid_with_or_without_final_newline() -> None:
    state = GridPrefix()
    for token in (1, 2, 10, 3, 4):
        state = state.consume(token)
        assert state is not None
    assert state.terminal
    assert state.consume(15) == state

    state = GridPrefix()
    for token in (1, 2, 10, 3, 4, 10):
        state = state.consume(token)
        assert state is not None
    assert state.terminal
    assert state.consume(15) == state


def test_grid_prefix_rejects_nonrectangular_or_empty_termination() -> None:
    assert GridPrefix().consume(15) is None
    state = GridPrefix()
    for token in (1, 2, 10, 3):
        state = state.consume(token)
        assert state is not None
    assert state.consume(10) is None
    assert state.consume(15) is None


def test_grid_prefix_bounded_dimensions() -> None:
    state = GridPrefix()
    for _ in range(30):
        state = state.consume(1)
        assert state is not None
    assert 1 not in state.allowed_tokens()
    assert state.consume(10) is not None


def test_dfs_config_is_conservative_and_bounded() -> None:
    config = ConstrainedDFSConfig(max_candidates=4, max_new_tokens=1024, max_branch_tokens=4)
    assert config.max_candidates == 4
    assert config.max_new_tokens == 1024
