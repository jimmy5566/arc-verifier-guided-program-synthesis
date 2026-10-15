from __future__ import annotations

import pytest

from scripts.run_eval60_fixed4_d1_worker_v1 import fixed_four_groups


def test_fixed_four_groups_accepts_historical_four_plus_four_shape() -> None:
    identifiers = [f"v{index}" for index in range(8)]
    roots = {identifier: 100 if index < 4 else 120 for index, identifier in enumerate(identifiers)}

    assert fixed_four_groups(identifiers, roots) == [identifiers[:4], identifiers[4:]]


def test_fixed_four_groups_accepts_eight_equal_roots_as_two_lanesets() -> None:
    identifiers = [f"v{index}" for index in range(8)]

    assert fixed_four_groups(identifiers, {identifier: 100 for identifier in identifiers}) == [
        identifiers[:4], identifiers[4:]
    ]


def test_fixed_four_groups_rejects_incompatible_root_mixing() -> None:
    identifiers = [f"v{index}" for index in range(8)]
    roots = {identifier: 100 if index < 3 else 120 for index, identifier in enumerate(identifiers)}

    with pytest.raises(RuntimeError, match="would mix incompatible roots"):
        fixed_four_groups(identifiers, roots)
