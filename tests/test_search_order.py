from inference.search_order import CumulativeNLLBestFirst, FairDFS, LDSUnitDiscrepancy, RegretBandFairDFS, RetainedWorkItem, make_search_order_policy


def item(
    identifier: int,
    root: int,
    regret: float,
    *,
    nll: float | None = None,
    rank: int = 1,
    discrepancy: int = 0,
) -> RetainedWorkItem:
    return RetainedWorkItem(
        identifier,
        root,
        None,
        float(identifier) if nll is None else nll,
        regret,
        identifier,
        local_token_rank=rank,
        discrepancy_count=discrepancy,
    )


def take(policy, count: int) -> list[int]:
    values = []
    for _ in range(count):
        current = policy.pop()
        assert current is not None
        values.append(current.work_item_id)
        policy.note_expansion(current)
    return values


def test_current_dfs_is_depth_first_with_low_nll_siblings_first() -> None:
    policy = make_search_order_policy("CURRENT_DFS")
    policy.push_successors((item(1, 1, 0.1, nll=0.1), item(2, 2, 0.1, nll=0.2)))
    first = policy.pop(); assert first is not None and first.work_item_id == 1
    policy.note_expansion(first)
    # A descendant is later pushed and must run before the old root sibling.
    policy.push_successors((item(3, 1, 0.2, nll=0.01),))
    assert take(policy, 2) == [3, 2]


def test_fair_q64_yields_to_a_retained_sibling() -> None:
    policy = FairDFS(quantum=2)
    policy.push_successors(tuple(item(identifier, 10, 0.1) for identifier in range(1, 4)))
    policy.push_successors(tuple(item(identifier, 20, 0.1) for identifier in range(4, 7)))
    assert take(policy, 6) == [1, 2, 4, 5, 3, 6]


def test_fair_policy_changes_only_order_not_retained_membership() -> None:
    values = [item(1, 10, 0.1), item(2, 10, 1.1), item(3, 20, 2.1), item(4, 20, 3.1)]
    policy = FairDFS(quantum=1)
    for value in values:
        policy.push(value)
    observed = take(policy, len(values))
    assert set(observed) == {value.work_item_id for value in values}
    assert len(observed) == len(set(observed))


def test_regret_band_cycle_is_exact_when_all_bands_have_work() -> None:
    policy = RegretBandFairDFS(quantum=64)
    next_id = 1
    for _ in range(8):
        for band, regret in enumerate((0.1, 1.1, 2.1, 3.1)):
            policy.push(item(next_id, next_id, regret)); next_id += 1
    selected_bands = []
    for _ in range(8):
        current = policy.pop(); assert current is not None
        selected_bands.append(current.regret_band)
        policy.note_expansion(current)
    assert selected_bands == [0, 0, 1, 0, 2, 1, 0, 3]


def test_regret_band_empty_band_skips_without_losing_work() -> None:
    policy = RegretBandFairDFS(quantum=1)
    policy.push(item(1, 1, 0.1))
    policy.push(item(2, 2, 3.1))
    assert take(policy, 2) == [1, 2]
    assert policy.pop() is None


def test_regret_boundary_four_is_retained_band_three() -> None:
    assert item(1, 1, 0.0).regret_band == 0
    assert item(2, 1, 1.0).regret_band == 1
    assert item(3, 1, 2.0).regret_band == 2
    assert item(4, 1, 3.0).regret_band == 3
    assert item(5, 1, 4.0).regret_band == 3


def test_lds_always_exhausts_lower_discrepancy_first() -> None:
    policy = LDSUnitDiscrepancy()
    policy.push(item(1, 1, 0.1, nll=0.3, discrepancy=1, rank=2))
    policy.push(item(2, 2, 0.1, nll=0.2, discrepancy=0, rank=1))
    policy.push(item(3, 3, 0.1, nll=0.1, discrepancy=2, rank=2))
    assert take(policy, 3) == [2, 1, 3]


def test_lds_is_depth_first_within_one_discrepancy_layer() -> None:
    policy = LDSUnitDiscrepancy()
    policy.push_successors((
        item(1, 1, 0.1, nll=0.1, discrepancy=0),
        item(2, 2, 0.1, nll=0.2, discrepancy=0),
    ))
    first = policy.pop()
    assert first is not None and first.work_item_id == 1
    policy.note_expansion(first)
    policy.push_successors((item(3, 1, 0.2, nll=0.5, discrepancy=0),))
    assert take(policy, 2) == [3, 2]


def test_lds_keeps_every_retained_item_exactly_once() -> None:
    values = (
        item(1, 10, 0.1, discrepancy=0),
        item(2, 10, 0.2, discrepancy=1, rank=2),
        item(3, 20, 0.3, discrepancy=1, rank=3),
        item(4, 20, 0.4, discrepancy=2, rank=2),
    )
    policy = LDSUnitDiscrepancy()
    policy.push_successors(values)
    observed = take(policy, len(values))
    assert set(observed) == {value.work_item_id for value in values}
    assert len(observed) == len(set(observed))
    assert policy.pop() is None


def test_lds_factory_uses_frozen_policy_name() -> None:
    policy = make_search_order_policy("LDS_UNIT_DISCREPANCY_V1")
    assert isinstance(policy, LDSUnitDiscrepancy)


def test_nll_best_first_is_global_not_depth_first() -> None:
    policy = CumulativeNLLBestFirst()
    policy.push_successors((
        item(1, 10, 0.1, nll=0.20),
        item(2, 20, 0.1, nll=0.40),
    ))
    first = policy.pop()
    assert first is not None and first.work_item_id == 1
    policy.note_expansion(first)
    # A descendant with worse cumulative NLL must wait behind an older,
    # unrelated frontier item with lower cumulative NLL.
    policy.push_successors((item(3, 10, 0.2, nll=0.50),))
    assert take(policy, 2) == [2, 3]


def test_nll_best_first_uses_frozen_deterministic_ties() -> None:
    policy = CumulativeNLLBestFirst()
    policy.push(item(3, 3, 0.1, nll=0.5))
    policy.push(item(1, 1, 0.1, nll=0.5))
    policy.push(item(2, 2, 0.1, nll=0.5))
    assert take(policy, 3) == [1, 2, 3]


def test_nll_best_first_keeps_every_retained_item_exactly_once() -> None:
    values = (
        item(1, 10, 0.1, nll=0.8),
        item(2, 10, 1.1, nll=0.2),
        item(3, 20, 2.1, nll=0.6),
        item(4, 20, 3.1, nll=0.4),
    )
    policy = CumulativeNLLBestFirst()
    policy.push_successors(values)
    observed = take(policy, len(values))
    assert observed == [2, 4, 3, 1]
    assert len(observed) == len(set(observed))
    assert policy.pop() is None


def test_nll_best_first_factory_uses_frozen_policy_name() -> None:
    policy = make_search_order_policy("CUMULATIVE_NLL_BEST_FIRST_V1")
    assert isinstance(policy, CumulativeNLLBestFirst)
