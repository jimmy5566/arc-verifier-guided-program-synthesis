from types import SimpleNamespace

from inference.root_adaptive_batch_policy import (
    FairCompatibilitySelector,
    RootAwareAdmissionQueue,
    RootBatchPolicy,
    RootBatchPolicyEntry,
)


def test_policy_lookup_and_monotone_boundary() -> None:
    policy = RootBatchPolicy((
        RootBatchPolicyEntry(2048, 16, 16),
        RootBatchPolicyEntry(3072, 8, 8),
        RootBatchPolicyEntry(5120, 6, 4),
    ))
    assert policy.lookup(2048).physical_batch_ceiling == 16
    assert policy.lookup(2049).resident_capacity == 8
    assert policy.lookup(5120).physical_batch_ceiling == 4


def test_policy_rejects_non_monotone_batch_ceiling() -> None:
    try:
        RootBatchPolicy((RootBatchPolicyEntry(1024, 8, 4), RootBatchPolicyEntry(2048, 8, 5)))
    except ValueError as error:
        assert "may not increase" in str(error)
    else:  # pragma: no cover - explicit fail path
        raise AssertionError("non-monotone policy unexpectedly accepted")


def test_root_aware_admission_uses_9_then_7_without_mixing_initial_eight() -> None:
    ids = [f"a{index}" for index in range(16)]
    roots = {key: 2078 if index < 9 else 2072 for index, key in enumerate(ids)}
    queue = RootAwareAdmissionQueue(ids, root_lengths=roots, resident_capacity=8)
    assert queue.admit_available() == ids[:8]
    queue.terminate("a0")
    assert queue.admit_available() == ["a8"]
    queue.terminate("a1")
    assert queue.admit_available() == ["a9"]


def test_root_aware_tie_uses_frozen_order() -> None:
    ids = ["a", "b", "c", "d"]
    roots = {"a": 10, "b": 20, "c": 10, "d": 20}
    queue = RootAwareAdmissionQueue(ids, root_lengths=roots, resident_capacity=2)
    assert queue.admit_available() == ["a", "c"]


def test_fair_selector_prefers_largest_then_serves_small_class_before_starvation() -> None:
    def cell(name: str, cache: str, position: int) -> SimpleNamespace:
        return SimpleNamespace(cell_key=name, request=SimpleNamespace(cache_key=(cache,), position=position))

    large = [cell(f"l{index}", "large", 10) for index in range(9)]
    small = [cell(f"s{index}", "small", 10) for index in range(7)]
    selector = FairCompatibilitySelector(max_wait=2)
    first = selector.select([*large, *small], lambda _position: 16)
    second = selector.select([*large, *small], lambda _position: 16)
    third = selector.select([*large, *small], lambda _position: 16)
    assert first is not None and first.selected_width == 9
    assert second is not None and second.selected_width == 9
    assert third is not None and third.selected_width == 7
    assert third.class_wait_before == 2


def test_selector_respects_arbitrary_safe_ceiling_and_exact_position() -> None:
    cells = [SimpleNamespace(cell_key=f"x{index}", request=SimpleNamespace(cache_key=("same",), position=42)) for index in range(7)]
    selected = FairCompatibilitySelector().select(cells, lambda position: 5 if position == 42 else 1)
    assert selected is not None
    assert selected.selected_width == 5
    assert selected.safe_ceiling == 5
