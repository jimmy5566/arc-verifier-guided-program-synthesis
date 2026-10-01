from inference.root_length_memory_profile import (
    PROFILE_L,
    PROFILE_M,
    PROFILE_OVERSIZE,
    PROFILE_S,
    PROFILE_XL,
    PROFILE_XXL,
    deterministic_resident_groups,
    make_root_profile_execution_plan,
    required_capacity_for_root,
    select_memory_profile,
)


def test_frozen_profile_boundaries() -> None:
    cases = {
        2048: PROFILE_S, 2049: PROFILE_M, 2653: PROFILE_M, 2654: PROFILE_L,
        6493: PROFILE_L, 6494: PROFILE_XL, 17245: PROFILE_XL, 17246: PROFILE_XXL,
        36701: PROFILE_XXL, 36702: PROFILE_OVERSIZE,
    }
    assert {root: select_memory_profile(root) for root in cases} == cases


def test_capacity_is_root_adaptive_not_a_global_3072_cap() -> None:
    assert required_capacity_for_root(1928) == 3072
    assert required_capacity_for_root(3000) == 4096
    assert required_capacity_for_root(4000) == 5120


def test_resident_groups_preserve_frozen_order_and_full_aug16_coverage() -> None:
    ids = tuple(f"aug{index:02d}" for index in range(16))
    assert deterministic_resident_groups(ids, PROFILE_S) == (ids,)
    assert deterministic_resident_groups(ids, PROFILE_M) == (ids[:8], ids[8:])
    assert deterministic_resident_groups(ids, PROFILE_L) == (ids[:8], ids[8:])
    assert deterministic_resident_groups(ids, PROFILE_XL) == (ids[:4], ids[4:8], ids[8:12], ids[12:])
    assert deterministic_resident_groups(ids, PROFILE_XXL) == tuple(ids[index:index + 2] for index in range(0, 16, 2))
    assert deterministic_resident_groups(ids, PROFILE_OVERSIZE) == ()


def test_execution_plan_uses_only_root_length_and_frozen_order() -> None:
    ids = tuple(f"aug{index:02d}" for index in range(16))
    plan = make_root_profile_execution_plan(3000, ids)
    assert plan.profile is PROFILE_L
    assert plan.initial_kv_capacity == 3072
    assert plan.final_required_kv_capacity == 4096
    assert plan.resident_groups == (ids[:8], ids[8:])
