"""Frozen root-length memory profiles for target-blind AUG16 DFS execution.

The table governs resident logical state and physical batch ceiling only.  It
never reads CUDA free memory, decoder outcomes, or any Gold-derived label.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


KV_BLOCK_TOKENS = 256
MAX_NEW_TOKENS = 931


@dataclass(frozen=True)
class MemoryProfile:
    name: str
    lower_root_length: int
    upper_root_length: int | None
    resident_width: int
    physical_batch_ceiling: int
    b16_to_b8_fallback: bool


PROFILE_S = MemoryProfile("PROFILE_S", 0, 2048, 16, 16, True)
PROFILE_M = MemoryProfile("PROFILE_M", 2049, 2653, 16, 8, False)
PROFILE_L = MemoryProfile("PROFILE_L", 2654, 6493, 8, 8, False)
PROFILE_XL = MemoryProfile("PROFILE_XL", 6494, 17245, 4, 4, False)
PROFILE_XXL = MemoryProfile("PROFILE_XXL", 17246, 36701, 2, 2, False)
PROFILE_OVERSIZE = MemoryProfile("PROFILE_OVERSIZE", 36702, None, 0, 0, False)
FROZEN_PROFILES = (PROFILE_S, PROFILE_M, PROFILE_L, PROFILE_XL, PROFILE_XXL, PROFILE_OVERSIZE)


def select_memory_profile(root_max: int) -> MemoryProfile:
    """Select the single pre-registered profile for an AUG16 root maximum."""
    if int(root_max) < 0:
        raise ValueError("root length cannot be negative")
    root_max = int(root_max)
    for profile in FROZEN_PROFILES:
        if root_max >= profile.lower_root_length and (profile.upper_root_length is None or root_max <= profile.upper_root_length):
            return profile
    raise AssertionError("frozen root-length profile table is incomplete")


def aligned_capacity(required_length: int, *, block_tokens: int = KV_BLOCK_TOKENS) -> int:
    if int(required_length) <= 0:
        return 0
    if int(block_tokens) != KV_BLOCK_TOKENS:
        raise ValueError("this engineering path is frozen to 256-token KV blocks")
    return ((int(required_length) + block_tokens - 1) // block_tokens) * block_tokens


def required_capacity_for_root(root_max: int, *, max_new_tokens: int = MAX_NEW_TOKENS) -> int:
    """Return capacity for KV storage only; it never changes profile choice."""
    if int(max_new_tokens) != MAX_NEW_TOKENS:
        raise ValueError("this engineering path is frozen to max_new_tokens=931")
    return aligned_capacity(int(root_max) + int(max_new_tokens))


def deterministic_resident_groups(candidate_ids: Iterable[str], profile: MemoryProfile) -> tuple[tuple[str, ...], ...]:
    """Partition frozen AUG16 order without reordering or dropping any view."""
    ids = tuple(str(item) for item in candidate_ids)
    if len(ids) != 16 or len(set(ids)) != 16:
        raise ValueError("resident grouping requires the exact sixteen distinct frozen AUG16 IDs")
    if profile is PROFILE_OVERSIZE:
        return ()
    width = profile.resident_width
    if len(ids) % width:
        raise AssertionError("frozen profile width must partition AUG16 exactly")
    return tuple(ids[index:index + width] for index in range(0, len(ids), width))
