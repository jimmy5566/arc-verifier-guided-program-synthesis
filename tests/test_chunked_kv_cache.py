"""CPU-only contract tests for the allocator-friendly DFS cache owner."""
from __future__ import annotations

import torch

from inference.chunked_kv_cache import ChunkedDynamicCache


def _legacy(length: int = 1928, layers: int = 2):
    return tuple(
        (
            torch.full((1, 2, length, 4), float(index + 1)),
            torch.full((1, 2, length, 4), float(-(index + 1))),
        )
        for index in range(layers)
    )


def _append(cache: ChunkedDynamicCache, count: int, value: float) -> None:
    for layer_index in range(len(cache)):
        key = torch.full((1, 2, count, 4), value + layer_index)
        val = torch.full((1, 2, count, 4), -value - layer_index)
        returned_key, returned_value = cache.update(key, val, layer_index)
        assert int(returned_key.shape[-2]) == cache.get_seq_length(layer_index)
        assert int(returned_value.shape[-2]) == cache.get_seq_length(layer_index)


def test_chunked_cache_grows_only_at_frozen_256_token_boundaries() -> None:
    events: list[dict] = []
    cache = ChunkedDynamicCache.from_legacy_cache(
        _legacy(), owner_id="owner-a", growth_observer=events.append,
    )
    owner_identity = id(cache)
    assert isinstance(cache.to_legacy_cache(), tuple)
    assert cache.get_seq_length() == 1928
    assert cache.capacity_lengths() == [2048, 2048]
    assert [int(item.shape[-2]) for item in cache[0]] == [1928, 1928]

    # Exact transitions: 2048 -> 2304 -> 2560 -> 2816 -> 3072 capacity.
    for count, expected_length, expected_capacity in (
        (120, 2048, 2048), (1, 2049, 2304), (255, 2304, 2304),
        (1, 2305, 2560), (255, 2560, 2560), (1, 2561, 2816),
        (255, 2816, 2816), (1, 2817, 3072),
    ):
        _append(cache, count, float(expected_length))
        assert cache.get_seq_length() == expected_length
        assert cache.capacity_lengths() == [expected_capacity, expected_capacity]
        assert [int(item.shape[-2]) for item in cache[0]] == [expected_length, expected_length]
        assert id(cache) == owner_identity

    layer_zero_capacities = [event["new_capacity"] for event in events if event["layer_index"] == 0]
    assert layer_zero_capacities == [2048, 2304, 2560, 2816, 3072]


def test_chunked_cache_rollback_preserves_capacity_and_sibling_isolation() -> None:
    left = ChunkedDynamicCache.from_legacy_cache(_legacy(), owner_id="left")
    right = ChunkedDynamicCache.from_legacy_cache(_legacy(), owner_id="right")
    left_owner, right_owner = id(left), id(right)
    left_key_buffer_ptr = left.layers[0]._key_buffer.data_ptr()
    right_key_buffer_ptr = right.layers[0]._key_buffer.data_ptr()
    assert left_key_buffer_ptr != right_key_buffer_ptr

    parent = left.get_seq_length()
    _append(left, 1, 11.0)
    assert left.get_seq_length() == parent + 1
    assert float(left.layers[0].keys[0, 0, parent, 0]) == 11.0
    left.crop(parent)
    assert left.get_seq_length() == parent
    assert left.capacity_lengths() == [2048, 2048]
    assert int(left.layers[0].keys.shape[-2]) == parent

    # A later sibling writes the same physical suffix only after rollback;
    # its value cannot leak into another owner or into parent-visible history.
    _append(left, 1, 29.0)
    assert float(left.layers[0].keys[0, 0, parent, 0]) == 29.0
    assert float(right.layers[0].keys[0, 0, parent - 1, 0]) == 1.0
    left.crop(parent)
    assert torch.all(left.layers[0].keys[..., :parent, :] == 1.0)
    assert id(left) == left_owner and id(right) == right_owner
    assert left.layers[0]._key_buffer.data_ptr() == left_key_buffer_ptr
    assert right.layers[0]._key_buffer.data_ptr() == right_key_buffer_ptr


def test_chunked_cache_adopts_only_suffix_and_never_exposes_capacity_tail() -> None:
    cache = ChunkedDynamicCache.from_legacy_cache(_legacy(), owner_id="owner")
    incoming = _legacy(1929)
    # Make the single appended position distinct from root contents.
    incoming = tuple(
        (
            torch.cat((key[..., :1928, :], torch.full_like(key[..., 1928:, :], 41.0)), dim=-2),
            torch.cat((value[..., :1928, :], torch.full_like(value[..., 1928:, :], -41.0)), dim=-2),
        )
        for key, value in incoming
    )
    ptr = cache.layers[0]._key_buffer.data_ptr()
    cache.adopt_legacy_suffix(incoming)
    assert cache.get_seq_length() == 1929
    assert cache.layers[0]._key_buffer.data_ptr() == ptr
    assert int(cache.layers[0].keys.shape[-2]) == 1929
    assert float(cache.layers[0].keys[0, 0, 1928, 0]) == 41.0
    assert int(cache.layers[0]._key_buffer.shape[-2]) == 2048
