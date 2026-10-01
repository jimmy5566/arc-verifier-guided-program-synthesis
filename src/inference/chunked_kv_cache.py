"""Allocator-friendly Transformers cache with distinct logical and allocated length.

Each layer owns fixed-capacity K/V buffers.  ``keys`` and ``values`` are only
valid-length views, so the model never attends to spare capacity.  ``crop``
changes views only; it never shrinks or reallocates storage.
"""
from __future__ import annotations

from typing import Any, Callable

try:  # The runner uses Transformers; retaining a fallback keeps CPU-only imports light.
    from transformers.cache_utils import Cache as _TransformersCache
except ImportError:  # pragma: no cover - exercised only outside the project runtime.
    _TransformersCache = object


GrowthObserver = Callable[[dict[str, Any]], None]


def _capacity_for(required: int, block_tokens: int) -> int:
    if required <= 0:
        return 0
    return ((int(required) + block_tokens - 1) // block_tokens) * block_tokens


class ChunkedDynamicLayer:
    is_sliding = False

    def __init__(self, *, block_tokens: int, owner_id: str, layer_index: int,
                 growth_observer: GrowthObserver | None = None) -> None:
        self.block_tokens = int(block_tokens)
        self.owner_id = owner_id
        self.layer_index = int(layer_index)
        self.growth_observer = growth_observer
        self.keys = None
        self.values = None
        self._key_buffer = None
        self._value_buffer = None
        self.valid_length = 0
        self.capacity_length = 0

    def _grow(self, required: int, exemplar: Any) -> None:
        import torch

        new_capacity = _capacity_for(required, self.block_tokens)
        if new_capacity <= self.capacity_length:
            return
        old_capacity = self.capacity_length
        old_key, old_value = self._key_buffer, self._value_buffer
        shape = list(exemplar.shape); shape[-2] = new_capacity
        new_key = torch.empty(tuple(shape), device=exemplar.device, dtype=exemplar.dtype)
        new_value = torch.empty(tuple(shape), device=exemplar.device, dtype=exemplar.dtype)
        if self.valid_length:
            new_key[..., :self.valid_length, :].copy_(old_key[..., :self.valid_length, :])
            new_value[..., :self.valid_length, :].copy_(old_value[..., :self.valid_length, :])
        self._key_buffer, self._value_buffer = new_key, new_value
        self.capacity_length = new_capacity
        if self.growth_observer is not None:
            self.growth_observer({
                "owner_id": self.owner_id, "layer_index": self.layer_index,
                "old_capacity": old_capacity, "new_capacity": new_capacity,
                "valid_length_before": self.valid_length,
                "requested_bytes": int(new_key.numel() * new_key.element_size() + new_value.numel() * new_value.element_size()),
                "copied_existing_bytes": int(
                    0 if old_key is None else old_key[..., :self.valid_length, :].numel() * old_key.element_size()
                ) + int(
                    0 if old_value is None else old_value[..., :self.valid_length, :].numel() * old_value.element_size()
                ),
            })

    def _set_views(self) -> None:
        self.keys = self._key_buffer[..., :self.valid_length, :] if self._key_buffer is not None else None
        self.values = self._value_buffer[..., :self.valid_length, :] if self._value_buffer is not None else None

    def update(self, key_states: Any, value_states: Any, cache_kwargs: dict[str, Any] | None = None) -> tuple[Any, Any]:
        del cache_kwargs
        append = int(key_states.shape[-2])
        old = self.valid_length
        self._grow(old + append, key_states)
        self._key_buffer[..., old:old + append, :].copy_(key_states)
        self._value_buffer[..., old:old + append, :].copy_(value_states)
        self.valid_length = old + append
        self._set_views()
        return self.keys, self.values

    def adopt_valid(self, key_states: Any, value_states: Any) -> None:
        """Adopt only the changed suffix from a temporary packed-cache lane."""
        new_length = int(key_states.shape[-2])
        old = self.valid_length
        if new_length < old:
            self.crop(new_length)
            return
        self._grow(new_length, key_states)
        if new_length > old:
            self._key_buffer[..., old:new_length, :].copy_(key_states[..., old:new_length, :])
            self._value_buffer[..., old:new_length, :].copy_(value_states[..., old:new_length, :])
        self.valid_length = new_length
        self._set_views()

    def get_seq_length(self, cache_position=None) -> int:
        del cache_position
        return int(self.valid_length)

    def get_max_cache_shape(self) -> int:
        return int(self.capacity_length)

    def get_mask_sizes(self, cache_position: Any) -> tuple[int, int]:
        return int(self.valid_length + cache_position.shape[0]), 0

    def crop(self, max_length: int) -> None:
        if max_length < 0:
            max_length = self.valid_length - abs(max_length)
        self.valid_length = min(self.valid_length, max(0, int(max_length)))
        self._set_views()

    def reset(self) -> None:
        self.keys = self.values = self._key_buffer = self._value_buffer = None
        self.valid_length = self.capacity_length = 0

    def reorder_cache(self, beam_idx: Any) -> None:
        raise RuntimeError("chunked cache beam reordering is outside this fixed single-lane DFS contract")

    def batch_repeat_interleave(self, repeats: int) -> None:
        raise RuntimeError("chunked cache batch expansion is outside this fixed single-lane DFS contract")

    def batch_select_indices(self, indices: Any) -> None:
        raise RuntimeError("chunked cache batch selection is outside this fixed single-lane DFS contract")

    @property
    def max_batch_size(self) -> int:
        return int(self._key_buffer.shape[0]) if self._key_buffer is not None else 0

    @property
    def max_cache_len(self) -> int:
        return int(self.capacity_length)

    @property
    def is_compileable(self) -> bool:
        return False


class ChunkedDynamicCache(_TransformersCache):
    """Public Cache-compatible owner whose layer tensors expose valid views only."""

    def __init__(self, *, block_tokens: int = 256, owner_id: str = "unknown",
                 growth_observer: GrowthObserver | None = None) -> None:
        if int(block_tokens) != 256:
            raise ValueError("this experiment is frozen to KV_BLOCK_TOKENS=256")
        self.block_tokens = int(block_tokens)
        self.owner_id = owner_id
        self.growth_observer = growth_observer
        self.layers: list[ChunkedDynamicLayer] = []
        # ``Cache.__init__`` eagerly appends model-configured layers.  This
        # cache instead receives every real Qwen layer from a frozen root
        # prefill, so initialise the public Cache fields deliberately without
        # allocating any synthetic layer or capacity tail.
        self.layer_classes = ChunkedDynamicLayer
        self.cache_processor = None
        self.layer_init_kwargs: dict[str, Any] = {}
        self.num_hidden_layers = 0

    @classmethod
    def from_legacy_cache(cls, legacy: tuple[tuple[Any, Any], ...], *, block_tokens: int = 256,
                          owner_id: str, growth_observer: GrowthObserver | None = None) -> "ChunkedDynamicCache":
        cache = cls(block_tokens=block_tokens, owner_id=owner_id, growth_observer=growth_observer)
        for layer_index, (key, value) in enumerate(legacy):
            layer = ChunkedDynamicLayer(block_tokens=block_tokens, owner_id=owner_id, layer_index=layer_index,
                                        growth_observer=growth_observer)
            layer.update(key, value)
            cache.layers.append(layer)
        return cache

    def append_new_layers(self, layer_idx: int) -> None:
        raise RuntimeError("chunked cache must be initialized from the full root legacy cache")

    def __len__(self) -> int:
        return len(self.layers)

    def __iter__(self):
        for layer in self.layers:
            yield layer.keys, layer.values

    def __getitem__(self, layer_idx: int) -> tuple[Any, Any]:
        if not 0 <= int(layer_idx) < len(self.layers):
            raise KeyError(f"chunked cache has {len(self.layers)} layers, requested {layer_idx}")
        layer = self.layers[int(layer_idx)]
        return layer.keys, layer.values

    def update(self, key_states: Any, value_states: Any, layer_idx: int,
               cache_kwargs: dict[str, Any] | None = None) -> tuple[Any, Any]:
        return self.layers[layer_idx].update(key_states, value_states, cache_kwargs)

    def get_seq_length(self, layer_idx: int = 0, cache_position=None) -> int:
        return self.layers[layer_idx].get_seq_length(cache_position)

    def get_mask_sizes(self, cache_position: Any, layer_idx: int) -> tuple[int, int]:
        return self.layers[layer_idx].get_mask_sizes(cache_position)

    def get_max_cache_shape(self, layer_idx: int = 0) -> int:
        return self.layers[layer_idx].get_max_cache_shape()

    def crop(self, max_length: int) -> None:
        for layer in self.layers:
            layer.crop(max_length)

    def to_legacy_cache(self) -> tuple[tuple[Any, Any], ...]:
        return tuple((layer.keys, layer.values) for layer in self.layers)

    def adopt_legacy_suffix(self, legacy: tuple[tuple[Any, Any], ...]) -> None:
        if len(legacy) != len(self.layers):
            raise RuntimeError("chunked cache layer count mismatch during adoption")
        for layer, (key, value) in zip(self.layers, legacy, strict=True):
            layer.adopt_valid(key, value)

    def reset(self) -> None:
        for layer in self.layers:
            layer.reset()

    def reorder_cache(self, beam_idx: Any) -> None:
        raise RuntimeError("chunked cache beam reordering is outside this fixed single-lane DFS contract")

    def batch_repeat_interleave(self, repeats: int) -> None:
        raise RuntimeError("chunked cache batch expansion is outside this fixed single-lane DFS contract")

    def batch_select_indices(self, indices: Any) -> None:
        raise RuntimeError("chunked cache batch selection is outside this fixed single-lane DFS contract")

    def capacity_lengths(self) -> list[int]:
        return [int(layer.capacity_length) for layer in self.layers]

    def valid_lengths(self) -> list[int]:
        return [int(layer.valid_length) for layer in self.layers]

    @property
    def max_batch_size(self) -> int:
        return self.layers[0].max_batch_size if self.layers else 0

    @property
    def max_cache_len(self) -> int:
        return max(self.capacity_lengths(), default=0)

    @property
    def is_compileable(self) -> bool:
        return False

    @property
    def is_sliding(self) -> list[bool]:
        return [False] * len(self.layers)
