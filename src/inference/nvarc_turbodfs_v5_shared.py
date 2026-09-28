"""Shared-model execution adapter for the frozen V5 TurboDFS decoder.

The V5 decoder already supports a batch of independent lanes and keeps KV
cache rows batched by lane.  This module makes that fact explicit at the
execution boundary: a *single* model instance may execute one or two logical
cells, provided that both cells were adapted with the identical task/depth
adapter.  It deliberately delegates all search decisions to
``inference_frontier_floor_turbo_dfs``; it is not a second decoder.

The wrapper is intentionally small.  Its main purpose is to make model,
adapter, context and cache ownership auditable before the scheduler uses the
shared route on a live run.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from inference.nvarc_turbodfs_v5 import FrontierFloorTurboDFSConfig, FrontierFloorTurboDFSResult, inference_frontier_floor_turbo_dfs


@dataclass(frozen=True)
class SearchContext:
    """Immutable identity of one logical V5 cell.

    ``input_ids`` is intentionally owned by the context rather than the
    executor.  That makes an accidental cross-context prompt/cache route fail
    at the boundary before a model call is made.
    """

    context_id: str
    cell_key: str
    adapter_identity: str
    input_ids: Any


@dataclass(frozen=True)
class SharedExecution:
    """A decoder result plus the immutable row-to-context routing map."""

    contexts: tuple[SearchContext, ...]
    result: FrontierFloorTurboDFSResult

    def lane_for_context(self, context_id: str) -> int:
        for lane, context in enumerate(self.contexts):
            if context.context_id == context_id:
                return lane
        raise KeyError(context_id)

    def context_nodes(self, context_id: str) -> tuple[dict[str, Any], ...]:
        lane = self.lane_for_context(context_id)
        return tuple({**row, "search_context_id": context_id, "cache_slot": lane} for row in self.result.nodes if int(row.get("lane", -1)) == lane)

    def context_probabilities(self, context_id: str) -> tuple[dict[str, Any], ...]:
        lane = self.lane_for_context(context_id)
        return tuple({**row, "search_context_id": context_id, "cache_slot": lane} for row in self.result.branch_probabilities if int(row.get("lane", -1)) == lane)


class SharedForwardExecutor:
    """One-model V5 executor for up to two same-adapter logical cells.

    V5's recursive lane implementation supplies the per-lane KV isolation.
    This class checks the invariants that must hold before calling it and
    records the exact context order used as the cache-slot order.
    """

    def __init__(self, model: Any, config: FrontierFloorTurboDFSConfig) -> None:
        self.model = model
        self.config = config

    @staticmethod
    def _validate(contexts: tuple[SearchContext, ...]) -> None:
        if not 1 <= len(contexts) <= 2:
            raise ValueError("shared V5 executor accepts exactly one or two contexts")
        if len({item.context_id for item in contexts}) != len(contexts):
            raise ValueError("logical contexts must have distinct context IDs")
        if len({item.cell_key for item in contexts}) != len(contexts):
            raise ValueError("logical contexts must have distinct cell keys")
        if len({item.adapter_identity for item in contexts}) != 1:
            raise ValueError("different adapter states cannot share a V5 model forward")
        widths: set[int] = set()
        for item in contexts:
            shape = getattr(item.input_ids, "shape", None)
            if shape is None or len(shape) != 2 or int(shape[0]) != 1:
                raise ValueError("each shared V5 context requires one [1, tokens] prompt row")
            widths.add(int(shape[1]))
        if len(widths) != 1:
            raise ValueError("shared V5 contexts require equal frozen prompt widths")

    def execute(self, contexts: tuple[SearchContext, ...]) -> SharedExecution:
        self._validate(contexts)
        import torch

        # The concatenation order is frozen as cache-slot order.  The decoder
        # validates batched-cache size at every recursive expansion.
        input_ids = torch.cat([item.input_ids for item in contexts], dim=0).to(self.model.device)
        result = inference_frontier_floor_turbo_dfs(self.model, input_ids=input_ids, config=self.config)
        if len(result.candidates) != len(contexts):
            raise RuntimeError("shared V5 decoder returned an unexpected context count")
        return SharedExecution(contexts=contexts, result=result)
