"""Independent-state Regret DFS with deterministic dynamic ready batching.

This module deliberately implements a *new execution regime*, rather than
changing the bounded Regret decoder policy.  Each logical cell owns its own
DFS state, cache and recursion stack.  A scheduler may batch only requests
whose real KV cache geometry and position agree.  A lone request is forwarded
as physical batch one; the module never manufactures a padding lane.

The coroutine is a mechanically equivalent single-lane form of
``nvarc_turbodfs_d1.d1_turbo_dfs``.  It exists so that phase-zero can compare
the refactored B1 state machine with the authoritative scalar implementation
before a dynamic B2 execution is permitted.
"""
from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass
import hashlib
import json
import math
import time
from typing import Any, Callable, Literal

from inference.nvarc_turbodfs_d1 import (
    D1TurboDFSConfig,
    D1TurboDFSResult,
    _node,
    _prefix_hash,
    _record_frontier,
    _retained,
    _trace,
)
from inference.nvarc_turbodfs_reference import ReferenceTurboDFSCandidate, _cache_batch_size


@dataclass
class CacheOwner:
    """The one mutable ``DynamicCache`` owned by a logical DFS view.

    A recursive DFS frame owns only a sequence-length checkpoint.  It must
    never retain a child-generation cache object: descendants advance this
    owner in place and crop it back to their parent checkpoint on return.
    This is the critical ownership boundary for Dynamic Ready B2.
    """

    cache: Any


@dataclass(frozen=True)
class ReadyForwardRequest:
    """One real incremental forward request from one independent DFS cell."""

    cell_key: str
    ordinal: int
    token_id: int
    position: int
    cache_owner: CacheOwner
    cache_key: tuple[Any, ...]
    # The expanded node whose continuation logits this request will produce.
    # It lets optional diagnostics bind a reply to the exact DFS state without
    # changing scheduling or decoding behaviour.
    parent_node_id: int

    @property
    def cache(self) -> Any:
        """The current cache object, retained for existing scheduler callers."""
        return self.cache_owner.cache


@dataclass
class ReadyCell:
    """Mutable runtime for one DFS cell; it never shares decoder state."""

    cell_key: str
    config: D1TurboDFSConfig
    state: dict[str, Any]
    started_unix: float
    created_perf: float
    generator: Generator[ReadyForwardRequest, Any, list[ReferenceTurboDFSCandidate]]
    request: ReadyForwardRequest | None = None
    result: list[ReferenceTurboDFSCandidate] | None = None
    request_count: int = 0
    prompt_forwards: int = 1
    prefill_seconds: float = 0.0
    active_elapsed_seconds: float = 0.0
    cache_owner: CacheOwner | None = None


def ready_incremental_forward_kwargs(*, token_ids: list[int], position: int, cache: Any, device: Any) -> dict[str, Any]:
    """Build the complete Qwen continuation contract for ready-cell forwards.

    ``position_ids`` controls RoPE positions, while ``cache_position`` controls
    the Transformers causal-mask/cache update path.  They happen to carry the
    same absolute index for this one-token decoder, but are *not*
    interchangeable: omitting ``cache_position`` lets the patched Qwen
    runtime infer it from mutable call history.  Passing it explicitly keeps a
    ReadyForwardRequest self-contained and independent of unrelated cells.
    """
    import torch

    if not token_ids:
        raise ValueError("incremental forward requires at least one token")
    batch = len(token_ids)
    return {
        "input_ids": torch.tensor([[token] for token in token_ids], device=device, dtype=torch.long),
        "position_ids": torch.full((batch, 1), int(position), device=device, dtype=torch.long),
        # Transformers expects this as the sequence dimension, not one item
        # per batch lane.  Dynamic B2 only groups identical positions.
        "cache_position": torch.tensor([int(position)], device=device, dtype=torch.long),
        "past_key_values": cache,
        "return_dict": True,
        "use_cache": True,
    }


def _new_state(config: D1TurboDFSConfig) -> dict[str, Any]:
    """Exact one-lane state layout used by the historical scalar D1 route."""
    # The dynamic engine is intentionally a one-logical-lane state machine.
    # Independent counters are unnecessary because every state is its own cell.
    return {
        "nodes": [], "branch_probabilities": [], "frontier_floor_events": [],
        "frontier_samples": [], "search_trace": [], "model_forwards": 0,
        "tokens_advanced": 0, "completed_candidates": 0,
        "next_candidate_id": 0, "next_node_id": 0, "next_trace_event_id": 0,
        "next_frontier_insert_order": 0, "next_frontier_pop_order": 0,
        "trace_pending": {}, "trace_started_perf": time.perf_counter(),
        # B1.1 may account the fixed local time budget by work attributable to
        # this cell rather than by wall time since construction.  The default
        # remains the historical wall-clock predicate until explicitly enabled.
        "active_time_accounting": False, "active_elapsed_seconds": 0.0,
        "diagnostic_trace": config.diagnostic_trace, "model_forward_seconds": 0.0,
        "retained_successors_by_parent": {},
        # An opt-in observational hook used by the frozen Pilot18 protocol.
        # It is intentionally not part of the decoder configuration: callers
        # receive the exact same token decisions, frontier and cache path.  A
        # hook observes an EOS successor decision after the frozen retention
        # and floor semantics have been resolved; it never supplies a value
        # back to the coroutine.
        "eos_event_sink": None,
        "max_frontier_size": 0, "expanded_nodes": 0, "budget_exhausted": False,
        "independent_lane_budgets": False, "performance_profile": False,
        "performance_telemetry": {
            "batch_preparation_seconds": 0.0, "trace_telemetry_seconds": 0.0,
            "node_and_state_seconds": 0.0, "frontier_bookkeeping_seconds": 0.0,
            "kv_pack_or_select_seconds": 0.0,
        },
        # Kept for the shared helpers' fixed schema; never selected in this mode.
        "model_forwards_by_lane": [0], "tokens_advanced_by_lane": [0],
        "completed_candidates_by_lane": [0], "expanded_nodes_by_lane": [0],
        "budget_exhausted_by_lane": [False], "next_candidate_id_by_lane": [0],
        "next_node_id_by_lane": [0], "next_trace_event_id_by_lane": [0],
        "next_frontier_insert_order_by_lane": [0], "next_frontier_pop_order_by_lane": [0],
        "max_frontier_size_by_lane": [0], "frontier_floor_events_by_lane": [0],
    }


def _eos_checkpoint_region(nodes_expanded: int) -> str:
    """Return the frozen Pilot18 observational budget region."""
    if nodes_expanded <= 512:
        return "0_512"
    if nodes_expanded <= 1024:
        return "512_1024"
    if nodes_expanded <= 2048:
        return "1024_2048"
    return "2048_4096"


def _emit_eos_event(state: dict[str, Any], **event: Any) -> None:
    """Send compact EOS telemetry without participating in decoding.

    The callback is deliberately stored in runtime state rather than in
    ``D1TurboDFSConfig`` so the frozen policy/retention contract is unchanged.
    Its return value is ignored.  It is opt-in and absent from every existing
    execution path.
    """
    sink = state.get("eos_event_sink")
    if sink is not None:
        sink(event)


def _legacy_cache(cache: Any) -> tuple[tuple[Any, ...], ...]:
    """Return a non-mutating legacy KV view, or fail closed if unavailable."""
    if hasattr(cache, "to_legacy_cache"):
        cache = cache.to_legacy_cache()
    if not isinstance(cache, (tuple, list)) or not cache:
        raise RuntimeError("dynamic-ready requires a splittable legacy KV cache")
    rows: list[tuple[Any, ...]] = []
    for layer in cache:
        if not isinstance(layer, (tuple, list)) or not layer:
            raise RuntimeError("dynamic-ready encountered non-layer KV cache")
        if not all(hasattr(value, "shape") and len(value.shape) >= 1 for value in layer):
            raise RuntimeError("dynamic-ready encountered non-tensor KV cache")
        rows.append(tuple(layer))
    return tuple(rows)


def clone_legacy_cache(cache: Any) -> tuple[tuple[Any, ...], ...]:
    """Deep-clone a splittable root KV cache without retaining storage aliases.

    The dynamic-ready audit uses this only to test whether the root cache
    returned by a scalar prefill is the source of cross-cell interference.
    A legacy tuple is accepted by the same model path as the native cache
    object, while each tensor has independent storage.
    """
    return tuple(tuple(value.detach().clone() for value in layer) for layer in _legacy_cache(cache))


def _is_transformers_cache(cache: Any) -> bool:
    """Return whether this is a standard Transformers cache object.

    The stock Qwen3 implementation uses the cache API to construct its causal
    mask.  A legacy tuple is acceptable to older patched backends, but cannot
    be handed to stock Transformers 4.55 as an inference cache.
    """
    return hasattr(cache, "get_mask_sizes") and hasattr(cache, "to_legacy_cache")


def _restore_cache_kind(legacy: tuple[tuple[Any, ...], ...], exemplar: Any) -> Any:
    """Rebuild a cache of the same public representation as ``exemplar``."""
    if not _is_transformers_cache(exemplar):
        return legacy
    from transformers.cache_utils import DynamicCache

    return DynamicCache.from_legacy_cache(legacy)


def replace_cache_contents_in_place(
    destination: Any,
    source: Any,
    *,
    observer: Callable[[str, dict[str, Any]], None] | None = None,
    lane_index: int | None = None,
) -> None:
    """Copy ``source`` into an existing Transformers DynamicCache object.

    Dynamic B2 produces a temporary packed cache and temporary lane slices.
    A DFS lane may *adopt their contents* but must never adopt their cache
    object: suspended recursion frames and every ReadyForwardRequest refer to
    the lane's original owner object.  Transformers 4.55 stores dynamic
    layers as public ``layers`` objects with ``keys`` and ``values`` tensors.
    Replacing that list leaves the destination cache identity unchanged while
    a deep clone avoids retaining storage aliases to the temporary B2 result.

    This is intentionally not a zero-copy optimisation.  Ownership and
    bounded lifetime come first; the B2 memory plateau gate measures its cost.
    """
    if not _is_transformers_cache(destination):
        raise RuntimeError("in-place cache adoption requires a Transformers DynamicCache destination")
    legacy = _legacy_cache(source)
    from transformers.cache_utils import DynamicLayer

    destination_id = id(destination)
    layers = []
    for layer_index, layer in enumerate(legacy):
        if len(layer) != 2:
            raise RuntimeError("DynamicCache adoption expects key/value pairs")
        if observer is not None:
            observer("before_materialized_adopt_layer", {
                "lane_index": lane_index, "layer_index": layer_index,
            })
        key, value = layer
        layers.append(DynamicLayer.from_tensors(key.detach().clone(), value.detach().clone()))
        if observer is not None:
            observer("after_materialized_adopt_layer", {
                "lane_index": lane_index, "layer_index": layer_index,
            })
    destination.layers = layers
    if id(destination) != destination_id:
        raise AssertionError("cache adoption replaced the owner object")


def cache_geometry(cache: Any) -> tuple[Any, ...]:
    """Stable compatibility key excluding the real batch dimension."""
    legacy = _legacy_cache(cache)
    geometry: list[Any] = [len(legacy)]
    for layer in legacy:
        geometry.append(tuple((str(value.dtype), str(value.device), tuple(int(x) for x in value.shape[1:])) for value in layer))
    return tuple(geometry)


def cache_sha256(cache: Any) -> str:
    """Hash public cache contents for optional parity diagnostics only."""
    digest = hashlib.sha256()
    for layer in _legacy_cache(cache):
        for value in layer:
            tensor = value.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(repr(tuple(int(item) for item in tensor.shape)).encode("ascii"))
            digest.update(tensor.view(__import__("torch").uint8).numpy().tobytes())
    return digest.hexdigest()


def _logits_diagnostic(logits: Any, arc_tokens: tuple[int, ...]) -> dict[str, Any]:
    """Return the exact model output evidence required to locate B1 drift."""
    import torch

    last = logits[0, -1].detach()
    values = torch.log_softmax(last.float(), dim=-1)
    ranking = torch.argsort(last, descending=True).detach().cpu().tolist()
    raw = last.cpu().contiguous()
    return {
        "full_logits_sha256": hashlib.sha256(raw.view(torch.uint8).numpy().tobytes()).hexdigest(),
        "arc_logits": {str(token): float(last[token].item()) for token in arc_tokens},
        "arc_logprobs": {str(token): float(values[token].item()) for token in arc_tokens},
        "arc_ranking": [int(token) for token in ranking if int(token) in arc_tokens],
    }


def _cat_caches(
    caches: list[Any],
    *,
    observer: Callable[[str, dict[str, Any]], None] | None = None,
) -> tuple[tuple[Any, ...], ...]:
    import torch

    if not caches:
        raise ValueError("cannot concatenate no caches")
    legacy = [_legacy_cache(cache) for cache in caches]
    if observer is not None:
        observer("legacy_views", {"legacy": legacy})
    reference = cache_geometry(legacy[0])
    if any(cache_geometry(item) != reference for item in legacy[1:]):
        raise RuntimeError("attempted dynamic batch with incompatible real KV caches")
    packed_layers = []
    for layer_i in range(len(legacy[0])):
        packed = tuple(torch.cat([item[layer_i][part_i] for item in legacy], dim=0)
                       for part_i in range(len(legacy[0][layer_i])))
        packed_layers.append(packed)
        if observer is not None and layer_i in {0, 8, 16, 24, 35, len(legacy[0]) - 1}:
            observer("packed_layer", {"layer_index": layer_i, "legacy": legacy, "packed_layers": packed_layers})
    merged = tuple(packed_layers)
    if observer is not None:
        observer("merged_legacy", {"legacy": legacy, "merged": merged})
    return merged


def _split_cache(
    cache: Any,
    expected: int,
    *,
    observer: Callable[[str, dict[str, Any]], None] | None = None,
) -> list[tuple[tuple[Any, ...], ...]]:
    """Materialize independent legacy cache slices for the historical path.

    The optional observer is diagnostic-only.  It records the last layer/lane
    about to allocate so a controlled OOM waterfall can name the exact
    transient phase without hashing or retaining additional tensors.
    """
    legacy = _legacy_cache(cache)
    size = _cache_batch_size(legacy)
    if size != expected:
        raise RuntimeError(f"dynamic-ready output cache batch mismatch {size} != {expected}")
    lanes: list[list[tuple[Any, ...]]] = [[] for _ in range(expected)]
    for layer_index, layer in enumerate(legacy):
        for lane_index in range(expected):
            if observer is not None:
                observer("before_split_tensor", {
                    "layer_index": layer_index,
                    "lane_index": lane_index,
                    "split_lane_count": len(lanes[lane_index]),
                })
            lane_layer = tuple(value[lane_index:lane_index + 1].contiguous() for value in layer)
            lanes[lane_index].append(lane_layer)
            if observer is not None:
                observer("after_split_tensor", {
                    "layer_index": layer_index,
                    "lane_index": lane_index,
                    "split_lane_count": len(lanes[lane_index]),
                })
    return [tuple(lane) for lane in lanes]


def replace_cache_contents_streaming_in_place(
    destination: Any,
    source: Any,
    *,
    observer: Callable[[str, dict[str, Any]], None] | None = None,
    lane_index: int | None = None,
) -> None:
    """Adopt one lane progressively without a full temporary layer list.

    Unlike :func:`replace_cache_contents_in_place`, this replaces each public
    ``DynamicLayer`` immediately after cloning it.  The CacheOwner object is
    preserved, but the old layer becomes reclaimable before the next source
    layer is cloned.  It is used only by the opt-in hardware memory path.
    """
    if not _is_transformers_cache(destination):
        raise RuntimeError("streaming cache adoption requires a Transformers DynamicCache destination")
    legacy = _legacy_cache(source)
    from transformers.cache_utils import DynamicLayer

    destination_id = id(destination)
    for layer_index, layer in enumerate(legacy):
        if len(layer) != 2:
            raise RuntimeError("DynamicCache streaming adoption expects key/value pairs")
        if observer is not None:
            observer("before_streaming_adopt_layer", {"lane_index": lane_index, "layer_index": layer_index})
        key, value = layer
        adopted_layer = DynamicLayer.from_tensors(key.detach().clone(), value.detach().clone())
        if layer_index < len(destination.layers):
            destination.layers[layer_index] = adopted_layer
        else:
            destination.layers.append(adopted_layer)
        if observer is not None:
            observer("after_streaming_adopt_layer", {"lane_index": lane_index, "layer_index": layer_index})
    if len(destination.layers) > len(legacy):
        del destination.layers[len(legacy):]
    if id(destination) != destination_id:
        raise AssertionError("streaming cache adoption replaced the owner object")


def _streaming_split_and_adopt(
    cache: Any,
    requests: list[ReadyForwardRequest],
    *,
    observer: Callable[[str, dict[str, Any]], None] | None = None,
) -> None:
    """Consume each output lane immediately instead of retaining ``split_legacy``.

    Lane slices are views of the batched output cache.  Streaming adoption
    deep-clones their contents into the existing owner object before the next
    lane is considered, so no complete list of B16 temporary split copies is
    materialized.  The production default remains the historical materialized
    path unless an explicit caller opts in.
    """
    legacy = _legacy_cache(cache)
    size = _cache_batch_size(legacy)
    if size != len(requests):
        raise RuntimeError(f"dynamic-ready output cache batch mismatch {size} != {len(requests)}")
    if observer is not None:
        observer("before_streaming_split", {"physical_batch": size})
    for lane_index, request in enumerate(requests):
        if observer is not None:
            observer("before_streaming_lane_view", {"lane_index": lane_index})
        lane_view = tuple(tuple(value[lane_index:lane_index + 1] for value in layer) for layer in legacy)
        if observer is not None:
            observer("after_streaming_lane_view", {"lane_index": lane_index})
        owner_cache = request.cache_owner.cache
        owner_id = id(owner_cache)
        if hasattr(owner_cache, "adopt_legacy_suffix"):
            owner_cache.adopt_legacy_suffix(lane_view)
            if id(request.cache_owner.cache) != owner_id:
                raise RuntimeError("streaming cache adoption replaced a chunked logical owner object")
            del lane_view
            if observer is not None:
                observer("after_streaming_lane_adoption", {"lane_index": lane_index})
            continue
        if _is_transformers_cache(owner_cache):
            replace_cache_contents_streaming_in_place(
                owner_cache, lane_view, observer=observer, lane_index=lane_index,
            )
            if id(request.cache_owner.cache) != owner_id:
                raise RuntimeError("streaming cache adoption replaced a logical owner object")
        else:
            # CPU-only compatibility for deterministic unit tests.  Production
            # Clean-HF takes the DynamicCache branch above.
            request.cache_owner.cache = tuple(
                tuple(value.detach().clone() for value in layer) for layer in lane_view
            )
        del lane_view
        if observer is not None:
            observer("after_streaming_lane_adoption", {"lane_index": lane_index})
    if observer is not None:
        observer("after_streaming_split_complete", {"physical_batch": size})


def _ready_dfs(
    *, cell_key: str, logits: Any, max_new_tokens: int, score: float, regret: float,
    pos: int, cache_owner: CacheOwner, config: D1TurboDFSConfig, started_unix: float,
    state: dict[str, Any], parent_node: int | None, prefix: tuple[int, ...],
    ordinal: list[int], cache_strategy: Literal["rollback", "snapshot"],
) -> Generator[ReadyForwardRequest, Any, list[ReferenceTurboDFSCandidate]]:
    """Single-lane coroutine equivalent to the D1 recursive search function."""
    import torch

    # In production ``rollback`` mode this local reference always points to
    # the one logical view owner.  ``snapshot`` is a diagnostic-only oracle:
    # it clones this parent cache for each child request, deliberately trading
    # memory for immutable branch state.
    cache = cache_owner.cache
    entry_sequence_length = int(_legacy_cache(cache)[0][0].shape[-2])
    if config.calibration_assertions:
        cached = _cache_batch_size(cache)
        if cached is not None and cached != 1:
            raise RuntimeError(f"dynamic-ready scalar cache batch mismatch {cached}")
    log_probs = logits.float().cpu().log_softmax(-1)[0]
    values = [(token, float(log_probs[token].item())) for token in config.arc_tokens]
    ranked = sorted(values, key=lambda pair: (-pair[1], pair[0]))
    probabilities = [math.exp(logprob) for _token, logprob in values]
    suffixes: list[ReferenceTurboDFSCandidate] = []
    candidates: list[tuple[float, float, int, bool, int | None]] = []
    _record_frontier(state, [candidates])
    state["branch_probabilities"].append({
        "forward_index": state["model_forwards"], "lane": 0, "parent_node_id": parent_node,
        "token_position": pos, "prefix_length": len(prefix), "cumulative_score_before": score,
        "cumulative_regret_before": regret,
        "full_arc_logprobs": [{"token_id": token, "logprob": logprob} for token, logprob in values],
        "top1_token_id": ranked[0][0], "top1_logprob": ranked[0][1],
        "top2_token_id": ranked[1][0], "top2_logprob": ranked[1][1],
        "margin": ranked[0][1] - ranked[1][1],
        "entropy": -sum(p * lp for p, (_token, lp) in zip(probabilities, values, strict=True)),
        "batch_size": 1, "active_lane_count": 1, "active_mask": (True,),
        "decoder_policy": config.policy_id,
    })
    kept, prune_reason = _retained(config, ranked, score_before=score, regret_before=regret,
                                   remaining=max_new_tokens, generated_length=len(prefix))
    state["retained_successors_by_parent"][parent_node] = sorted(int(value[2]) for value in kept)
    legal_tokens = {token for token, _lp in values if token == 15 or max_new_tokens > 1}
    kept_tokens = {value[2] for value in kept}
    for token, logprob in values:
        rank = next(index for index, (ranked_token, _x) in enumerate(ranked, start=1) if ranked_token == token)
        trace_common = {
            "event_kind": "successor", "parent_node_id": parent_node, "lane": 0,
            "prefix_length": len(prefix) + 1, "prefix_hash": _prefix_hash(prefix + (token,)),
            "last_token": token, "local_token_rank": rank, "token_logprob": logprob,
            "path_cumulative_nll": score - logprob,
            "path_cumulative_regret": regret + ranked[0][1] - logprob,
            "frontier_insert_order": None, "frontier_pop_order": None,
            "frontier_size_at_insert": None, "frontier_size_at_pop": None,
            "successors_considered": len(legal_tokens), "successors_retained": len(kept),
            "prune_reason": None, "dedup_rejected": False, "dedup_reason": None,
            "candidate_completed": False, "candidate_completion_index": None,
            "nodes_expanded_so_far": state["expanded_nodes"],
            "candidates_completed_so_far": state["completed_candidates"],
        }
        if token not in legal_tokens:
            _node(state, parent_node_id=parent_node, lane=0, token_position=pos, branch_depth=len(prefix) + 1,
                  selected_token=token, token_logprob=logprob, cumulative_score=score - logprob,
                  state="pruned", prune_reason="max_new_tokens", termination_reason="max_new_tokens", branch_rank=None)
            _trace(state, **{**trace_common, "prune_reason": "max_new_tokens"})
        elif token not in kept_tokens:
            _node(state, parent_node_id=parent_node, lane=0, token_position=pos, branch_depth=len(prefix) + 1,
                  selected_token=token, token_logprob=logprob, cumulative_score=score - logprob,
                  cumulative_regret=regret + ranked[0][1] - logprob, state="pruned",
                  prune_reason=prune_reason, termination_reason=None, branch_rank=None)
            _trace(state, **{**trace_common, "prune_reason": prune_reason})
    restored = False
    if not kept and legal_tokens:
        all_legal = [(score - logprob, regret + ranked[0][1] - logprob, token, logprob)
                     for token, logprob in ranked if token in legal_tokens]
        kept = sorted(all_legal, key=lambda value: (value[0], value[2]))[:config.frontier_floor]
        restored = True
        state["frontier_floor_events"].append({
            "activation_index": len(state["frontier_floor_events"]), "lane": 0,
            "parent_node_id": parent_node, "token_position": pos, "prefix_length": len(prefix),
            "restored_count": len(kept), "restored_tokens": [value[2] for value in kept],
            "restored_cumulative_nll": [value[0] for value in kept],
            "reason": "policy_pruning_empty_frontier", "decoder_policy": config.policy_id,
        })

    # EOS is one legal successor decision, not a separate decode branch.  The
    # event below is emitted only after the ordinary frozen retention and
    # frontier-floor logic has finished deciding its status.  This makes every
    # event mutually exclusive while leaving the token loop untouched.
    eos_logprob = next(logprob for token, logprob in values if token == 15)
    eos_rank = next(index for index, (token, _logprob) in enumerate(ranked, start=1) if token == 15)
    eos_legal = 15 in legal_tokens
    eos_retained = any(token == 15 for _score, _regret, token, _logprob in kept)

    def emit_eos(primary_class: str, *, retained: bool, floor_restore_rank: int | None,
                 candidate_completed: bool, candidate_completion_index: int | None,
                 raw_prune_reason: str | None, raw_termination_reason: str | None,
                 candidate_terminal_node_id: int | None = None) -> None:
        _emit_eos_event(
            state,
            cell_key=cell_key,
            parent_node_id=parent_node,
            parent_prefix_hash=_prefix_hash(prefix),
            token_position=pos,
            prefix_length=len(prefix),
            nodes_expanded_so_far=int(state["expanded_nodes"]),
            candidates_completed_so_far=int(state["completed_candidates"]),
            model_forwards_so_far=int(state["model_forwards"]),
            eos_token_id=15,
            eos_local_rank=eos_rank,
            eos_logprob=eos_logprob,
            best_token_logprob=ranked[0][1],
            eos_logprob_gap_from_best=ranked[0][1] - eos_logprob,
            eos_path_cumulative_nll=score - eos_logprob,
            eos_path_cumulative_regret=regret + ranked[0][1] - eos_logprob,
            eos_legal=eos_legal,
            eos_retained=retained,
            frontier_floor_activated=restored,
            frontier_floor_restore_rank=floor_restore_rank,
            candidate_completed=candidate_completed,
            candidate_completion_index=candidate_completion_index,
            candidate_terminal_node_id=candidate_terminal_node_id,
            prune_reason=raw_prune_reason,
            termination_reason=raw_termination_reason,
            eos_primary_class=primary_class,
            checkpoint_region=_eos_checkpoint_region(int(state["expanded_nodes"])),
        )

    if not eos_legal:
        emit_eos("EOS_MAX_NEW_TOKENS_BLOCKED", retained=False, floor_restore_rank=None,
                 candidate_completed=False, candidate_completion_index=None,
                 raw_prune_reason="max_new_tokens", raw_termination_reason="max_new_tokens")
    elif not eos_retained:
        emit_eos("EOS_POLICY_PRUNED", retained=False, floor_restore_rank=None,
                 candidate_completed=False, candidate_completion_index=None,
                 raw_prune_reason=prune_reason, raw_termination_reason=None)
    for restore_rank, (next_score, next_regret, token, logprob) in enumerate(kept, start=1):
        rank = next(index for index, (ranked_token, _x) in enumerate(ranked, start=1) if ranked_token == token)
        retained_trace = {
            "event_kind": "successor", "parent_node_id": parent_node, "lane": 0,
            "prefix_length": len(prefix) + 1, "prefix_hash": _prefix_hash(prefix + (token,)),
            "last_token": token, "local_token_rank": rank, "token_logprob": logprob,
            "path_cumulative_nll": next_score, "path_cumulative_regret": next_regret,
            "frontier_insert_order": None, "frontier_pop_order": None,
            "frontier_size_at_insert": None, "frontier_size_at_pop": None,
            "successors_considered": len(legal_tokens), "successors_retained": len(kept),
            "prune_reason": None, "dedup_rejected": False, "dedup_reason": None,
            "candidate_completed": False, "candidate_completion_index": None,
            "nodes_expanded_so_far": state["expanded_nodes"],
            "candidates_completed_so_far": state["completed_candidates"],
        }
        if token == 15:
            if state["completed_candidates"] >= config.max_completed_candidates:
                state["budget_exhausted"] = True
                _node(state, parent_node_id=parent_node, lane=0, token_position=pos, branch_depth=len(prefix) + 1,
                      selected_token=token, token_logprob=logprob, cumulative_score=next_score,
                      cumulative_regret=next_regret, state="pruned", prune_reason="candidate_budget",
                      termination_reason="candidate_budget", branch_rank=None)
                _trace(state, **{**retained_trace, "prune_reason": "candidate_budget"})
                emit_eos("EOS_CANDIDATE_BUDGET_PRUNED", retained=True,
                         floor_restore_rank=restore_rank if restored else None,
                         candidate_completed=False, candidate_completion_index=None,
                         raw_prune_reason="candidate_budget", raw_termination_reason="candidate_budget")
                continue
            node_id = _node(state, parent_node_id=parent_node, lane=0, token_position=pos, branch_depth=len(prefix) + 1,
                            selected_token=token, token_logprob=logprob, cumulative_score=next_score,
                            cumulative_regret=next_regret, state="completed", prune_reason=None,
                            termination_reason="eos", branch_rank=None, frontier_floor_activated=restored,
                            frontier_floor_restore_rank=restore_rank if restored else None)
            candidate_id = state["next_candidate_id"]; state["next_candidate_id"] += 1
            suffixes.append(ReferenceTurboDFSCandidate(candidate_id, (15,), next_score, node_id,
                                                         state["model_forwards"], time.time()))
            _trace(state, **{**retained_trace, "candidate_completed": True,
                             "candidate_completion_index": candidate_id})
            emit_eos("EOS_FRONTIER_FLOOR_RESTORED_AND_COMPLETED" if restored else "EOS_COMPLETED",
                     retained=True, floor_restore_rank=restore_rank if restored else None,
                     candidate_completed=True, candidate_completion_index=candidate_id,
                     raw_prune_reason=None, raw_termination_reason="eos",
                     candidate_terminal_node_id=node_id)
            state["completed_candidates"] += 1
        else:
            candidates.append((next_score, next_regret, token, restored, restore_rank if restored else None))
            insert_order = state["next_frontier_insert_order"]; state["next_frontier_insert_order"] += 1
            event = _trace(state, **{**retained_trace, "frontier_insert_order": insert_order,
                                     "frontier_size_at_insert": len(candidates)})
            if event is not None:
                state["trace_pending"][(parent_node, token, next_score)] = event
    candidates.sort(key=lambda value: (value[0], value[2]))
    _record_frontier(state, [candidates])
    elapsed_budget = (lambda: float(state["active_elapsed_seconds"])) if state["active_time_accounting"] else (lambda: time.time() - started_unix)
    while (not state["budget_exhausted"] and elapsed_budget() < config.local_time_limit_seconds and
           (config.absolute_end_time_unix is None or time.time() < config.absolute_end_time_unix)):
        if not candidates or state["expanded_nodes"] >= config.max_expanded_nodes:
            if state["expanded_nodes"] >= config.max_expanded_nodes:
                state["budget_exhausted"] = True
            break
        next_score, next_regret, token, restored, restore_rank = candidates.pop(0)
        node_id = _node(state, parent_node_id=parent_node, lane=0, token_position=pos, branch_depth=len(prefix) + 1,
                        selected_token=token, token_logprob=-(next_score - score), cumulative_score=next_score,
                        cumulative_regret=next_regret, state="expanded", prune_reason=None,
                        termination_reason=None, branch_rank=1, frontier_floor_activated=restored,
                        frontier_floor_restore_rank=restore_rank)
        state["expanded_nodes"] += 1
        trace_event = state["trace_pending"].pop((parent_node, token, next_score), None)
        if trace_event is not None:
            trace_event.update({"frontier_pop_order": state["next_frontier_pop_order"],
                                "frontier_size_at_pop": len(candidates), "expanded_node_id": node_id,
                                "nodes_expanded_so_far": state["expanded_nodes"],
                                "elapsed_seconds": time.perf_counter() - state["trace_started_perf"]})
            state["next_frontier_pop_order"] += 1
        _record_frontier(state, [candidates])
        if cache_strategy == "rollback":
            # A child exploration may have advanced this single mutable cache.
            # Restore the current frame's parent state before every sibling.
            if _is_transformers_cache(cache_owner.cache):
                cache_owner.cache.crop(entry_sequence_length)
            elif int(_legacy_cache(cache_owner.cache)[0][0].shape[-2]) != entry_sequence_length:
                raise RuntimeError("rollback DFS requires a croppable Transformers DynamicCache")
            if cache_owner.cache is not cache:
                raise RuntimeError("DFS cache owner object changed during backtracking")
            forward_owner = cache_owner
        else:
            forward_owner = CacheOwner(_restore_cache_kind(clone_legacy_cache(cache_owner.cache), cache_owner.cache))
        request = ReadyForwardRequest(
            cell_key, ordinal[0], token, pos, forward_owner, cache_geometry(forward_owner.cache), node_id,
        )
        ordinal[0] += 1
        outputs = yield request
        if config.calibration_assertions and int(outputs.logits.shape[0]) != 1:
            raise RuntimeError("dynamic-ready B1 reply must have one logical lane")
        if _is_transformers_cache(forward_owner.cache):
            if outputs.past_key_values is not forward_owner.cache:
                raise RuntimeError("B1 reply did not update the logical cache owner in place")
        else:
            # Legacy tuples exist only in CPU compatibility tests and the
            # immutable snapshot oracle.  They cannot support crop-based
            # production rollback, so their branch-local owner may advance to
            # the model-returned tuple.
            forward_owner.cache = outputs.past_key_values
        state["model_forwards"] += 1; state["tokens_advanced"] += 1
        descendants = yield from _ready_dfs(
            cell_key=cell_key, logits=outputs.logits[:, -1], max_new_tokens=max_new_tokens - 1,
            score=next_score, regret=next_regret, pos=pos + 1, cache_owner=forward_owner,
            config=config, started_unix=started_unix, state=state, parent_node=node_id,
            prefix=prefix + (token,), ordinal=ordinal, cache_strategy=cache_strategy,
        )
        suffixes.extend(ReferenceTurboDFSCandidate(item.candidate_id, (token,) + item.token_ids,
                                                    item.cumulative_nll, item.terminal_node_id,
                                                    item.discovery_forward_index, item.discovery_unix)
                        for item in descendants)
        # The child restores its own entry state (this token included); remove
        # that token before this parent moves to the next sibling or returns.
        if cache_strategy == "rollback":
            cache_owner.cache.crop(entry_sequence_length)
    return suffixes


def start_ready_cell(*, model: Any, input_ids: Any, config: D1TurboDFSConfig, cell_key: str,
                     normalize_root_cache: bool, active_time_accounting: bool = False,
                     root_cache_transform: Callable[[Any], Any] | None = None,
                     cache_strategy: Literal["rollback", "snapshot"] = "rollback",
                     release_prefill_temporaries: bool = False,
                     prefill_output_references: list[Any] | None = None,
                     eos_event_sink: Callable[[dict[str, Any]], None] | None = None) -> ReadyCell:
    """Run the required scalar prefill, then expose the first incremental request."""
    import torch

    if input_ids.ndim != 2 or tuple(input_ids.shape[:1]) != (1,):
        raise ValueError("dynamic-ready cells require one prompt at a time")
    state = _new_state(config)
    state["eos_event_sink"] = eos_event_sink
    started_unix = time.time()
    created_perf = time.perf_counter()
    state["active_time_accounting"] = active_time_accounting
    state["per_forward_trace"] = []
    with torch.no_grad():
        started = time.perf_counter()
        outputs = model(input_ids=input_ids, return_dict=True, use_cache=True)
        prefill_seconds = time.perf_counter() - started
        state["model_forward_seconds"] += prefill_seconds
        state["active_elapsed_seconds"] += prefill_seconds
    state["model_forwards"] = 1
    root = _node(state, parent_node_id=None, lane=0, token_position=int(input_ids.size(1)), branch_depth=0,
                 selected_token=None, token_logprob=None, cumulative_score=0.0, cumulative_regret=0.0,
                 state="root", prune_reason=None, termination_reason=None, branch_rank=None)
    root_cache = _legacy_cache(outputs.past_key_values) if normalize_root_cache else outputs.past_key_values
    if root_cache_transform is not None:
        root_cache = root_cache_transform(root_cache)
    # The root coroutine needs only the final-position logits.  A slice is a
    # view, so retaining it can retain the whole prompt-length logits tensor.
    # The opt-in hygiene path materializes the identical final-position values
    # in independent storage and lets a caller explicitly retain the original
    # output only long enough to measure its lifetime.  It never changes the
    # cache owner, prompt, token choice, or DFS state machine.
    root_logits = outputs.logits[:, -1]
    if release_prefill_temporaries:
        root_logits = root_logits.clone()
        if prefill_output_references is not None:
            prefill_output_references.append(outputs)
    if cache_strategy == "rollback" and not _is_transformers_cache(root_cache):
        # Preserve the old CPU-only legacy-cache tests without claiming that
        # such a tuple implements production rollback semantics.
        cache_strategy = "snapshot"
    state["cache_strategy"] = cache_strategy
    cache_owner = CacheOwner(root_cache)
    # This is deliberately observational. The forensic runner needs to
    # distinguish a prefill mismatch from a later shared-execution mismatch,
    # but ordinary production runs must not retain prompt/logit diagnostics.
    if config.diagnostic_trace:
        prompt_tokens = [int(token) for token in input_ids[0].detach().cpu().tolist()]
        state["prefill_trace"] = {
            "cell_key": cell_key,
            "prompt_token_ids": prompt_tokens,
            "prompt_token_sha256": hashlib.sha256(
                ",".join(str(token) for token in prompt_tokens).encode("ascii")
            ).hexdigest(),
            "prompt_length": len(prompt_tokens),
            "cache_type": type(root_cache).__module__ + "." + type(root_cache).__qualname__,
            "root_cache_sha256": cache_sha256(root_cache),
            "root_cache_geometry": repr(cache_geometry(root_cache)),
            "root_cache_valid_length": _request_cache_length(root_cache),
            "cache_owner_id": id(cache_owner),
            **_logits_diagnostic(outputs.logits, config.arc_tokens),
        }
    generator = _ready_dfs(cell_key=cell_key, logits=root_logits, max_new_tokens=config.max_new_tokens,
                           score=0.0, regret=0.0, pos=int(input_ids.size(1)), cache_owner=cache_owner, config=config,
                           started_unix=started_unix, state=state, parent_node=root, prefix=tuple(), ordinal=[0],
                           cache_strategy=cache_strategy)
    cell = ReadyCell(cell_key, config, state, started_unix, created_perf, generator,
                     prefill_seconds=prefill_seconds, active_elapsed_seconds=prefill_seconds,
                     cache_owner=cache_owner)
    _advance_cell(cell)
    if config.diagnostic_trace:
        request = cell.request
        state["prefill_trace"]["first_ready_request"] = None if request is None else {
            "request_ordinal": int(request.ordinal),
            "token_id": int(request.token_id),
            "position": int(request.position),
            "parent_node_id": request.parent_node_id,
            "cache_compatibility_key": repr(request.cache_key),
            "cache_owner_id": id(request.cache_owner),
            "valid_kv_length": _request_cache_length(request.cache),
        }
    if release_prefill_temporaries and prefill_output_references is None:
        del outputs
    return cell


def _advance_cell(cell: ReadyCell, outputs: Any | None = None) -> None:
    """Advance one coroutine segment and charge only its own CPU work."""
    started = time.perf_counter()
    try:
        cell.request = next(cell.generator) if outputs is None else cell.generator.send(outputs)
    except StopIteration as completed:
        cell.result = completed.value
        cell.request = None
    elapsed = time.perf_counter() - started
    cell.active_elapsed_seconds += elapsed
    cell.state["active_elapsed_seconds"] += elapsed


def _reply(cell: ReadyCell, outputs: Any) -> None:
    _advance_cell(cell, outputs)
    cell.request_count += 1


SchedulingPolicy = Literal[
    "serial", "round_robin", "dynamic_ready", "memory_aware_aug16", "fixed_b8_aug16",
]


@dataclass(frozen=True)
class MemoryAwareAug16Config:
    """Frozen physical-scheduling policy for the sixteen-lane AUG16 run.

    This is deliberately an executor-only contract.  It carries no decoder,
    prompt, candidate, or TTT setting: callers still create the exact same
    ``ReadyCell`` instances and this module still uses the one common
    pack/forward/streaming-adopt path.
    """

    soft_peak_allocated_bytes: int
    hard_peak_allocated_bytes: int
    logical_lane_count: int = 16
    split_batch_size: int = 8

    def __post_init__(self) -> None:
        if self.soft_peak_allocated_bytes <= 0:
            raise ValueError("memory-aware soft threshold must be positive")
        if self.hard_peak_allocated_bytes < self.soft_peak_allocated_bytes:
            raise ValueError("memory-aware hard threshold must be >= soft threshold")
        if self.logical_lane_count != 16 or self.split_batch_size != 8:
            raise ValueError("MEMORY_AWARE_AUG16 is frozen to sixteen lanes split into B8 groups")


@dataclass(frozen=True)
class FixedB8Aug16Config:
    """Frozen PROFILE_M executor: sixteen resident lanes, fair physical B8.

    Unlike ``MemoryAwareAug16Config``, this policy never attempts a B16
    forward and does not inspect CUDA memory to decide its schedule.  It is
    the table-driven PROFILE_M implementation, not a fallback heuristic.
    """

    logical_lane_count: int = 16
    split_batch_size: int = 8

    def __post_init__(self) -> None:
        if self.logical_lane_count != 16 or self.split_batch_size != 8:
            raise ValueError("FIXED_B8_AUG16 is frozen to sixteen lanes split into B8 groups")


def _request_cache_length(cache: Any) -> int | None:
    """Return the sequence dimension for telemetry without changing cache state."""
    try:
        legacy = _legacy_cache(cache)
        return int(legacy[0][0].shape[-2])
    except Exception:  # pragma: no cover - telemetry must not alter decoding
        return None


def _compatible_group_in_frozen_order(cells: list[ReadyCell], limit: int) -> list[ReadyCell]:
    """Choose the largest compatible READY subset, breaking ties by input order."""
    ready = [cell for cell in cells if cell.request is not None]
    if not ready:
        return []
    candidates: list[list[ReadyCell]] = []
    for seed in ready:
        assert seed.request is not None
        group = [
            other for other in ready
            if other.request is not None
            and other.request.cache_key == seed.request.cache_key
            and other.request.position == seed.request.position
        ]
        candidates.append(group[:limit])
    # ``max`` preserves the first frozen-order candidate on an equal-size tie.
    return max(candidates, key=len)


def _select_memory_aware_aug16(
    *,
    cells: list[ReadyCell],
    b16_enabled: bool,
    group_cursor: int,
    capture_ready_keys: bool,
) -> tuple[list[ReadyCell], int, str, str | None, list[str]]:
    """Select a B16 lane set or one fair B8 group using frozen cell order."""
    ready_keys = [cell.cell_key for cell in cells if cell.request is not None] if capture_ready_keys else []
    if not ready_keys and not any(cell.request is not None for cell in cells):
        return [], group_cursor, "B16_ENABLED" if b16_enabled else "B8_SPLIT_MODE", None, ready_keys
    if b16_enabled:
        return (
            _compatible_group_in_frozen_order(cells, 16),
            group_cursor,
            "B16_ENABLED",
            None,
            ready_keys,
        )

    groups = (cells[:8], cells[8:])
    for offset in range(2):
        selected_group = (group_cursor + offset) % 2
        selected = _compatible_group_in_frozen_order(list(groups[selected_group]), 8)
        if selected:
            # Move after the group actually served.  If its peer was empty,
            # the next call retries the skipped group first, then serves this
            # one again only when the peer remains naturally unavailable.
            return selected, (selected_group + 1) % 2, "B8_SPLIT_MODE", "A" if selected_group == 0 else "B", ready_keys
    raise AssertionError("memory-aware ready set was nonempty but no B8 group was selected")


def _select_ready_cells(
    *,
    cells: list[ReadyCell],
    scheduling_policy: SchedulingPolicy,
    round_robin_cursor: int,
    capture_ready_keys: bool = True,
) -> tuple[list[ReadyCell], int | None, int | None, list[str]]:
    """Select the next logical B1 lane without changing its physical forward.

    ``serial`` deliberately preserves the historical lexicographic scheduling
    behaviour.  ``round_robin`` instead treats the supplied cell list as the
    frozen logical order and advances its cursor after *every* B1 reply.  A
    completed cell is the only cell that may be skipped by round-robin.

    ``dynamic_ready`` keeps the deterministic lexical ready set used by the
    future B2 scheduler; it is intentionally separate so a caller cannot
    accidentally call that behaviour "round robin".
    """
    ready = [cell for cell in cells if cell.request is not None]
    if not ready:
        return [], None, None, []
    if scheduling_policy == "round_robin":
        if not cells:
            return [], None, None, []
        for offset in range(len(cells)):
            index = (round_robin_cursor + offset) % len(cells)
            cell = cells[index]
            if cell.request is not None:
                keys = [item.cell_key for item in ready] if capture_ready_keys else []
                return [cell], index, (index + 1) % len(cells), keys
        raise AssertionError("ready set was nonempty but no round-robin cell was selected")
    ordered = sorted(
        ready,
        key=lambda cell: (cell.cell_key, cell.request.ordinal if cell.request else -1),
    )
    keys = [item.cell_key for item in ordered] if capture_ready_keys else []
    return [ordered[0]], None, None, keys


def execute_ready_forward(
    *,
    model: Any,
    selected: list[ReadyCell],
    requests: list[ReadyForwardRequest],
    cache_pack_observer: Callable[[str, dict[str, Any]], None] | None = None,
    release_batch_temporaries_for_audit: bool = False,
    streaming_split_and_adopt: bool = False,
) -> tuple[list[Any], dict[str, float]]:
    """Execute one real physical ready-cell forward for independent lanes.

    The production scheduler and bounded B4 characterization use this exact
    pack/forward/split/adoption path.  It deliberately owns neither READY
    selection nor coroutine resumption: callers retain those responsibilities
    so a micro benchmark can inspect a single physical forward without
    altering the DFS scheduler policy.
    """
    import torch

    if not selected or len(selected) != len(requests):
        raise ValueError("selected cells and requests must be nonempty and aligned")
    # The cache pack/split/adoption protocol is lane-count agnostic.  Keep a
    # concrete upper bound solely as a malformed-caller guard; hardware
    # characterization intentionally exercises B8/B12/B16 with independent
    # cache owners, while the scientific B1--B4 scheduler remains unchanged.
    if len(selected) > 16:
        raise ValueError("physical ready-cell batches above sixteen are not implemented")
    if any(request is None for request in requests):  # pragma: no cover - type guard for external callers
        raise ValueError("physical ready-cell forward received an empty request")
    first = selected[0]
    if any(request.cache_key != requests[0].cache_key or request.position != requests[0].position
           for request in requests[1:]):
        raise RuntimeError("attempted physical batch with incompatible ready-cell requests")

    started = time.perf_counter()
    cache_pack_seconds = 0.0
    cache_adoption_seconds = 0.0
    with torch.no_grad():
        if len(selected) == 1:
            request = requests[0]
            model_started = time.perf_counter()
            outputs = model(**ready_incremental_forward_kwargs(
                token_ids=[request.token_id], position=request.position,
                cache=request.cache, device=model.device,
            ))
            model_call_seconds = time.perf_counter() - model_started
            outputs_by_cell = [outputs]
        else:
            pack_started = time.perf_counter()
            merged_legacy = _cat_caches(
                [request.cache for request in requests], observer=cache_pack_observer,
            )
            if cache_pack_observer is not None:
                cache_pack_observer("after_packed_legacy_cache", {"merged_legacy": merged_legacy})
                cache_pack_observer("before_restore", {"merged_legacy": merged_legacy})
            merged_cache = _restore_cache_kind(merged_legacy, requests[0].cache)
            cache_pack_seconds = time.perf_counter() - pack_started
            if cache_pack_observer is not None:
                cache_pack_observer("after_restore", {"merged_legacy": merged_legacy, "merged_cache": merged_cache})
                cache_pack_observer("before_model_forward", {"merged_legacy": merged_legacy, "merged_cache": merged_cache})
            model_started = time.perf_counter()
            outputs = model(**ready_incremental_forward_kwargs(
                token_ids=[request.token_id for request in requests], position=first.request.position,
                cache=merged_cache, device=model.device,
            ))
            model_call_seconds = time.perf_counter() - model_started
            if cache_pack_observer is not None:
                cache_pack_observer("after_b2_model_forward", {
                    "merged_legacy": merged_legacy, "merged_cache": merged_cache, "outputs": outputs,
                })
            adopt_started = time.perf_counter()
            adopted_cache_ids: list[int] = []
            if streaming_split_and_adopt:
                if cache_pack_observer is not None:
                    cache_pack_observer("before_split_creation", {
                        "merged_legacy": merged_legacy, "merged_cache": merged_cache, "outputs": outputs,
                    })
                output_cache = outputs.past_key_values
                _streaming_split_and_adopt(
                    output_cache, requests, observer=cache_pack_observer,
                )
                adopted_cache_ids = [id(request.cache_owner.cache) for request in requests]
                # Keep only per-lane logits in replies.  The output cache has
                # been copied into the original owners and can be released.
                outputs_by_cell = [
                    type("Reply", (), {
                        "logits": outputs.logits[lane:lane + 1],
                        "past_key_values": request.cache_owner.cache,
                    })()
                    for lane, request in enumerate(requests)
                ]
                if cache_pack_observer is not None:
                    cache_pack_observer("after_split_legacy", {
                        "merged_legacy": merged_legacy, "merged_cache": merged_cache,
                        "outputs": outputs, "streaming": True,
                    })
                    cache_pack_observer("after_split_adoption", {
                        "merged_legacy": merged_legacy, "merged_cache": merged_cache,
                        "outputs": outputs, "adopted_cache_ids": adopted_cache_ids,
                        "streaming": True,
                    })
                if release_batch_temporaries_for_audit:
                    output_cache = None
                    merged_legacy = None
                    merged_cache = None
                    outputs = None
                    if cache_pack_observer is not None:
                        cache_pack_observer("after_release_temporaries", {"outputs_by_cell": outputs_by_cell})
            else:
                if cache_pack_observer is not None:
                    cache_pack_observer("before_split_creation", {
                        "merged_legacy": merged_legacy, "merged_cache": merged_cache, "outputs": outputs,
                    })
                split_legacy = _split_cache(
                    outputs.past_key_values, len(selected), observer=cache_pack_observer,
                )
                if cache_pack_observer is not None:
                    cache_pack_observer("after_split_legacy", {
                        "merged_legacy": merged_legacy, "merged_cache": merged_cache,
                        "outputs": outputs, "split_legacy": split_legacy,
                    })
                    cache_pack_observer("before_cache_adoption", {
                        "merged_legacy": merged_legacy, "merged_cache": merged_cache,
                        "outputs": outputs, "split_legacy": split_legacy,
                    })
                # Each suspended DFS frame retains its original DynamicCache
                # owner.  Adopt tensor contents only; never replace that owner.
                for lane_index, (request, lane_legacy) in enumerate(zip(requests, split_legacy, strict=True)):
                    owner_cache = request.cache_owner.cache
                    owner_id = id(owner_cache)
                    if _is_transformers_cache(owner_cache):
                        replace_cache_contents_in_place(
                            owner_cache, lane_legacy, observer=cache_pack_observer, lane_index=lane_index,
                        )
                        if id(request.cache_owner.cache) != owner_id:
                            raise RuntimeError("batched cache adoption replaced a logical owner object")
                        adopted_cache_ids.append(owner_id)
                    else:
                        # CPU compatibility only.  Production Clean-HF reaches
                        # the DynamicCache branch above.
                        request.cache_owner.cache = lane_legacy
                        adopted_cache_ids.append(id(request.cache_owner.cache))
                outputs_by_cell = []
                for lane, request in enumerate(requests):
                    outputs_by_cell.append(type("Reply", (), {
                        "logits": outputs.logits[lane:lane + 1],
                        "past_key_values": request.cache_owner.cache,
                    })())
                if cache_pack_observer is not None:
                    cache_pack_observer("after_split_adoption", {
                        "merged_legacy": merged_legacy,
                        "merged_cache": merged_cache,
                        "split_legacy": split_legacy,
                        "adopted_cache_ids": adopted_cache_ids,
                    })
                if release_batch_temporaries_for_audit:
                    merged_legacy = None
                    merged_cache = None
                    split_legacy = None
                    outputs = None
                    if cache_pack_observer is not None:
                        cache_pack_observer("after_release_temporaries", {"outputs_by_cell": outputs_by_cell})
            cache_adoption_seconds = time.perf_counter() - adopt_started
    elapsed = time.perf_counter() - started
    return outputs_by_cell, {
        "model_call_seconds": model_call_seconds,
        "cache_pack_seconds": cache_pack_seconds,
        "cache_adoption_seconds": cache_adoption_seconds,
        "scheduler_elapsed_seconds": elapsed,
    }


def run_ready_scheduler(
    *,
    model: Any,
    cells: list[ReadyCell],
    dynamic_batch2: bool = False,
    scheduling_policy: SchedulingPolicy = "serial",
    observer: Callable[[str, dict[str, Any]], None] | None = None,
    cache_pack_observer: Callable[[str, dict[str, Any]], None] | None = None,
    release_b2_temporaries_for_audit: bool = False,
    streaming_split_and_adopt: bool = False,
    collect_event_trace: bool = True,
    max_physical_batch: int | None = None,
    memory_aware_config: MemoryAwareAug16Config | None = None,
    fixed_b8_config: FixedB8Aug16Config | None = None,
    memory_stats_reader: Callable[[], dict[str, int]] | None = None,
    memory_peak_reset: Callable[[], None] | None = None,
    memory_synchronize: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Drive independent cells through one shared physical batching path.

    Only READY-cell selection differs between policies.  The request
    construction, cache merge/split, model invocation, reply, and DFS resume
    path below are intentionally common to every policy.

    ``streaming_split_and_adopt`` is intentionally an explicit executor
    choice.  It preserves every logical cell's owner object while copying a
    physical batch's continuation cache back one lane at a time.  Existing
    callers retain the historical split/adopt path unless they opt in.
    """
    import torch

    if scheduling_policy not in {"serial", "round_robin", "dynamic_ready", "memory_aware_aug16", "fixed_b8_aug16"}:
        raise ValueError(f"unknown scheduling policy: {scheduling_policy}")
    physical_limit = (2 if dynamic_batch2 else 1) if max_physical_batch is None else max_physical_batch
    if physical_limit not in {1, 2, 4, 8, 12, 16}:
        raise ValueError("max_physical_batch must be one of 1, 2, 4, 8, 12, or 16")
    if scheduling_policy == "round_robin" and physical_limit != 1:
        raise ValueError("round_robin is a B1 policy and cannot enable physical batching")
    if scheduling_policy == "memory_aware_aug16":
        if memory_aware_config is None:
            raise ValueError("memory_aware_aug16 requires an explicit frozen MemoryAwareAug16Config")
        if physical_limit != 16:
            raise ValueError("memory_aware_aug16 requires max_physical_batch=16")
        if len(cells) != memory_aware_config.logical_lane_count:
            raise ValueError("memory_aware_aug16 requires exactly the frozen sixteen logical cells")
        if memory_stats_reader is None:
            if not torch.cuda.is_available():
                raise RuntimeError("memory_aware_aug16 requires CUDA unless deterministic test hooks are supplied")

            def memory_stats_reader() -> dict[str, int]:
                return {
                    "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                    "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                    "current_allocated_bytes": int(torch.cuda.memory_allocated()),
                    "current_reserved_bytes": int(torch.cuda.memory_reserved()),
                }

            def memory_peak_reset() -> None:
                # This resets only the local B16 observation window.  It is
                # not allocator control and deliberately does not empty cache.
                torch.cuda.reset_peak_memory_stats()

            def memory_synchronize() -> None:
                torch.cuda.synchronize()
        else:
            memory_peak_reset = memory_peak_reset or (lambda: None)
            memory_synchronize = memory_synchronize or (lambda: None)
    if scheduling_policy == "fixed_b8_aug16":
        if fixed_b8_config is None:
            raise ValueError("fixed_b8_aug16 requires an explicit frozen FixedB8Aug16Config")
        if physical_limit != 8:
            raise ValueError("fixed_b8_aug16 requires max_physical_batch=8")
        if len(cells) != fixed_b8_config.logical_lane_count:
            raise ValueError("fixed_b8_aug16 requires exactly the frozen sixteen logical cells")
    events: list[dict[str, Any]] = []
    forwards = 0
    logical_advances = 0
    b1_forwards = 0
    b2_forwards = 0
    b3_forwards = 0
    b4_forwards = 0
    physical_batch_histogram: dict[int, int] = {}
    model_call_seconds_total = 0.0
    cache_pack_seconds_total = 0.0
    cache_adoption_seconds_total = 0.0
    scheduler_overhead_seconds_total = 0.0
    round_robin_cursor = 0
    b16_enabled = scheduling_policy == "memory_aware_aug16"
    memory_group_cursor = 0
    b16_disabled = False
    b16_disable_reason: str | None = None
    b16_disable_forward_index: int | None = None
    b16_disable_peak_allocated_bytes: int | None = None
    b16_disable_peak_reserved_bytes: int | None = None
    b16_forward_count = 0
    b8_forward_count = 0
    physical_batch_sequence: list[dict[str, Any]] = []
    memory_transitions: list[dict[str, Any]] = []
    b16_peak_samples: list[dict[str, int]] = []
    previous_b16_stats: dict[str, int] | None = None
    max_cache_length_observed = 0
    while True:
        memory_mode: str | None = None
        memory_group: str | None = None
        selected_index: int | None = None
        next_cursor: int | None = None
        if scheduling_policy in {"memory_aware_aug16", "fixed_b8_aug16"}:
            selected, memory_group_cursor, memory_mode, memory_group, ready_keys = _select_memory_aware_aug16(
                cells=cells,
                b16_enabled=b16_enabled if scheduling_policy == "memory_aware_aug16" else False,
                group_cursor=memory_group_cursor,
                capture_ready_keys=collect_event_trace,
            )
            if scheduling_policy == "fixed_b8_aug16":
                memory_mode = "B8_FIXED_SPLIT_MODE"
        else:
            selected, selected_index, next_cursor, ready_keys = _select_ready_cells(
                cells=cells,
                scheduling_policy=scheduling_policy,
                round_robin_cursor=round_robin_cursor,
                capture_ready_keys=collect_event_trace,
            )
        if not selected:
            break
        first = selected[0]
        if physical_limit > 1 and scheduling_policy not in {"memory_aware_aug16", "fixed_b8_aug16"}:
            ready = sorted(
                (cell for cell in cells if cell.request is not None),
                key=lambda cell: (cell.cell_key, cell.request.ordinal if cell.request else -1),
            )
            for other in ready:
                if other is first:
                    continue
                if len(selected) >= physical_limit:
                    break
                if other.request is not None and other.request.cache_key == first.request.cache_key and other.request.position == first.request.position:
                    selected.append(other)
        requests = [cell.request for cell in selected]
        assert all(request is not None for request in requests)
        for request in requests:
            assert request is not None
            length = _request_cache_length(request.cache)
            if length is not None:
                max_cache_length_observed = max(max_cache_length_observed, length)
        if observer is not None:
            observer("before_forward", {
                "physical_forward_index": forwards + 1,
                "selected_cells": selected,
                "requests": requests,
                "physical_batch": len(selected),
                "scheduler_mode": memory_mode if memory_mode is not None else scheduling_policy,
                "split_group": memory_group,
            })
        trace_inputs = []
        if first.config.diagnostic_trace:
            for cell, request in zip(selected, requests, strict=True):
                assert request is not None
                parent = next(
                    (node for node in reversed(cell.state["nodes"])
                     if node.get("node_id") == request.parent_node_id),
                    None,
                )
                trace_inputs.append({
                    "cell_key": cell.cell_key, "logical_step": int(request.ordinal),
                    "request_ordinal": request.ordinal,
                    "token_id": request.token_id, "absolute_position": request.position,
                    "cache_type": type(request.cache).__module__ + "." + type(request.cache).__qualname__,
                    "input_cache_sha256": cache_sha256(request.cache),
                    "cache_geometry": repr(request.cache_key),
                    "request_parent_node_id": request.parent_node_id,
                    "cache_owner_id": id(request.cache_owner),
                    "valid_kv_length": _request_cache_length(request.cache),
                    "physical_batch_width": len(selected),
                    "physical_batch_member_ids": [member.cell_key for member in selected],
                    "parent_prefix_hash": None if parent is None else parent.get("prefix_hash"),
                    "selected_token": None if parent is None else parent.get("selected_token"),
                    "selected_token_logprob": None if parent is None else parent.get("token_logprob"),
                    "cumulative_regret": None if parent is None else parent.get("cumulative_regret"),
                })
        is_observed_b16 = scheduling_policy == "memory_aware_aug16" and b16_enabled and len(selected) == 16
        if is_observed_b16:
            assert memory_peak_reset is not None
            memory_peak_reset()
        # Telemetry remains host-wall only except the explicit memory-aware
        # local B16 peak window.  That window synchronizes only after the
        # real pack/forward/streaming-adoption sequence has completed.
        outputs_by_cell, forward_telemetry = execute_ready_forward(
            model=model,
            selected=selected,
            requests=requests,
            cache_pack_observer=cache_pack_observer,
            release_batch_temporaries_for_audit=release_b2_temporaries_for_audit,
            streaming_split_and_adopt=streaming_split_and_adopt,
        )
        b16_stats: dict[str, int] | None = None
        if is_observed_b16:
            assert memory_synchronize is not None and memory_stats_reader is not None
            memory_synchronize()
            b16_stats = {key: int(value) for key, value in memory_stats_reader().items()}
            required_stats = {
                "peak_allocated_bytes", "peak_reserved_bytes",
                "current_allocated_bytes", "current_reserved_bytes",
            }
            missing_stats = required_stats.difference(b16_stats)
            if missing_stats:
                raise RuntimeError(f"memory-aware peak reader omitted fields: {sorted(missing_stats)}")
        elapsed = forward_telemetry["scheduler_elapsed_seconds"]
        model_call_seconds = forward_telemetry["model_call_seconds"]
        cache_pack_seconds = forward_telemetry["cache_pack_seconds"]
        cache_adoption_seconds = forward_telemetry["cache_adoption_seconds"]
        scheduler_overhead_seconds = max(0.0, elapsed - model_call_seconds - cache_pack_seconds - cache_adoption_seconds)
        if observer is not None:
            observer("after_model_forward", {
                "physical_forward_index": forwards + 1,
                "selected_cells": selected,
                "requests": requests,
                "outputs_by_cell": outputs_by_cell,
                "physical_batch": len(selected),
                "scheduler_mode": memory_mode if memory_mode is not None else scheduling_policy,
                "split_group": memory_group,
                "model_elapsed_seconds": elapsed,
            })
        forwards += 1
        logical_advances += len(selected)
        b1_forwards += int(len(selected) == 1)
        b2_forwards += int(len(selected) == 2)
        b3_forwards += int(len(selected) == 3)
        b4_forwards += int(len(selected) == 4)
        physical_batch_histogram[len(selected)] = physical_batch_histogram.get(len(selected), 0) + 1
        b16_forward_count += int(scheduling_policy == "memory_aware_aug16" and len(selected) == 16)
        b8_forward_count += int(memory_mode in {"B8_SPLIT_MODE", "B8_FIXED_SPLIT_MODE"} and len(selected) == 8)
        model_call_seconds_total += model_call_seconds
        cache_pack_seconds_total += cache_pack_seconds
        cache_adoption_seconds_total += cache_adoption_seconds
        scheduler_overhead_seconds_total += scheduler_overhead_seconds
        for cell in selected:
            # One physical B2 forward is fully attributable to every logical
            # cell it advances; dividing it would extend the scientific budget.
            cell.state["model_forward_seconds"] += elapsed
            cell.active_elapsed_seconds += elapsed
            cell.state["active_elapsed_seconds"] += elapsed
        cursor_before = round_robin_cursor if scheduling_policy == "round_robin" else None
        if scheduling_policy == "round_robin":
            assert selected_index is not None and next_cursor is not None
            round_robin_cursor = next_cursor
        physical_batch_sequence.append({
            "forward_index": forwards,
            "physical_batch": len(selected),
            "scheduler_mode": memory_mode if memory_mode is not None else scheduling_policy,
            "split_group": memory_group,
            "cell_keys": [cell.cell_key for cell in selected],
        })
        if collect_event_trace:
            events.append({
                "forward_index": forwards,
                "physical_batch": len(selected),
                "position": first.request.position,
                "host_model_call_seconds": model_call_seconds,
                "host_cache_pack_seconds": cache_pack_seconds,
                "host_cache_adoption_seconds": cache_adoption_seconds,
                "host_scheduler_overhead_seconds": scheduler_overhead_seconds,
                "cache_geometry": repr(first.request.cache_key),
                "cell_keys": [cell.cell_key for cell in selected],
                "request_ordinals": [request.ordinal for request in requests],
                "scheduling_policy": scheduling_policy,
                "ready_cell_keys": ready_keys,
                "round_robin_cursor_before": cursor_before,
                "round_robin_cursor_after": round_robin_cursor if scheduling_policy == "round_robin" else None,
                "scheduler_mode": memory_mode if memory_mode is not None else scheduling_policy,
                "memory_split_group": memory_group,
                "b16_peak_memory": b16_stats,
                "wall_seconds": elapsed,
            })
        for trace, cell, reply in zip(trace_inputs or [None] * len(selected), selected, outputs_by_cell, strict=True):
            if trace is not None:
                # Capture the physical reply *before* resuming the DFS
                # coroutine.  In rollback mode that resume can descend and
                # crop the one mutable owner several times before yielding its
                # next request; hashing afterwards misattributes a later
                # parent state to this forward.
                trace.update(_logits_diagnostic(reply.logits, cell.config.arc_tokens))
                trace["output_cache_sha256"] = cache_sha256(reply.past_key_values)
                trace["output_valid_kv_length"] = _request_cache_length(reply.past_key_values)
            _reply(cell, reply)
            current_length = _request_cache_length(cell.cache_owner.cache) if cell.cache_owner is not None else None
            if current_length is not None:
                max_cache_length_observed = max(max_cache_length_observed, current_length)
            if trace is not None:
                trace["retained_successors"] = cell.state["retained_successors_by_parent"].get(
                    trace["request_parent_node_id"],
                )
                decision = next(
                    (row for row in reversed(cell.state["branch_probabilities"])
                     if row.get("parent_node_id") == trace["request_parent_node_id"]),
                    None,
                )
                if decision is not None:
                    trace.update({
                        "top_token": decision.get("top1_token_id"),
                        "top_k_ordering": [
                            int(row["token_id"])
                            for row in sorted(
                                decision.get("full_arc_logprobs", []),
                                key=lambda row: (-float(row["logprob"]), int(row["token_id"])),
                            )
                        ],
                        "incremental_regret": None if trace.get("selected_token_logprob") is None else (
                            float(decision["top1_logprob"]) - float(trace["selected_token_logprob"])
                        ),
                    })
                trace["frontier_size"] = (
                    cell.state["frontier_samples"][-1]["total"]
                    if cell.state["frontier_samples"] else 0
                )
                trace["completed_candidate_count"] = int(cell.state["completed_candidates"])
                trace["termination_state"] = (
                    None if cell.result is None else cell.result.termination_reason
                )
                trace["cache_owner_id_after_reply"] = (
                    None if cell.cache_owner is None else id(cell.cache_owner)
                )
                cell.state["per_forward_trace"].append(trace)
        if len(selected) > 1 and cache_pack_observer is not None:
            cache_pack_observer("after_logical_resume", {"outputs_by_cell": outputs_by_cell})
        if b16_stats is not None:
            b16_peak_samples.append({"forward_index": forwards, **b16_stats})
            if b16_stats["peak_allocated_bytes"] >= memory_aware_config.soft_peak_allocated_bytes:
                b16_enabled = False
                b16_disabled = True
                b16_disable_reason = (
                    "HARD_PEAK_ALLOCATED_THRESHOLD" if b16_stats["peak_allocated_bytes"] >= memory_aware_config.hard_peak_allocated_bytes
                    else "SOFT_PEAK_ALLOCATED_THRESHOLD"
                )
                b16_disable_forward_index = forwards
                b16_disable_peak_allocated_bytes = b16_stats["peak_allocated_bytes"]
                b16_disable_peak_reserved_bytes = b16_stats["peak_reserved_bytes"]
                lengths = {
                    cell.cell_key: _request_cache_length(cell.cache_owner.cache) if cell.cache_owner is not None else None
                    for cell in cells
                }
                present_lengths = [value for value in lengths.values() if value is not None]
                progress = {
                    cell.cell_key: {
                        "nodes_expanded": sum(1 for node in cell.state["nodes"] if node.get("state") == "expanded"),
                        "request_count": int(cell.request_count),
                        "request_ordinal": cell.request.ordinal if cell.request is not None else None,
                    }
                    for cell in cells
                }
                memory_transitions.append({
                    "event": "B16_TO_B8",
                    "reason": b16_disable_reason,
                    "forward_index": forwards,
                    "peak_allocated_bytes": b16_stats["peak_allocated_bytes"],
                    "peak_reserved_bytes": b16_stats["peak_reserved_bytes"],
                    "current_allocated_bytes": b16_stats["current_allocated_bytes"],
                    "current_reserved_bytes": b16_stats["current_reserved_bytes"],
                    "previous_b16_peak_allocated_bytes": previous_b16_stats["peak_allocated_bytes"] if previous_b16_stats else None,
                    "previous_b16_peak_reserved_bytes": previous_b16_stats["peak_reserved_bytes"] if previous_b16_stats else None,
                    "owner_cache_lengths": lengths,
                    "owner_cache_length_min": min(present_lengths) if present_lengths else None,
                    "owner_cache_length_max": max(present_lengths) if present_lengths else None,
                    "owner_cache_length_mean": sum(present_lengths) / len(present_lengths) if present_lengths else None,
                    "per_cell_progress": progress,
                })
            previous_b16_stats = b16_stats
    telemetry = {
        "physical_forwards": forwards,
        "b1_forwards": b1_forwards,
        "b2_forwards": b2_forwards,
        "b3_forwards": b3_forwards,
        "b4_forwards": b4_forwards,
        "physical_batch_histogram": {str(width): count for width, count in sorted(physical_batch_histogram.items())},
        "logical_advances": logical_advances,
        "mean_effective_batch": logical_advances / forwards if forwards else 0.0,
        "active2_fraction": (2 * b2_forwards) / logical_advances if logical_advances else 0.0,
        "active4_logical_fraction": (4 * b4_forwards) / logical_advances if logical_advances else 0.0,
        "b4_forward_fraction": b4_forwards / forwards if forwards else 0.0,
        "model_call_seconds": model_call_seconds_total,
        "cache_pack_seconds": cache_pack_seconds_total,
        "cache_adoption_seconds": cache_adoption_seconds_total,
        "scheduler_overhead_seconds": scheduler_overhead_seconds_total,
    }
    if scheduling_policy == "memory_aware_aug16":
        telemetry.update({
            "scheduler_mode": "MEMORY_AWARE_AUG16",
            "b16_enabled_initially": True,
            "b16_disabled": b16_disabled,
            "b16_disable_reason": b16_disable_reason,
            "b16_disable_forward_index": b16_disable_forward_index,
            "b16_disable_peak_allocated_bytes": b16_disable_peak_allocated_bytes,
            "b16_disable_peak_reserved_bytes": b16_disable_peak_reserved_bytes,
            "soft_peak_allocated_bytes": memory_aware_config.soft_peak_allocated_bytes,
            "hard_peak_allocated_bytes": memory_aware_config.hard_peak_allocated_bytes,
            "b16_forward_count": b16_forward_count,
            "b8_forward_count": b8_forward_count,
            "other_batch_forward_counts": {
                str(width): count for width, count in sorted(physical_batch_histogram.items()) if width not in {8, 16}
            },
            "physical_batch_sequence": physical_batch_sequence,
            "memory_transitions": memory_transitions,
            "b16_peak_samples": b16_peak_samples,
            "max_cache_length_observed": max_cache_length_observed,
        })
    if scheduling_policy == "fixed_b8_aug16":
        telemetry.update({
            "scheduler_mode": "FIXED_B8_AUG16",
            "b16_enabled_initially": False,
            "b8_forward_count": b8_forward_count,
            "other_batch_forward_counts": {
                str(width): count for width, count in sorted(physical_batch_histogram.items()) if width != 8
            },
            "physical_batch_sequence": physical_batch_sequence,
            "max_cache_length_observed": max_cache_length_observed,
        })
    return {"scheduling_policy": scheduling_policy, "physical_forwards": forwards, "events": events,
            "mean_effective_batch": telemetry["mean_effective_batch"], "telemetry": telemetry}


def ready_result(cell: ReadyCell) -> D1TurboDFSResult:
    if cell.result is None:
        raise RuntimeError("dynamic-ready cell has not completed")
    state = cell.state
    elapsed_budget = cell.active_elapsed_seconds if state["active_time_accounting"] else time.time() - cell.started_unix
    timed_out = elapsed_budget >= cell.config.local_time_limit_seconds or (
        cell.config.absolute_end_time_unix is not None and time.time() >= cell.config.absolute_end_time_unix)
    reason = "budget_exhausted" if state["budget_exhausted"] else "wall_time" if timed_out else "search_exhausted"
    return D1TurboDFSResult((tuple(sorted(cell.result, key=lambda item: item.cumulative_nll)),), tuple(state["nodes"]),
                             tuple(state["branch_probabilities"]), tuple(state["frontier_floor_events"]),
                             tuple(state["frontier_samples"]), tuple(state["search_trace"]),
                             float(state["model_forward_seconds"]), int(state["model_forwards"]),
                             int(state["tokens_advanced"]), int(state["completed_candidates"]),
                             int(state["max_frontier_size"]), bool(state["budget_exhausted"]), timed_out, reason,
                             (), (), (), (), (), (), (), dict(state["performance_telemetry"]))


def normalized_result_signature(result: D1TurboDFSResult) -> str:
    """Hash only deterministic decoder evidence; discovery wall times are excluded."""
    def clean(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items() if key not in {"elapsed_seconds"}}
        if isinstance(value, tuple):
            return [clean(item) for item in value]
        if isinstance(value, list):
            return [clean(item) for item in value]
        return value
    payload = {
        "candidates": [[{"id": item.candidate_id, "tokens": list(item.token_ids), "nll": item.cumulative_nll,
                          "node": item.terminal_node_id, "forward": item.discovery_forward_index}
                         for item in lane] for lane in result.candidates],
        "nodes": clean(result.nodes), "branches": clean(result.branch_probabilities),
        "floors": clean(result.frontier_floor_events), "trace": clean(result.search_trace),
        "nodes_expanded": sum(1 for item in result.nodes if item.get("state") == "expanded"),
        "completed": result.completed_candidates, "termination": result.termination_reason,
    }
    # Trace dictionaries may acquire the same keys in a different insertion
    # order when a coroutine resumes.  Canonical JSON tests evidence equality,
    # not incidental Python dict construction order.
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


# Wall-clock observations are useful performance telemetry, but are never
# decoder semantics.  Keep this helper public so every diagnostic and gate
# applies the same representation rule rather than comparing Python tuples
# against JSON-decoded lists by accident.
NON_SEMANTIC_PARITY_FIELDS = frozenset({
    "elapsed_seconds",
    "wall_seconds",
    "wall_time_seconds",
    "timestamp",
    "timestamp_utc",
    "discovery_unix",
    "discovery_wall_seconds",
})


def canonical_semantic_value(value: Any) -> Any:
    """Canonical, JSON-safe representation of deterministic decoder evidence.

    This intentionally preserves probabilities, ranks, node IDs, Regret
    values, frontier state, candidates, and termination information.  It only
    removes explicitly non-semantic wall-clock observations and normalizes
    tuple/list representation introduced by JSON serialization.
    """
    if isinstance(value, dict):
        return {
            str(key): canonical_semantic_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if str(key) not in NON_SEMANTIC_PARITY_FIELDS
        }
    if isinstance(value, (tuple, list)):
        return [canonical_semantic_value(item) for item in value]
    return value


def semantic_value_sha256(value: Any) -> str:
    """Hash canonical decoder evidence for artifact-to-artifact parity."""
    payload = canonical_semantic_value(value)
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
