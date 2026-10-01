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
from typing import Any, Callable

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


@dataclass(frozen=True)
class ReadyForwardRequest:
    """One real incremental forward request from one independent DFS cell."""

    cell_key: str
    ordinal: int
    token_id: int
    position: int
    cache: Any
    cache_key: tuple[Any, ...]


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


def cache_geometry(cache: Any) -> tuple[Any, ...]:
    """Stable compatibility key excluding the real batch dimension."""
    legacy = _legacy_cache(cache)
    geometry: list[Any] = [len(legacy)]
    for layer in legacy:
        geometry.append(tuple((str(value.dtype), str(value.device), tuple(int(x) for x in value.shape[1:])) for value in layer))
    return tuple(geometry)


def _cat_caches(caches: list[Any]) -> tuple[tuple[Any, ...], ...]:
    import torch

    if not caches:
        raise ValueError("cannot concatenate no caches")
    legacy = [_legacy_cache(cache) for cache in caches]
    reference = cache_geometry(legacy[0])
    if any(cache_geometry(item) != reference for item in legacy[1:]):
        raise RuntimeError("attempted dynamic batch with incompatible real KV caches")
    return tuple(tuple(torch.cat([item[layer_i][part_i] for item in legacy], dim=0)
                             for part_i in range(len(legacy[0][layer_i])))
                 for layer_i in range(len(legacy[0])))


def _split_cache(cache: Any, expected: int) -> list[tuple[tuple[Any, ...], ...]]:
    legacy = _legacy_cache(cache)
    size = _cache_batch_size(legacy)
    if size != expected:
        raise RuntimeError(f"dynamic-ready output cache batch mismatch {size} != {expected}")
    return [tuple(tuple(value[index:index + 1].contiguous() for value in layer) for layer in legacy)
            for index in range(expected)]


def _ready_dfs(
    *, cell_key: str, logits: Any, max_new_tokens: int, score: float, regret: float,
    pos: int, cache: Any, config: D1TurboDFSConfig, started_unix: float,
    state: dict[str, Any], parent_node: int | None, prefix: tuple[int, ...],
    ordinal: list[int],
) -> Generator[ReadyForwardRequest, Any, list[ReferenceTurboDFSCandidate]]:
    """Single-lane coroutine equivalent to the D1 recursive search function."""
    import torch

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
        request = ReadyForwardRequest(cell_key, ordinal[0], token, pos, cache, cache_geometry(cache))
        ordinal[0] += 1
        outputs = yield request
        if config.calibration_assertions and int(outputs.logits.shape[0]) != 1:
            raise RuntimeError("dynamic-ready B1 reply must have one logical lane")
        state["model_forwards"] += 1; state["tokens_advanced"] += 1
        descendants = yield from _ready_dfs(
            cell_key=cell_key, logits=outputs.logits[:, -1], max_new_tokens=max_new_tokens - 1,
            score=next_score, regret=next_regret, pos=pos + 1, cache=outputs.past_key_values,
            config=config, started_unix=started_unix, state=state, parent_node=node_id,
            prefix=prefix + (token,), ordinal=ordinal,
        )
        suffixes.extend(ReferenceTurboDFSCandidate(item.candidate_id, (token,) + item.token_ids,
                                                    item.cumulative_nll, item.terminal_node_id,
                                                    item.discovery_forward_index, item.discovery_unix)
                        for item in descendants)
    return suffixes


def start_ready_cell(*, model: Any, input_ids: Any, config: D1TurboDFSConfig, cell_key: str,
                     normalize_root_cache: bool, active_time_accounting: bool = False,
                     root_cache_transform: Callable[[Any], Any] | None = None) -> ReadyCell:
    """Run the required scalar prefill, then expose the first incremental request."""
    import torch

    if input_ids.ndim != 2 or tuple(input_ids.shape[:1]) != (1,):
        raise ValueError("dynamic-ready cells require one prompt at a time")
    state = _new_state(config)
    started_unix = time.time()
    created_perf = time.perf_counter()
    state["active_time_accounting"] = active_time_accounting
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
    generator = _ready_dfs(cell_key=cell_key, logits=outputs.logits[:, -1], max_new_tokens=config.max_new_tokens,
                           score=0.0, regret=0.0, pos=int(input_ids.size(1)), cache=root_cache, config=config,
                           started_unix=started_unix, state=state, parent_node=root, prefix=tuple(), ordinal=[0])
    cell = ReadyCell(cell_key, config, state, started_unix, created_perf, generator,
                     prefill_seconds=prefill_seconds, active_elapsed_seconds=prefill_seconds)
    _advance_cell(cell)
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


def run_ready_scheduler(*, model: Any, cells: list[ReadyCell], dynamic_batch2: bool) -> dict[str, Any]:
    """Drive independent cells in deterministic ready order with real B1/B2 calls."""
    import torch

    events: list[dict[str, Any]] = []
    forwards = 0
    while True:
        ready = sorted((cell for cell in cells if cell.request is not None),
                       key=lambda cell: (cell.cell_key, cell.request.ordinal if cell.request else -1))
        if not ready:
            break
        first = ready[0]
        selected = [first]
        if dynamic_batch2:
            for other in ready[1:]:
                if other.request is not None and other.request.cache_key == first.request.cache_key and other.request.position == first.request.position:
                    selected.append(other)
                    break
        requests = [cell.request for cell in selected]
        assert all(request is not None for request in requests)
        started = time.perf_counter()
        # The authoritative decoder executes every model forward under
        # ``torch.no_grad``.  The scheduler must keep that invariant even
        # though it owns the incremental calls rather than the recursive body.
        with torch.no_grad():
            if len(selected) == 1:
                request = requests[0]
                outputs = model(input_ids=torch.tensor([[request.token_id]], device=model.device, dtype=torch.long),
                                position_ids=torch.tensor([[request.position]], device=model.device, dtype=torch.long),
                                past_key_values=request.cache, return_dict=True, use_cache=True)
                outputs_by_cell = [outputs]
            else:
                merged_cache = _cat_caches([request.cache for request in requests])
                outputs = model(input_ids=torch.tensor([[request.token_id] for request in requests], device=model.device, dtype=torch.long),
                                position_ids=torch.tensor([[request.position] for request in requests], device=model.device, dtype=torch.long),
                                past_key_values=merged_cache, return_dict=True, use_cache=True)
                split_cache = _split_cache(outputs.past_key_values, len(selected))
                outputs_by_cell = []
                for lane, cache in enumerate(split_cache):
                    outputs_by_cell.append(type("Reply", (), {"logits": outputs.logits[lane:lane + 1], "past_key_values": cache})())
        elapsed = time.perf_counter() - started
        forwards += 1
        for cell in selected:
            # One physical B2 forward is fully attributable to every logical
            # cell it advances; dividing it would extend the scientific budget.
            cell.state["model_forward_seconds"] += elapsed
            cell.active_elapsed_seconds += elapsed
            cell.state["active_elapsed_seconds"] += elapsed
        events.append({"forward_index": forwards, "physical_batch": len(selected), "position": first.request.position,
                       "cache_geometry": repr(first.request.cache_key), "cell_keys": [cell.cell_key for cell in selected],
                       "wall_seconds": elapsed})
        for cell, reply in zip(selected, outputs_by_cell, strict=True):
            _reply(cell, reply)
        del outputs
    return {"physical_forwards": forwards, "events": events,
            "mean_effective_batch": (sum(row["physical_batch"] for row in events) / len(events)) if events else 0.0}


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
