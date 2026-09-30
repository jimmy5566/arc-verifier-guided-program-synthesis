"""Bounded, target-blind D1 decoder variants for the real A/B pilot.

The module deliberately keeps V5's native-token legality, recursive KV-cache
route, ordering, max length, wall clock, floor semantics, and candidate format.
It adds a *common D1 external safety budget* to all policies, then changes only
the successor-retention predicate.  The safety budget is predeclared from
frozen V5 telemetry and is checked explicitly by the runner.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import time
from typing import Any

from inference.nvarc_turbodfs_reference import (
    PUBLIC_ARC_TOKENS,
    PUBLIC_LOCAL_TIME_LIMIT_SECONDS,
    PUBLIC_PAD_ID,
    ReferenceTurboDFSCandidate,
    _cache_batch_size,
)


POLICIES = (
    "V5_CURRENT",
    "CUMULATIVE_REGRET_r=4.00",
    "AFFINE_NLL_BUDGET_tau0=2.000_lambda=0.0400",
    "TOPK_LOCAL_k=2",
)


@dataclass(frozen=True)
class D1TurboDFSConfig:
    policy_id: str
    max_new_tokens: int
    max_score: float
    absolute_end_time_unix: float | None
    max_expanded_nodes: int
    max_completed_candidates: int
    frontier_floor: int = 1
    local_time_limit_seconds: float = PUBLIC_LOCAL_TIME_LIMIT_SECONDS
    pad_token_id: int = PUBLIC_PAD_ID
    arc_tokens: tuple[int, ...] = PUBLIC_ARC_TOKENS
    calibration_assertions: bool = True
    diagnostic_trace: bool = False
    # Disabled on normal decoder paths. The performance-only microprofile may
    # enable it to time executor sections without changing search state.
    performance_profile: bool = False
    # This is deliberately opt-in.  The frozen historical D1 route retains
    # its shared counter semantics; the Regret fast engine uses independent
    # logical-cell budgets while sharing only model forwards/KV lanes.
    independent_lane_budgets: bool = False

    def __post_init__(self) -> None:
        if self.policy_id not in POLICIES:
            raise ValueError(f"unsupported D1 policy: {self.policy_id}")
        if self.max_new_tokens < 1 or self.max_expanded_nodes < 1 or self.max_completed_candidates < 1:
            raise ValueError("D1 bounds must be positive")
        if not math.isfinite(self.max_score) or self.max_score <= 0:
            raise ValueError("max_score must be finite and positive")
        if self.frontier_floor != 1:
            raise ValueError("D1 is fixed to the frozen V5 frontier floor of one")
        if self.local_time_limit_seconds != PUBLIC_LOCAL_TIME_LIMIT_SECONDS:
            raise ValueError("D1 must preserve V5's public 540-second local time limit")
        if self.pad_token_id != PUBLIC_PAD_ID or tuple(self.arc_tokens) != PUBLIC_ARC_TOKENS:
            raise ValueError("D1 native-token contract differs from V5")


@dataclass(frozen=True)
class D1TurboDFSResult:
    candidates: tuple[tuple[ReferenceTurboDFSCandidate, ...], ...]
    nodes: tuple[dict[str, Any], ...]
    branch_probabilities: tuple[dict[str, Any], ...]
    frontier_floor_events: tuple[dict[str, Any], ...]
    frontier_samples: tuple[dict[str, Any], ...]
    search_trace: tuple[dict[str, Any], ...]
    model_forward_seconds: float
    model_forwards: int
    tokens_advanced: int
    completed_candidates: int
    max_frontier_size: int
    budget_exhausted: bool
    timed_out: bool
    termination_reason: str
    # Per-lane fields are populated for the cross-cell executor.  They remain
    # empty on the historical path so existing callers cannot accidentally
    # reinterpret the frozen aggregate telemetry.
    model_forwards_by_lane: tuple[int, ...] = ()
    tokens_advanced_by_lane: tuple[int, ...] = ()
    completed_candidates_by_lane: tuple[int, ...] = ()
    expanded_nodes_by_lane: tuple[int, ...] = ()
    budget_exhausted_by_lane: tuple[bool, ...] = ()
    termination_reason_by_lane: tuple[str, ...] = ()
    max_frontier_size_by_lane: tuple[int, ...] = ()
    performance_telemetry: dict[str, float] | None = None


def _node(runtime_state: dict[str, Any], **payload: Any) -> int:
    started = time.perf_counter() if runtime_state.get("performance_profile", False) else None
    payload.setdefault("frontier_floor_activated", False)
    payload.setdefault("frontier_floor_restore_rank", None)
    payload.setdefault("cumulative_regret", None)
    lane = int(payload.get("lane", 0))
    if runtime_state.get("independent_lane_budgets", False):
        value = int(runtime_state["next_node_id_by_lane"][lane])
        runtime_state["next_node_id_by_lane"][lane] += 1
    else:
        value = int(runtime_state["next_node_id"])
        runtime_state["next_node_id"] += 1
    runtime_state["nodes"].append({"node_id": value, **payload})
    if started is not None:
        runtime_state["performance_telemetry"]["node_and_state_seconds"] += time.perf_counter() - started
    return value


def _prefix_hash(prefix: tuple[int, ...]) -> str:
    """Stable diagnostic identifier; it never participates in search order."""
    return hashlib.sha256(",".join(str(value) for value in prefix).encode("ascii")).hexdigest()


def _trace(state: dict[str, Any], **payload: Any) -> dict[str, Any] | None:
    """Append-only diagnostic event recorder, disabled on the frozen D1 path."""
    if not state.get("diagnostic_trace", False):
        return None
    started = time.perf_counter() if state.get("performance_profile", False) else None
    lane = int(payload.get("lane", 0))
    if state.get("independent_lane_budgets", False):
        payload.setdefault("trace_event_id", int(state["next_trace_event_id_by_lane"][lane]))
        state["next_trace_event_id_by_lane"][lane] += 1
    else:
        payload.setdefault("trace_event_id", int(state["next_trace_event_id"]))
        state["next_trace_event_id"] += 1
    payload.setdefault("elapsed_seconds", time.perf_counter() - float(state["trace_started_perf"]))
    state["search_trace"].append(payload)
    if started is not None:
        state["performance_telemetry"]["trace_telemetry_seconds"] += time.perf_counter() - started
    return payload


def _record_frontier(state: dict[str, Any], candidates: list[list[Any]]) -> None:
    """Capture actual pending successor counts without changing search order."""
    started = time.perf_counter() if state.get("performance_profile", False) else None
    lanes = [len(values) for values in candidates]
    total = sum(lanes)
    state["max_frontier_size"] = max(state["max_frontier_size"], total)
    if state.get("independent_lane_budgets", False):
        for lane, value in enumerate(lanes):
            state["max_frontier_size_by_lane"][lane] = max(state["max_frontier_size_by_lane"][lane], value)
    state["frontier_samples"].append({"total": total, "lanes": lanes})
    if started is not None:
        state["performance_telemetry"]["frontier_bookkeeping_seconds"] += time.perf_counter() - started


def _lane_counter(state: dict[str, Any], name: str, lane: int) -> int:
    """Read a logical per-lane counter on the independent executor path."""
    if state.get("independent_lane_budgets", False):
        return int(state[f"{name}_by_lane"][lane])
    return int(state[name])


def _successors(config: D1TurboDFSConfig, ranked: list[tuple[int, float]], *, score_before: float,
                regret_before: float, remaining: int) -> tuple[list[tuple[float, float, int, float]], str]:
    """Return locally retained legal successors, sorted by frozen V5 score order."""
    legal = [(token, logprob) for token, logprob in ranked if token == 15 or remaining > 1]
    values = [(score_before - logprob, regret_before + ranked[0][1] - logprob, token, logprob) for token, logprob in legal]
    if config.policy_id == "V5_CURRENT":
        kept = [value for value in values if value[0] < config.max_score]
        reason = "public_cumulative_nll"
    elif config.policy_id == "CUMULATIVE_REGRET_r=4.00":
        kept = [value for value in values if value[1] <= 4.0]
        reason = "cumulative_regret"
    elif config.policy_id == "AFFINE_NLL_BUDGET_tau0=2.000_lambda=0.0400":
        length = remaining  # replaced below with position-aware offset in caller
        raise RuntimeError("affine policy requires generated_length argument")
    elif config.policy_id == "TOPK_LOCAL_k=2":
        legal_ranked = [(token, logprob) for token, logprob in ranked if token == 15 or remaining > 1][:2]
        legal_tokens = {token for token, _logprob in legal_ranked}
        kept = [value for value in values if value[2] in legal_tokens]
        reason = "topk_local"
    else:  # pragma: no cover - guarded by config validation
        raise AssertionError(config.policy_id)
    return sorted(kept, key=lambda value: (value[0], value[2])), reason


def _retained(config: D1TurboDFSConfig, ranked: list[tuple[int, float]], *, score_before: float,
              regret_before: float, remaining: int, generated_length: int) -> tuple[list[tuple[float, float, int, float]], str]:
    if config.policy_id != "AFFINE_NLL_BUDGET_tau0=2.000_lambda=0.0400":
        return _successors(config, ranked, score_before=score_before, regret_before=regret_before, remaining=remaining)
    legal = [(token, logprob) for token, logprob in ranked if token == 15 or remaining > 1]
    budget = 2.0 + 0.04 * (generated_length + 1)
    values = [(score_before - logprob, regret_before + ranked[0][1] - logprob, token, logprob) for token, logprob in legal]
    return sorted([value for value in values if value[0] <= budget], key=lambda value: (value[0], value[2])), "affine_nll_budget"


def d1_turbo_dfs(
    model: Any, *, logits: Any, max_new_tokens: int, scores: list[float], regrets: list[float], pos: int,
    cache: Any, config: D1TurboDFSConfig, started_unix: float, state: dict[str, Any],
    parent_nodes: list[int | None], prefixes: list[tuple[int, ...]],
    active_mask: list[bool] | None = None,
) -> list[list[ReferenceTurboDFSCandidate]]:
    import torch

    lanes = int(logits.size(0))
    if active_mask is None:
        active_mask = [True] * lanes
    if config.calibration_assertions:
        if not (len(scores) == len(regrets) == len(parent_nodes) == len(prefixes) == len(active_mask) == lanes):
            raise RuntimeError("D1 lane metadata mismatch")
        cached = _cache_batch_size(cache)
        if cached is not None and cached != lanes:
            raise RuntimeError(f"D1 KV cache batch mismatch cache={cached} lanes={lanes}")
    log_probs = logits.float().cpu().log_softmax(-1)
    suffixes: list[list[ReferenceTurboDFSCandidate]] = [[] for _ in range(lanes)]
    candidates: list[list[tuple[float, float, int, bool, int | None]]] = [[] for _ in range(lanes)]
    _record_frontier(state, candidates)

    for lane in range(lanes):
        # A physical pad/cache slot is never a logical decoder lane.  In
        # particular it must not reach retention, frontier-floor, nodes,
        # candidates, counters, or traces.
        if not active_mask[lane]:
            continue
        values = [(token, float(log_probs[lane, token].item())) for token in config.arc_tokens]
        ranked = sorted(values, key=lambda pair: (-pair[1], pair[0]))
        probabilities = [math.exp(logprob) for _token, logprob in values]
        state["branch_probabilities"].append({
            "forward_index": (state["model_forwards_by_lane"][lane] if state.get("independent_lane_budgets", False)
                              else state["model_forwards"]), "lane": lane, "parent_node_id": parent_nodes[lane],
            "token_position": pos, "prefix_length": len(prefixes[lane]), "cumulative_score_before": float(scores[lane]),
            "cumulative_regret_before": float(regrets[lane]), "full_arc_logprobs": [{"token_id": token, "logprob": logprob} for token, logprob in values],
            "top1_token_id": ranked[0][0], "top1_logprob": ranked[0][1], "top2_token_id": ranked[1][0], "top2_logprob": ranked[1][1],
            "margin": ranked[0][1] - ranked[1][1], "entropy": -sum(p * lp for p, (_token, lp) in zip(probabilities, values, strict=True)),
            "batch_size": lanes, "active_lane_count": sum(active_mask),
            "active_mask": tuple(bool(value) for value in active_mask), "decoder_policy": config.policy_id,
        })
        kept, prune_reason = _retained(config, ranked, score_before=float(scores[lane]), regret_before=float(regrets[lane]),
                                       remaining=max_new_tokens, generated_length=len(prefixes[lane]))
        legal_tokens = {token for token, _lp in values if token == 15 or max_new_tokens > 1}
        kept_tokens = {value[2] for value in kept}
        for token, logprob in values:
            rank = next(index for index, (ranked_token, _ranked_logprob) in enumerate(ranked, start=1) if ranked_token == token)
            common_trace = {
                "event_kind": "successor", "parent_node_id": parent_nodes[lane], "lane": lane,
                "prefix_length": len(prefixes[lane]) + 1, "prefix_hash": _prefix_hash(prefixes[lane] + (token,)),
                "last_token": token, "local_token_rank": rank, "token_logprob": logprob,
                "path_cumulative_nll": float(scores[lane]) - logprob,
                "path_cumulative_regret": float(regrets[lane]) + ranked[0][1] - logprob,
                "frontier_insert_order": None, "frontier_pop_order": None,
                "frontier_size_at_insert": None, "frontier_size_at_pop": None,
                "successors_considered": len(legal_tokens), "successors_retained": len(kept),
                "prune_reason": None, "dedup_rejected": False, "dedup_reason": None,
                "candidate_completed": False, "candidate_completion_index": None,
                "nodes_expanded_so_far": _lane_counter(state, "expanded_nodes", lane),
                "candidates_completed_so_far": _lane_counter(state, "completed_candidates", lane),
            }
            if token not in legal_tokens:
                _node(state, parent_node_id=parent_nodes[lane], lane=lane, token_position=pos, branch_depth=len(prefixes[lane]) + 1,
                      selected_token=token, token_logprob=logprob, cumulative_score=float(scores[lane]) - logprob,
                      state="pruned", prune_reason="max_new_tokens", termination_reason="max_new_tokens", branch_rank=None)
                _trace(state, **{**common_trace, "prune_reason": "max_new_tokens"})
            elif token not in kept_tokens:
                _node(state, parent_node_id=parent_nodes[lane], lane=lane, token_position=pos, branch_depth=len(prefixes[lane]) + 1,
                      selected_token=token, token_logprob=logprob, cumulative_score=float(scores[lane]) - logprob,
                      cumulative_regret=float(regrets[lane]) + ranked[0][1] - logprob,
                      state="pruned", prune_reason=prune_reason, termination_reason=None, branch_rank=None)
                _trace(state, **{**common_trace, "prune_reason": prune_reason})
        restored = False
        if not kept and legal_tokens:
            all_legal = [(float(scores[lane]) - logprob, float(regrets[lane]) + ranked[0][1] - logprob, token, logprob)
                         for token, logprob in ranked if token in legal_tokens]
            kept = sorted(all_legal, key=lambda value: (value[0], value[2]))[:config.frontier_floor]
            restored = True
            state["frontier_floor_events"].append({
                "activation_index": (int(state["frontier_floor_events_by_lane"][lane])
                                     if state.get("independent_lane_budgets", False)
                                     else len(state["frontier_floor_events"])),
                "lane": lane, "parent_node_id": parent_nodes[lane],
                "token_position": pos, "prefix_length": len(prefixes[lane]), "restored_count": len(kept),
                "restored_tokens": [value[2] for value in kept], "restored_cumulative_nll": [value[0] for value in kept],
                "reason": "policy_pruning_empty_frontier", "decoder_policy": config.policy_id,
            })
            if state.get("independent_lane_budgets", False):
                state["frontier_floor_events_by_lane"][lane] += 1
        for restore_rank, (score, regret, token, logprob) in enumerate(kept, start=1):
            rank = next(index for index, (ranked_token, _ranked_logprob) in enumerate(ranked, start=1) if ranked_token == token)
            retained_trace = {
                "event_kind": "successor", "parent_node_id": parent_nodes[lane], "lane": lane,
                "prefix_length": len(prefixes[lane]) + 1, "prefix_hash": _prefix_hash(prefixes[lane] + (token,)),
                "last_token": token, "local_token_rank": rank, "token_logprob": logprob,
                "path_cumulative_nll": score, "path_cumulative_regret": regret,
                "frontier_insert_order": None, "frontier_pop_order": None,
                "frontier_size_at_insert": None, "frontier_size_at_pop": None,
                "successors_considered": len(legal_tokens), "successors_retained": len(kept),
                "prune_reason": None, "dedup_rejected": False, "dedup_reason": None,
                "candidate_completed": False, "candidate_completion_index": None,
                "nodes_expanded_so_far": _lane_counter(state, "expanded_nodes", lane),
                "candidates_completed_so_far": _lane_counter(state, "completed_candidates", lane),
            }
            if token == 15:
                completed = (state["completed_candidates_by_lane"][lane] if state.get("independent_lane_budgets", False)
                             else state["completed_candidates"])
                if completed >= config.max_completed_candidates:
                    if state.get("independent_lane_budgets", False):
                        state["budget_exhausted_by_lane"][lane] = True
                    state["budget_exhausted"] = True
                    _node(state, parent_node_id=parent_nodes[lane], lane=lane, token_position=pos, branch_depth=len(prefixes[lane]) + 1,
                          selected_token=token, token_logprob=logprob, cumulative_score=score, cumulative_regret=regret,
                          state="pruned", prune_reason="candidate_budget", termination_reason="candidate_budget", branch_rank=None)
                    _trace(state, **{**retained_trace, "prune_reason": "candidate_budget"})
                    continue
                node_id = _node(state, parent_node_id=parent_nodes[lane], lane=lane, token_position=pos, branch_depth=len(prefixes[lane]) + 1,
                                selected_token=token, token_logprob=logprob, cumulative_score=score, cumulative_regret=regret,
                                state="completed", prune_reason=None, termination_reason="eos", branch_rank=None,
                                frontier_floor_activated=restored, frontier_floor_restore_rank=restore_rank if restored else None)
                if state.get("independent_lane_budgets", False):
                    candidate_id = int(state["next_candidate_id_by_lane"][lane])
                    state["next_candidate_id_by_lane"][lane] += 1
                    discovery_forward = int(state["model_forwards_by_lane"][lane])
                else:
                    candidate_id = int(state["next_candidate_id"])
                    state["next_candidate_id"] += 1
                    discovery_forward = int(state["model_forwards"])
                suffixes[lane].append(ReferenceTurboDFSCandidate(candidate_id, (15,), score, node_id, discovery_forward, time.time()))
                _trace(state, **{**retained_trace, "candidate_completed": True,
                                 "candidate_completion_index": candidate_id})
                state["completed_candidates"] += 1
                if state.get("independent_lane_budgets", False):
                    state["completed_candidates_by_lane"][lane] += 1
            else:
                candidates[lane].append((score, regret, token, restored, restore_rank if restored else None))
                insert_order = (int(state["next_frontier_insert_order_by_lane"][lane])
                                if state.get("independent_lane_budgets", False)
                                else int(state["next_frontier_insert_order"]))
                event = _trace(state, **{**retained_trace,
                                         "frontier_insert_order": insert_order,
                                         "frontier_size_at_insert": (len(candidates[lane])
                                                                      if state.get("independent_lane_budgets", False)
                                                                      else sum(len(values) for values in candidates))})
                if state.get("independent_lane_budgets", False):
                    state["next_frontier_insert_order_by_lane"][lane] += 1
                else:
                    state["next_frontier_insert_order"] += 1
                if event is not None:
                    state["trace_pending"][(lane, parent_nodes[lane], token, score)] = event
    for lane in range(lanes):
        candidates[lane].sort(key=lambda value: (value[0], value[2]))
    _record_frontier(state, candidates)

    while ((not state["budget_exhausted"] or state.get("independent_lane_budgets", False)) and time.time() - started_unix < config.local_time_limit_seconds and
           (config.absolute_end_time_unix is None or time.time() < config.absolute_end_time_unix)):
        tokens: list[int] = []; next_scores: list[float] = []; next_regrets: list[float] = []
        next_parents: list[int | None] = []; next_prefixes: list[tuple[int, ...]] = []
        next_active: list[bool] = []; active = 0
        for lane in range(lanes):
            expanded = (state["expanded_nodes_by_lane"][lane] if state.get("independent_lane_budgets", False)
                        else state["expanded_nodes"])
            lane_budget = (state["budget_exhausted_by_lane"][lane] if state.get("independent_lane_budgets", False)
                           else state["budget_exhausted"])
            if lane_budget or not candidates[lane] or expanded >= config.max_expanded_nodes:
                if expanded >= config.max_expanded_nodes:
                    if state.get("independent_lane_budgets", False):
                        state["budget_exhausted_by_lane"][lane] = True
                    state["budget_exhausted"] = True
                tokens.append(config.pad_token_id); next_scores.append(1000.0); next_regrets.append(1000.0)
                next_parents.append(parent_nodes[lane]); next_prefixes.append(prefixes[lane]); next_active.append(False); continue
            score, regret, token, restored, restore_rank = candidates[lane].pop(0)
            node_id = _node(state, parent_node_id=parent_nodes[lane], lane=lane, token_position=pos, branch_depth=len(prefixes[lane]) + 1,
                            selected_token=token, token_logprob=-(score - float(scores[lane])), cumulative_score=score,
                            cumulative_regret=regret, state="expanded", prune_reason=None, termination_reason=None, branch_rank=1,
                            frontier_floor_activated=restored, frontier_floor_restore_rank=restore_rank)
            state["expanded_nodes"] += 1
            if state.get("independent_lane_budgets", False):
                state["expanded_nodes_by_lane"][lane] += 1
            tokens.append(token); next_scores.append(score); next_regrets.append(regret)
            trace_event = state["trace_pending"].pop((lane, parent_nodes[lane], token, score), None)
            if trace_event is not None:
                pop_order = (int(state["next_frontier_pop_order_by_lane"][lane])
                             if state.get("independent_lane_budgets", False)
                             else int(state["next_frontier_pop_order"]))
                trace_event.update({"frontier_pop_order": pop_order,
                                    "frontier_size_at_pop": (len(candidates[lane])
                                                             if state.get("independent_lane_budgets", False)
                                                             else sum(len(values) for values in candidates)),
                                    "expanded_node_id": node_id,
                                    "nodes_expanded_so_far": _lane_counter(state, "expanded_nodes", lane),
                                    "elapsed_seconds": time.perf_counter() - float(state["trace_started_perf"])})
                if state.get("independent_lane_budgets", False):
                    state["next_frontier_pop_order_by_lane"][lane] += 1
                else:
                    state["next_frontier_pop_order"] += 1
            next_parents.append(node_id); next_prefixes.append(prefixes[lane] + (token,)); next_active.append(True); active += 1
        _record_frontier(state, candidates)
        if active == 0:
            break
        prep_started = time.perf_counter()
        next_input_ids = torch.tensor(tokens, device=model.device, dtype=torch.long).view(-1, 1)
        next_position_ids = torch.full((lanes, 1), pos, device=model.device)
        if state.get("performance_profile", False):
            state["performance_telemetry"]["batch_preparation_seconds"] += time.perf_counter() - prep_started
        forward_started = time.perf_counter()
        # The executor passes the cache through directly. It does not pack,
        # copy, or select per-lane KV tensors on this route.
        outputs = model(input_ids=next_input_ids, position_ids=next_position_ids,
                        past_key_values=cache, return_dict=True, use_cache=True)
        state["model_forward_seconds"] += time.perf_counter() - forward_started
        if config.calibration_assertions and int(outputs.logits.shape[0]) != lanes:
            raise RuntimeError("D1 model forward batch mismatch")
        state["model_forwards"] += 1; state["tokens_advanced"] += active
        if state.get("independent_lane_budgets", False):
            for lane, lane_active in enumerate(next_active):
                if lane_active:
                    state["model_forwards_by_lane"][lane] += 1
                    state["tokens_advanced_by_lane"][lane] += 1
        descendants = d1_turbo_dfs(model, logits=outputs.logits[:, -1], max_new_tokens=max_new_tokens - 1, scores=next_scores,
                                   regrets=next_regrets, pos=pos + 1, cache=outputs.past_key_values, config=config,
                                   started_unix=started_unix, state=state, parent_nodes=next_parents, prefixes=next_prefixes,
                                   active_mask=next_active)
        for lane, children in enumerate(descendants):
            for child in children:
                suffixes[lane].append(ReferenceTurboDFSCandidate(child.candidate_id, (tokens[lane],) + child.token_ids,
                                                                  child.cumulative_nll, child.terminal_node_id,
                                                                  child.discovery_forward_index, child.discovery_unix))
        del outputs
    return suffixes


def inference_d1_turbo_dfs(model: Any, *, input_ids: Any, config: D1TurboDFSConfig,
                            active_mask: tuple[bool, ...] | None = None) -> D1TurboDFSResult:
    """Run D1. ``active_mask`` is an executor-only physical-lane hook.

    Production callers leave it as ``None``.  It lets the cross-cell executor
    retain tensor/KV shape while proving an inactive physical slot has no
    logical search effects.
    """
    import torch
    if input_ids.ndim != 2 or int(input_ids.shape[0]) < 1:
        raise ValueError("D1 requires non-empty [lanes,tokens] input")
    lanes = int(input_ids.shape[0])
    logical_active = list(active_mask) if active_mask is not None else [True] * lanes
    if len(logical_active) != lanes:
        raise ValueError("D1 active_mask must match input batch lanes")
    state: dict[str, Any] = {"nodes": [], "branch_probabilities": [], "frontier_floor_events": [], "frontier_samples": [], "search_trace": [], "model_forwards": 0,
                             "tokens_advanced": 0, "completed_candidates": 0, "next_candidate_id": 0, "next_node_id": 0,
                             "next_trace_event_id": 0, "next_frontier_insert_order": 0, "next_frontier_pop_order": 0,
                             "trace_pending": {}, "trace_started_perf": time.perf_counter(), "diagnostic_trace": config.diagnostic_trace,
                             "model_forward_seconds": 0.0, "max_frontier_size": 0, "expanded_nodes": 0, "budget_exhausted": False,
                             "independent_lane_budgets": bool(config.independent_lane_budgets),
                             "performance_profile": bool(config.performance_profile),
                             "performance_telemetry": {"batch_preparation_seconds": 0.0,
                                                       "trace_telemetry_seconds": 0.0,
                                                       "node_and_state_seconds": 0.0,
                                                       "frontier_bookkeeping_seconds": 0.0,
                                                       "kv_pack_or_select_seconds": 0.0},
                             "model_forwards_by_lane": [0] * lanes, "tokens_advanced_by_lane": [0] * lanes,
                             "completed_candidates_by_lane": [0] * lanes, "expanded_nodes_by_lane": [0] * lanes,
                             "budget_exhausted_by_lane": [False] * lanes, "next_candidate_id_by_lane": [0] * lanes,
                             "next_node_id_by_lane": [0] * lanes, "next_trace_event_id_by_lane": [0] * lanes,
                             "next_frontier_insert_order_by_lane": [0] * lanes, "next_frontier_pop_order_by_lane": [0] * lanes,
                             "max_frontier_size_by_lane": [0] * lanes,
                             "frontier_floor_events_by_lane": [0] * lanes}
    started = time.time()
    with torch.no_grad():
        forward_started = time.perf_counter()
        outputs = model(input_ids=input_ids, return_dict=True, use_cache=True)
        state["model_forward_seconds"] += time.perf_counter() - forward_started
        state["model_forwards"] += 1
        if config.independent_lane_budgets:
            for lane in range(lanes):
                if logical_active[lane]:
                    state["model_forwards_by_lane"][lane] = 1
        roots = [(_node(state, parent_node_id=None, lane=lane, token_position=int(input_ids.size(1)), branch_depth=0,
                        selected_token=None, token_logprob=None, cumulative_score=0.0, state="root", prune_reason=None,
                        termination_reason=None, branch_rank=None, cumulative_regret=0.0)
                  if logical_active[lane] else None) for lane in range(lanes)]
        candidates = d1_turbo_dfs(model, logits=outputs.logits[:, -1], max_new_tokens=config.max_new_tokens, scores=[0.0] * lanes,
                                  regrets=[0.0] * lanes, pos=int(input_ids.size(1)), cache=outputs.past_key_values, config=config,
                                  started_unix=started, state=state, parent_nodes=roots, prefixes=[tuple() for _ in range(lanes)],
                                  active_mask=logical_active)
        del outputs
    timed_out = time.time() - started >= config.local_time_limit_seconds or (config.absolute_end_time_unix is not None and time.time() >= config.absolute_end_time_unix)
    lane_reason = tuple(
        "budget_exhausted" if state["budget_exhausted_by_lane"][lane] else "wall_time" if timed_out else "search_exhausted"
        for lane in range(lanes)
    )
    return D1TurboDFSResult(tuple(tuple(sorted(items, key=lambda item: item.cumulative_nll)) for items in candidates), tuple(state["nodes"]),
                             tuple(state["branch_probabilities"]), tuple(state["frontier_floor_events"]), tuple(state["frontier_samples"]), tuple(state["search_trace"]), float(state["model_forward_seconds"]), int(state["model_forwards"]),
                             int(state["tokens_advanced"]), int(state["completed_candidates"]), int(state["max_frontier_size"]),
                             bool(state["budget_exhausted"]), timed_out,
                             "budget_exhausted" if state["budget_exhausted"] else "wall_time" if timed_out else "search_exhausted",
                             tuple(int(value) for value in state["model_forwards_by_lane"]),
                             tuple(int(value) for value in state["tokens_advanced_by_lane"]),
                             tuple(int(value) for value in state["completed_candidates_by_lane"]),
                             tuple(int(value) for value in state["expanded_nodes_by_lane"]),
                             tuple(bool(value) for value in state["budget_exhausted_by_lane"]), lane_reason,
                             tuple(int(value) for value in state["max_frontier_size_by_lane"]),
                             dict(state["performance_telemetry"]))
