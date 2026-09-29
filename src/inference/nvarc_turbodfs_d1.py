"""Bounded, target-blind D1 decoder variants for the real A/B pilot.

The module deliberately keeps V5's native-token legality, recursive KV-cache
route, ordering, max length, wall clock, floor semantics, and candidate format.
It adds a *common D1 external safety budget* to all policies, then changes only
the successor-retention predicate.  The safety budget is predeclared from
frozen V5 telemetry and is checked explicitly by the runner.
"""
from __future__ import annotations

from dataclasses import dataclass
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
    model_forwards: int
    tokens_advanced: int
    completed_candidates: int
    max_frontier_size: int
    budget_exhausted: bool
    timed_out: bool
    termination_reason: str


def _node(runtime_state: dict[str, Any], **payload: Any) -> int:
    payload.setdefault("frontier_floor_activated", False)
    payload.setdefault("frontier_floor_restore_rank", None)
    payload.setdefault("cumulative_regret", None)
    value = int(runtime_state["next_node_id"])
    runtime_state["next_node_id"] += 1
    runtime_state["nodes"].append({"node_id": value, **payload})
    return value


def _record_frontier(state: dict[str, Any], candidates: list[list[Any]]) -> None:
    """Capture actual pending successor counts without changing search order."""
    lanes = [len(values) for values in candidates]
    total = sum(lanes)
    state["max_frontier_size"] = max(state["max_frontier_size"], total)
    state["frontier_samples"].append({"total": total, "lanes": lanes})


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
) -> list[list[ReferenceTurboDFSCandidate]]:
    import torch

    lanes = int(logits.size(0))
    if config.calibration_assertions:
        if not (len(scores) == len(regrets) == len(parent_nodes) == len(prefixes) == lanes):
            raise RuntimeError("D1 lane metadata mismatch")
        cached = _cache_batch_size(cache)
        if cached is not None and cached != lanes:
            raise RuntimeError(f"D1 KV cache batch mismatch cache={cached} lanes={lanes}")
    log_probs = logits.float().cpu().log_softmax(-1)
    suffixes: list[list[ReferenceTurboDFSCandidate]] = [[] for _ in range(lanes)]
    candidates: list[list[tuple[float, float, int, bool, int | None]]] = [[] for _ in range(lanes)]
    _record_frontier(state, candidates)

    for lane in range(lanes):
        values = [(token, float(log_probs[lane, token].item())) for token in config.arc_tokens]
        ranked = sorted(values, key=lambda pair: (-pair[1], pair[0]))
        probabilities = [math.exp(logprob) for _token, logprob in values]
        state["branch_probabilities"].append({
            "forward_index": state["model_forwards"], "lane": lane, "parent_node_id": parent_nodes[lane],
            "token_position": pos, "prefix_length": len(prefixes[lane]), "cumulative_score_before": float(scores[lane]),
            "cumulative_regret_before": float(regrets[lane]), "full_arc_logprobs": [{"token_id": token, "logprob": logprob} for token, logprob in values],
            "top1_token_id": ranked[0][0], "top1_logprob": ranked[0][1], "top2_token_id": ranked[1][0], "top2_logprob": ranked[1][1],
            "margin": ranked[0][1] - ranked[1][1], "entropy": -sum(p * lp for p, (_token, lp) in zip(probabilities, values, strict=True)),
            "batch_size": lanes, "active_lane_count": None, "decoder_policy": config.policy_id,
        })
        kept, prune_reason = _retained(config, ranked, score_before=float(scores[lane]), regret_before=float(regrets[lane]),
                                       remaining=max_new_tokens, generated_length=len(prefixes[lane]))
        legal_tokens = {token for token, _lp in values if token == 15 or max_new_tokens > 1}
        kept_tokens = {value[2] for value in kept}
        for token, logprob in values:
            if token not in legal_tokens:
                _node(state, parent_node_id=parent_nodes[lane], lane=lane, token_position=pos, branch_depth=len(prefixes[lane]) + 1,
                      selected_token=token, token_logprob=logprob, cumulative_score=float(scores[lane]) - logprob,
                      state="pruned", prune_reason="max_new_tokens", termination_reason="max_new_tokens", branch_rank=None)
            elif token not in kept_tokens:
                _node(state, parent_node_id=parent_nodes[lane], lane=lane, token_position=pos, branch_depth=len(prefixes[lane]) + 1,
                      selected_token=token, token_logprob=logprob, cumulative_score=float(scores[lane]) - logprob,
                      cumulative_regret=float(regrets[lane]) + ranked[0][1] - logprob,
                      state="pruned", prune_reason=prune_reason, termination_reason=None, branch_rank=None)
        restored = False
        if not kept and legal_tokens:
            all_legal = [(float(scores[lane]) - logprob, float(regrets[lane]) + ranked[0][1] - logprob, token, logprob)
                         for token, logprob in ranked if token in legal_tokens]
            kept = sorted(all_legal, key=lambda value: (value[0], value[2]))[:config.frontier_floor]
            restored = True
            state["frontier_floor_events"].append({
                "activation_index": len(state["frontier_floor_events"]), "lane": lane, "parent_node_id": parent_nodes[lane],
                "token_position": pos, "prefix_length": len(prefixes[lane]), "restored_count": len(kept),
                "restored_tokens": [value[2] for value in kept], "restored_cumulative_nll": [value[0] for value in kept],
                "reason": "policy_pruning_empty_frontier", "decoder_policy": config.policy_id,
            })
        for restore_rank, (score, regret, token, logprob) in enumerate(kept, start=1):
            if token == 15:
                if state["completed_candidates"] >= config.max_completed_candidates:
                    state["budget_exhausted"] = True
                    _node(state, parent_node_id=parent_nodes[lane], lane=lane, token_position=pos, branch_depth=len(prefixes[lane]) + 1,
                          selected_token=token, token_logprob=logprob, cumulative_score=score, cumulative_regret=regret,
                          state="pruned", prune_reason="candidate_budget", termination_reason="candidate_budget", branch_rank=None)
                    continue
                node_id = _node(state, parent_node_id=parent_nodes[lane], lane=lane, token_position=pos, branch_depth=len(prefixes[lane]) + 1,
                                selected_token=token, token_logprob=logprob, cumulative_score=score, cumulative_regret=regret,
                                state="completed", prune_reason=None, termination_reason="eos", branch_rank=None,
                                frontier_floor_activated=restored, frontier_floor_restore_rank=restore_rank if restored else None)
                suffixes[lane].append(ReferenceTurboDFSCandidate(state["next_candidate_id"], (15,), score, node_id, state["model_forwards"], time.time()))
                state["next_candidate_id"] += 1; state["completed_candidates"] += 1
            else:
                candidates[lane].append((score, regret, token, restored, restore_rank if restored else None))
    for lane in range(lanes):
        candidates[lane].sort(key=lambda value: (value[0], value[2]))
    _record_frontier(state, candidates)

    while (not state["budget_exhausted"] and time.time() - started_unix < config.local_time_limit_seconds and
           (config.absolute_end_time_unix is None or time.time() < config.absolute_end_time_unix)):
        tokens: list[int] = []; next_scores: list[float] = []; next_regrets: list[float] = []
        next_parents: list[int | None] = []; next_prefixes: list[tuple[int, ...]] = []; active = 0
        for lane in range(lanes):
            if not candidates[lane] or state["expanded_nodes"] >= config.max_expanded_nodes:
                if state["expanded_nodes"] >= config.max_expanded_nodes:
                    state["budget_exhausted"] = True
                tokens.append(config.pad_token_id); next_scores.append(1000.0); next_regrets.append(1000.0)
                next_parents.append(parent_nodes[lane]); next_prefixes.append(prefixes[lane]); continue
            score, regret, token, restored, restore_rank = candidates[lane].pop(0)
            node_id = _node(state, parent_node_id=parent_nodes[lane], lane=lane, token_position=pos, branch_depth=len(prefixes[lane]) + 1,
                            selected_token=token, token_logprob=-(score - float(scores[lane])), cumulative_score=score,
                            cumulative_regret=regret, state="expanded", prune_reason=None, termination_reason=None, branch_rank=1,
                            frontier_floor_activated=restored, frontier_floor_restore_rank=restore_rank)
            state["expanded_nodes"] += 1; tokens.append(token); next_scores.append(score); next_regrets.append(regret)
            next_parents.append(node_id); next_prefixes.append(prefixes[lane] + (token,)); active += 1
        _record_frontier(state, candidates)
        if active == 0:
            break
        outputs = model(input_ids=torch.tensor(tokens, device=model.device, dtype=torch.long).view(-1, 1),
                        position_ids=torch.full((lanes, 1), pos, device=model.device), past_key_values=cache, return_dict=True, use_cache=True)
        if config.calibration_assertions and int(outputs.logits.shape[0]) != lanes:
            raise RuntimeError("D1 model forward batch mismatch")
        state["model_forwards"] += 1; state["tokens_advanced"] += active
        descendants = d1_turbo_dfs(model, logits=outputs.logits[:, -1], max_new_tokens=max_new_tokens - 1, scores=next_scores,
                                   regrets=next_regrets, pos=pos + 1, cache=outputs.past_key_values, config=config,
                                   started_unix=started_unix, state=state, parent_nodes=next_parents, prefixes=next_prefixes)
        for lane, children in enumerate(descendants):
            for child in children:
                suffixes[lane].append(ReferenceTurboDFSCandidate(child.candidate_id, (tokens[lane],) + child.token_ids,
                                                                  child.cumulative_nll, child.terminal_node_id,
                                                                  child.discovery_forward_index, child.discovery_unix))
        del outputs
    return suffixes


def inference_d1_turbo_dfs(model: Any, *, input_ids: Any, config: D1TurboDFSConfig) -> D1TurboDFSResult:
    import torch
    if input_ids.ndim != 2 or int(input_ids.shape[0]) < 1:
        raise ValueError("D1 requires non-empty [lanes,tokens] input")
    lanes = int(input_ids.shape[0])
    state: dict[str, Any] = {"nodes": [], "branch_probabilities": [], "frontier_floor_events": [], "frontier_samples": [], "model_forwards": 0,
                             "tokens_advanced": 0, "completed_candidates": 0, "next_candidate_id": 0, "next_node_id": 0,
                             "max_frontier_size": 0, "expanded_nodes": 0, "budget_exhausted": False}
    started = time.time()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, return_dict=True, use_cache=True)
        state["model_forwards"] += 1
        roots = [_node(state, parent_node_id=None, lane=lane, token_position=int(input_ids.size(1)), branch_depth=0,
                       selected_token=None, token_logprob=None, cumulative_score=0.0, state="root", prune_reason=None,
                       termination_reason=None, branch_rank=None, cumulative_regret=0.0) for lane in range(lanes)]
        candidates = d1_turbo_dfs(model, logits=outputs.logits[:, -1], max_new_tokens=config.max_new_tokens, scores=[0.0] * lanes,
                                  regrets=[0.0] * lanes, pos=int(input_ids.size(1)), cache=outputs.past_key_values, config=config,
                                  started_unix=started, state=state, parent_nodes=roots, prefixes=[tuple() for _ in range(lanes)])
        del outputs
    timed_out = time.time() - started >= config.local_time_limit_seconds or (config.absolute_end_time_unix is not None and time.time() >= config.absolute_end_time_unix)
    return D1TurboDFSResult(tuple(tuple(sorted(items, key=lambda item: item.cumulative_nll)) for items in candidates), tuple(state["nodes"]),
                             tuple(state["branch_probabilities"]), tuple(state["frontier_floor_events"]), tuple(state["frontier_samples"]), int(state["model_forwards"]),
                             int(state["tokens_advanced"]), int(state["completed_candidates"]), int(state["max_frontier_size"]),
                             bool(state["budget_exhausted"]), timed_out,
                             "budget_exhausted" if state["budget_exhausted"] else "wall_time" if timed_out else "search_exhausted")
