"""TurboDFS V5: V4 public search with a preregistered frontier floor.

V5 deliberately keeps the public NVARC recursive search, native token set,
score definition, ordering and clocks.  It changes exactly one edge case: if
public cumulative-NLL pruning would leave a lane with no legal successor,
the best already-considered legal successor is restored.  This is a search
semantics extension, not a claim of public-reference parity.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
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


@dataclass(frozen=True)
class FrontierFloorTurboDFSConfig:
    max_new_tokens: int
    max_score: float
    absolute_end_time_unix: float | None
    frontier_floor: int = 1
    local_time_limit_seconds: float = PUBLIC_LOCAL_TIME_LIMIT_SECONDS
    pad_token_id: int = PUBLIC_PAD_ID
    arc_tokens: tuple[int, ...] = PUBLIC_ARC_TOKENS
    capture_full_arc_distribution: bool = True
    calibration_assertions: bool = True

    def __post_init__(self) -> None:
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if not math.isfinite(self.max_score) or self.max_score <= 0:
            raise ValueError("max_score must be finite and positive")
        if self.frontier_floor not in (1, 2):
            raise ValueError("V5 supports only preregistered frontier floor 1 or 2")
        if self.local_time_limit_seconds != PUBLIC_LOCAL_TIME_LIMIT_SECONDS:
            raise ValueError("V5 must retain the public 540-second local decoder limit")
        if self.pad_token_id != PUBLIC_PAD_ID or tuple(self.arc_tokens) != PUBLIC_ARC_TOKENS:
            raise ValueError("V5 native-token contract differs from V4/public decoder")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["arc_tokens"] = list(self.arc_tokens)
        return value


@dataclass(frozen=True)
class FrontierFloorTurboDFSResult:
    candidates: tuple[tuple[ReferenceTurboDFSCandidate, ...], ...]
    nodes: tuple[dict[str, Any], ...]
    branch_probabilities: tuple[dict[str, Any], ...]
    frontier_floor_events: tuple[dict[str, Any], ...]
    model_forwards: int
    tokens_advanced: int
    completed_candidates: int
    max_frontier_size: int
    timed_out: bool
    termination_reason: str


def _record_node(state: dict[str, Any], **payload: Any) -> int:
    """Keep V4 tree fields, adding explicit V5 provenance where relevant."""
    payload.setdefault("frontier_floor_activated", False)
    payload.setdefault("frontier_floor_restore_rank", None)
    return state["new_node"](**payload)


def frontier_floor_turbo_dfs(
    model: Any, *, logits: Any, max_new_tokens: int, scores: list[float], pos: int,
    cache: Any, config: FrontierFloorTurboDFSConfig, started_unix: float,
    state: dict[str, Any], parent_nodes: list[int | None], prefixes: list[tuple[int, ...]],
) -> list[list[ReferenceTurboDFSCandidate]]:
    """Recursive public TurboDFS with the sole V5 zero-frontier repair."""
    import torch

    n = int(logits.size(0))
    if config.calibration_assertions:
        if len(scores) != n or len(parent_nodes) != n or len(prefixes) != n:
            raise RuntimeError("V5 lane metadata does not match public logits batch")
        cache_n = _cache_batch_size(cache)
        if cache_n is not None and cache_n != n:
            raise RuntimeError(f"V5 KV cache batch mismatch: cache={cache_n}, lanes={n}")

    log_probs = logits.float().cpu().log_softmax(-1)
    nll = torch.tensor(scores, dtype=torch.float32).view(n, 1) - log_probs
    suffixes: list[list[ReferenceTurboDFSCandidate]] = [[] for _ in range(n)]
    candidates: list[list[tuple[float, int, bool, int | None]]] = [[] for _ in range(n)]
    state["max_frontier_size"] = max(state["max_frontier_size"], sum(len(x) for x in candidates))

    for lane in range(n):
        values = [(token, float(log_probs[lane, token].item())) for token in config.arc_tokens]
        ranked = sorted(values, key=lambda pair: (-pair[1], pair[0]))
        probs = [math.exp(value) for _token, value in values]
        entropy = -sum(prob * value for prob, (_token, value) in zip(probs, values, strict=True))
        state["branch_probabilities"].append({
            "forward_index": state["model_forwards"], "lane": lane,
            "parent_node_id": parent_nodes[lane], "token_position": pos,
            "prefix_length": len(prefixes[lane]), "cumulative_score_before": float(scores[lane]),
            "full_arc_logprobs": [{"token_id": token, "logprob": value} for token, value in values],
            "top1_token_id": ranked[0][0], "top1_logprob": ranked[0][1],
            "top2_token_id": ranked[1][0], "top2_logprob": ranked[1][1],
            "margin": ranked[0][1] - ranked[1][1], "entropy": entropy,
            "batch_size": n, "active_lane_count": None,
        })
        # S contains only legal native-token continuations.  At the final
        # position, non-EOS tokens cannot legally be expanded further.
        successors: list[tuple[float, int, float]] = []
        survivors: list[tuple[float, int, float]] = []
        for token in config.arc_tokens:
            score = float(nll[lane, token].item())
            token_logprob = float(log_probs[lane, token].item())
            legal = token == 15 or max_new_tokens > 1
            if legal:
                successors.append((score, int(token), token_logprob))
            if legal and score < config.max_score:  # strict public inequality
                survivors.append((score, int(token), token_logprob))
            elif legal:
                _record_node(
                    state, parent_node_id=parent_nodes[lane], lane=lane, token_position=pos,
                    branch_depth=len(prefixes[lane]) + 1, selected_token=int(token),
                    token_logprob=token_logprob, cumulative_score=score, state="pruned",
                    prune_reason="public_cumulative_nll", termination_reason=None, branch_rank=None,
                )
            else:
                _record_node(
                    state, parent_node_id=parent_nodes[lane], lane=lane, token_position=pos,
                    branch_depth=len(prefixes[lane]) + 1, selected_token=int(token),
                    token_logprob=token_logprob, cumulative_score=score, state="pruned",
                    prune_reason="max_new_tokens", termination_reason="max_new_tokens", branch_rank=None,
                )
        restored = False
        if not survivors and successors:
            # Public pruning would kill this lane.  Stable (score, token)
            # order preserves the public token ordering on numerical ties.
            survivors = sorted(successors, key=lambda item: (item[0], item[1]))[:config.frontier_floor]
            restored = True
            state["frontier_floor_events"].append({
                "activation_index": len(state["frontier_floor_events"]), "lane": lane,
                "parent_node_id": parent_nodes[lane], "token_position": pos,
                "prefix_length": len(prefixes[lane]), "restored_count": len(survivors),
                "restored_tokens": [token for _score, token, _lp in survivors],
                "restored_cumulative_nll": [score for score, _token, _lp in survivors],
                "reason": "public_pruning_empty_frontier",
            })
        for restore_rank, (score, token, token_logprob) in enumerate(sorted(survivors, key=lambda item: item[0]), start=1):
            floor_mark = restored
            rank = restore_rank if restored else None
            if token == 15:
                node_id = _record_node(
                    state, parent_node_id=parent_nodes[lane], lane=lane, token_position=pos,
                    branch_depth=len(prefixes[lane]) + 1, selected_token=token,
                    token_logprob=token_logprob, cumulative_score=score, state="completed",
                    prune_reason=None, termination_reason="eos", branch_rank=None,
                    frontier_floor_activated=floor_mark, frontier_floor_restore_rank=rank,
                )
                candidate = ReferenceTurboDFSCandidate(
                    candidate_id=state["next_candidate_id"], token_ids=(15,), cumulative_nll=score,
                    terminal_node_id=node_id, discovery_forward_index=state["model_forwards"],
                    discovery_unix=time.time(),
                )
                state["next_candidate_id"] += 1
                state["completed_candidates"] += 1
                suffixes[lane].append(candidate)
            else:
                candidates[lane].append((score, token, floor_mark, rank))

    for lane in range(n):
        candidates[lane] = sorted(candidates[lane], key=lambda item: item[0])
    state["max_frontier_size"] = max(state["max_frontier_size"], sum(len(x) for x in candidates))

    while (time.time() - started_unix < config.local_time_limit_seconds and
           (config.absolute_end_time_unix is None or time.time() < config.absolute_end_time_unix)):
        batch_tokens: list[int] = []
        batch_scores: list[float] = []
        next_parent_nodes: list[int | None] = []
        next_prefixes: list[tuple[int, ...]] = []
        alive = 0
        for lane in range(n):
            if not candidates[lane]:
                batch_tokens.append(config.pad_token_id); batch_scores.append(1000.0)
                next_parent_nodes.append(parent_nodes[lane]); next_prefixes.append(prefixes[lane])
                continue
            score, token, floor_mark, restore_rank = candidates[lane].pop(0)
            rank = 1 + sum(1 for other_score, *_rest in candidates[lane] if other_score < score)
            node_id = _record_node(
                state, parent_node_id=parent_nodes[lane], lane=lane, token_position=pos,
                branch_depth=len(prefixes[lane]) + 1, selected_token=token,
                token_logprob=-(score - float(scores[lane])), cumulative_score=score,
                state="expanded", prune_reason=None, termination_reason=None, branch_rank=rank,
                frontier_floor_activated=floor_mark, frontier_floor_restore_rank=restore_rank,
            )
            batch_tokens.append(token); batch_scores.append(score); next_parent_nodes.append(node_id)
            next_prefixes.append(prefixes[lane] + (token,)); alive += 1
        if alive == 0:
            break
        if config.calibration_assertions and len({next_parent_nodes[i] for i in range(n) if batch_tokens[i] != config.pad_token_id}) != alive:
            raise RuntimeError("V5 lane branch state aliases before model forward")
        outputs = model(
            input_ids=torch.tensor(batch_tokens, device=model.device, dtype=torch.long).view(-1, 1),
            position_ids=torch.full((n, 1), pos, device=model.device),
            past_key_values=cache, return_dict=True, use_cache=True,
        )
        if config.calibration_assertions and int(outputs.logits.shape[0]) != n:
            raise RuntimeError("V5 model forward returned wrong lane count")
        state["model_forwards"] += 1; state["tokens_advanced"] += alive
        descendants = frontier_floor_turbo_dfs(
            model, logits=outputs.logits[:, -1], max_new_tokens=max_new_tokens - 1,
            scores=batch_scores, pos=pos + 1, cache=outputs.past_key_values, config=config,
            started_unix=started_unix, state=state, parent_nodes=next_parent_nodes,
            prefixes=next_prefixes,
        )
        for lane, beams in enumerate(descendants):
            for candidate in beams:
                suffixes[lane].append(ReferenceTurboDFSCandidate(
                    candidate_id=candidate.candidate_id, token_ids=(batch_tokens[lane],) + candidate.token_ids,
                    cumulative_nll=candidate.cumulative_nll, terminal_node_id=candidate.terminal_node_id,
                    discovery_forward_index=candidate.discovery_forward_index,
                    discovery_unix=candidate.discovery_unix,
                ))
        del outputs
    return suffixes


def inference_frontier_floor_turbo_dfs(
    model: Any, *, input_ids: Any, config: FrontierFloorTurboDFSConfig,
) -> FrontierFloorTurboDFSResult:
    """V5 entrypoint with V4-equivalent root/cache plumbing."""
    import torch
    if input_ids.ndim != 2 or int(input_ids.shape[0]) < 1:
        raise ValueError("input_ids must be a non-empty [lanes, tokens] tensor")
    n = int(input_ids.shape[0])
    state: dict[str, Any] = {
        "nodes": [], "branch_probabilities": [], "frontier_floor_events": [],
        "model_forwards": 0, "tokens_advanced": 0, "completed_candidates": 0,
        "next_candidate_id": 0, "next_node_id": 0, "max_frontier_size": 0,
    }
    def new_node(**payload: Any) -> int:
        node_id = int(state["next_node_id"]); state["next_node_id"] += 1
        state["nodes"].append({"node_id": node_id, **payload}); return node_id
    state["new_node"] = new_node
    started = time.time()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, return_dict=True, use_cache=True)
        if config.calibration_assertions and int(outputs.logits.shape[0]) != n:
            raise RuntimeError("V5 root forward returned wrong lane count")
        state["model_forwards"] += 1
        roots = [_record_node(
            state, parent_node_id=None, lane=lane, token_position=int(input_ids.size(1)), branch_depth=0,
            selected_token=None, token_logprob=None, cumulative_score=0.0, state="root",
            prune_reason=None, termination_reason=None, branch_rank=None,
        ) for lane in range(n)]
        raw = frontier_floor_turbo_dfs(
            model, logits=outputs.logits[:, -1], max_new_tokens=config.max_new_tokens,
            scores=[0.0] * n, pos=int(input_ids.size(1)), cache=outputs.past_key_values,
            config=config, started_unix=started, state=state, parent_nodes=roots,
            prefixes=[tuple() for _ in range(n)],
        )
        del outputs
    timed_out = (time.time() - started >= config.local_time_limit_seconds or
                 (config.absolute_end_time_unix is not None and time.time() >= config.absolute_end_time_unix))
    return FrontierFloorTurboDFSResult(
        candidates=tuple(tuple(sorted(beams, key=lambda c: c.cumulative_nll)) for beams in raw),
        nodes=tuple(state["nodes"]), branch_probabilities=tuple(state["branch_probabilities"]),
        frontier_floor_events=tuple(state["frontier_floor_events"]),
        model_forwards=int(state["model_forwards"]), tokens_advanced=int(state["tokens_advanced"]),
        completed_candidates=int(state["completed_candidates"]), max_frontier_size=int(state["max_frontier_size"]),
        timed_out=timed_out, termination_reason="wall_time" if timed_out else "search_exhausted",
    )
