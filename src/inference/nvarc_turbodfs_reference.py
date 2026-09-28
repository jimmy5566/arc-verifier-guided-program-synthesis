"""Direct, traceable implementation of NVARC's public TurboDFS.

This module is intentionally separate from the historical V1--V3 decoder.
It mirrors ``ARC-AGI1/002_ivan_arc1.ipynb`` cell 4 (``turbo_dfs`` and
``inference_turbo_dfs``) at NVARC commit
``846d0198efa752534594e321fc3289fc0a06c657``.  ARC2 supplies only the
model, saved adapter, tokenizer, and already-serialized prompt; this file
owns the search semantics.

The public decoder has exactly two termination clocks: a 540-second local
decoder allowance and the caller's absolute end time.  It has no forward
count, frontier-size, retained-candidate, or emergency-pruning cap.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time
from typing import Any


PUBLIC_NVARC_COMMIT = "846d0198efa752534594e321fc3289fc0a06c657"
PUBLIC_NOTEBOOK_PATH = "ARC-AGI1/002_ivan_arc1.ipynb"
PUBLIC_NOTEBOOK_CELL = 4
PUBLIC_LOCAL_TIME_LIMIT_SECONDS = 540.0
PUBLIC_ARC_TOKENS = tuple(range(11)) + (15,)
PUBLIC_PAD_ID = 13


@dataclass(frozen=True)
class ReferenceTurboDFSConfig:
    """Frozen public-search contract; no local safety caps are permitted."""

    max_new_tokens: int
    max_score: float
    absolute_end_time_unix: float | None
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
        if self.local_time_limit_seconds != PUBLIC_LOCAL_TIME_LIMIT_SECONDS:
            raise ValueError("V4 must use the public 540-second local decoder limit")
        if self.pad_token_id != PUBLIC_PAD_ID or tuple(self.arc_tokens) != PUBLIC_ARC_TOKENS:
            raise ValueError("V4 native-token contract differs from the public decoder")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["arc_tokens"] = list(self.arc_tokens)
        return value


@dataclass(frozen=True)
class ReferenceTurboDFSCandidate:
    candidate_id: int
    token_ids: tuple[int, ...]
    cumulative_nll: float
    terminal_node_id: int
    discovery_forward_index: int
    discovery_unix: float


@dataclass(frozen=True)
class ReferenceTurboDFSResult:
    candidates: tuple[tuple[ReferenceTurboDFSCandidate, ...], ...]
    nodes: tuple[dict[str, Any], ...]
    branch_probabilities: tuple[dict[str, Any], ...]
    model_forwards: int
    tokens_advanced: int
    completed_candidates: int
    max_frontier_size: int
    timed_out: bool
    termination_reason: str


def _cache_batch_size(cache: Any) -> int | None:
    """Best-effort non-mutating cache shape check for calibration telemetry."""
    for name in ("key_cache", "value_cache"):
        rows = getattr(cache, name, None)
        if rows:
            item = rows[0]
            if hasattr(item, "shape") and len(item.shape):
                return int(item.shape[0])
    if isinstance(cache, (list, tuple)) and cache:
        layer = cache[0]
        if isinstance(layer, (list, tuple)) and layer and hasattr(layer[0], "shape"):
            return int(layer[0].shape[0])
    return None


def reference_turbo_dfs(
    model: Any,
    *,
    logits: Any,
    max_new_tokens: int,
    scores: list[float],
    pos: int,
    cache: Any,
    config: ReferenceTurboDFSConfig,
    started_unix: float,
    state: dict[str, Any],
    parent_nodes: list[int | None],
    prefixes: list[tuple[int, ...]],
) -> list[list[ReferenceTurboDFSCandidate]]:
    """Faithful recursive public ``turbo_dfs`` with additive tracing only."""
    import torch

    n = int(logits.size(0))
    if config.calibration_assertions:
        if len(scores) != n or len(parent_nodes) != n or len(prefixes) != n:
            raise RuntimeError("V4 lane metadata does not match public logits batch")
        cache_n = _cache_batch_size(cache)
        if cache_n is not None and cache_n != n:
            raise RuntimeError(f"V4 KV cache batch mismatch: cache={cache_n}, lanes={n}")

    # This is exactly the public formula:
    # torch.tensor(scores).view(n, 1) - logits.float().cpu().log_softmax(-1)
    log_probs = logits.float().cpu().log_softmax(-1)
    nll = torch.tensor(scores, dtype=torch.float32).view(n, 1) - log_probs
    suffixes: list[list[ReferenceTurboDFSCandidate]] = [[] for _ in range(n)]
    candidates: list[list[tuple[float, int]]] = [[] for _ in range(n)]
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
        for token in config.arc_tokens:
            score = float(nll[lane, token].item())
            token_logprob = float(log_probs[lane, token].item())
            if score < config.max_score:  # strict public inequality
                if token == 15:
                    node_id = state["new_node"](
                        parent_node_id=parent_nodes[lane], lane=lane, token_position=pos,
                        branch_depth=len(prefixes[lane]) + 1, selected_token=token,
                        token_logprob=token_logprob, cumulative_score=score,
                        state="completed", prune_reason=None, termination_reason="eos",
                        branch_rank=None,
                    )
                    candidate = ReferenceTurboDFSCandidate(
                        candidate_id=state["next_candidate_id"], token_ids=(15,),
                        cumulative_nll=score, terminal_node_id=node_id,
                        discovery_forward_index=state["model_forwards"], discovery_unix=time.time(),
                    )
                    state["next_candidate_id"] += 1
                    state["completed_candidates"] += 1
                    suffixes[lane].append(candidate)
                elif max_new_tokens > 1:
                    candidates[lane].append((score, int(token)))
                else:
                    state["new_node"](
                        parent_node_id=parent_nodes[lane], lane=lane, token_position=pos,
                        branch_depth=len(prefixes[lane]) + 1, selected_token=int(token),
                        token_logprob=token_logprob, cumulative_score=score,
                        state="pruned", prune_reason="max_new_tokens", termination_reason="max_new_tokens",
                        branch_rank=None,
                    )
            else:
                state["new_node"](
                    parent_node_id=parent_nodes[lane], lane=lane, token_position=pos,
                    branch_depth=len(prefixes[lane]) + 1, selected_token=int(token),
                    token_logprob=token_logprob, cumulative_score=score,
                    state="pruned", prune_reason="public_cumulative_nll", termination_reason=None,
                    branch_rank=None,
                )

    # Public code sorts by score only; Python stability preserves ARC_TOKENS order on ties.
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
                batch_tokens.append(config.pad_token_id)
                batch_scores.append(1000.0)
                next_parent_nodes.append(parent_nodes[lane])
                next_prefixes.append(prefixes[lane])
                continue
            score, token = candidates[lane].pop(0)
            rank = 1 + sum(1 for other_score, _ in candidates[lane] if other_score < score)
            node_id = state["new_node"](
                parent_node_id=parent_nodes[lane], lane=lane, token_position=pos,
                branch_depth=len(prefixes[lane]) + 1, selected_token=token,
                token_logprob=-(score - float(scores[lane])), cumulative_score=score,
                state="expanded", prune_reason=None, termination_reason=None, branch_rank=rank,
            )
            batch_tokens.append(token)
            batch_scores.append(score)
            next_parent_nodes.append(node_id)
            next_prefixes.append(prefixes[lane] + (token,))
            alive += 1
        if alive == 0:
            break
        if config.calibration_assertions and len(set(next_parent_nodes[i] for i in range(n) if batch_tokens[i] != config.pad_token_id)) != alive:
            raise RuntimeError("V4 lane branch state aliases before model forward")
        outputs = model(
            input_ids=torch.tensor(batch_tokens, device=model.device, dtype=torch.long).view(-1, 1),
            position_ids=torch.full((n, 1), pos, device=model.device),
            past_key_values=cache, return_dict=True, use_cache=True,
        )
        if config.calibration_assertions and int(outputs.logits.shape[0]) != n:
            raise RuntimeError("V4 model forward returned wrong lane count")
        state["model_forwards"] += 1
        state["tokens_advanced"] += alive
        descendants = reference_turbo_dfs(
            model, logits=outputs.logits[:, -1], max_new_tokens=max_new_tokens - 1,
            scores=batch_scores, pos=pos + 1, cache=outputs.past_key_values,
            config=config, started_unix=started_unix, state=state,
            parent_nodes=next_parent_nodes, prefixes=next_prefixes,
        )
        for lane, beams in enumerate(descendants):
            for candidate in beams:
                # Public code mutates the returned suffix by inserting this branch token.
                suffixes[lane].append(ReferenceTurboDFSCandidate(
                    candidate_id=candidate.candidate_id,
                    token_ids=(batch_tokens[lane],) + candidate.token_ids,
                    cumulative_nll=candidate.cumulative_nll,
                    terminal_node_id=candidate.terminal_node_id,
                    discovery_forward_index=candidate.discovery_forward_index,
                    discovery_unix=candidate.discovery_unix,
                ))
        del outputs
    return suffixes


def inference_reference_turbo_dfs(
    model: Any, *, input_ids: Any, config: ReferenceTurboDFSConfig,
) -> ReferenceTurboDFSResult:
    """Direct equivalent of public ``inference_turbo_dfs`` plus trace capture."""
    import torch

    if input_ids.ndim != 2 or int(input_ids.shape[0]) < 1:
        raise ValueError("input_ids must be a non-empty [lanes, tokens] tensor")
    n = int(input_ids.shape[0])
    state: dict[str, Any] = {
        "nodes": [], "branch_probabilities": [], "model_forwards": 0,
        "tokens_advanced": 0, "completed_candidates": 0, "next_candidate_id": 0,
        "next_node_id": 0, "max_frontier_size": 0,
    }

    def new_node(**payload: Any) -> int:
        node_id = int(state["next_node_id"])
        state["next_node_id"] += 1
        state["nodes"].append({"node_id": node_id, **payload})
        return node_id

    state["new_node"] = new_node
    started = time.time()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, return_dict=True, use_cache=True)
        if config.calibration_assertions and int(outputs.logits.shape[0]) != n:
            raise RuntimeError("V4 root forward returned wrong lane count")
        state["model_forwards"] += 1
        roots = [new_node(parent_node_id=None, lane=lane, token_position=int(input_ids.size(1)),
                          branch_depth=0, selected_token=None, token_logprob=None,
                          cumulative_score=0.0, state="root", prune_reason=None,
                          termination_reason=None, branch_rank=None) for lane in range(n)]
        raw = reference_turbo_dfs(
            model, logits=outputs.logits[:, -1], max_new_tokens=config.max_new_tokens,
            scores=[0.0] * n, pos=int(input_ids.size(1)), cache=outputs.past_key_values,
            config=config, started_unix=started, state=state, parent_nodes=roots,
            prefixes=[tuple() for _ in range(n)],
        )
        del outputs
    timed_out = (time.time() - started >= config.local_time_limit_seconds or
                 (config.absolute_end_time_unix is not None and time.time() >= config.absolute_end_time_unix))
    # Public inference_turbo_dfs ranks beams by score only.
    ranked = tuple(tuple(sorted(beams, key=lambda candidate: candidate.cumulative_nll)) for beams in raw)
    termination = "wall_time" if timed_out else "search_exhausted"
    return ReferenceTurboDFSResult(
        candidates=ranked, nodes=tuple(state["nodes"]),
        branch_probabilities=tuple(state["branch_probabilities"]),
        model_forwards=int(state["model_forwards"]), tokens_advanced=int(state["tokens_advanced"]),
        completed_candidates=int(state["completed_candidates"]),
        max_frontier_size=int(state["max_frontier_size"]), timed_out=timed_out,
        termination_reason=termination,
    )
