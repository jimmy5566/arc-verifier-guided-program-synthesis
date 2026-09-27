"""Traceable, batched TurboDFS derived from the public NVARC notebook.

The algorithmic core follows ``ARC-AGI1/002_ivan_arc1.ipynb`` at NVARC
commit ``846d0198efa752534594e321fc3289fc0a06c657``: evaluate the native
ARC vocabulary, retain continuations below a cumulative-NLL threshold, then
advance one live branch per batch lane with its ``past_key_values``.  The
explicit hard caps in :class:`TurboDFSOptConfig` are experiment safety
semantics, not claims about the public notebook.

This module intentionally has no ARC target access.  It records compact,
JSON-serialisable search provenance so a later post-freeze analysis can
reconstruct the explored tree without rerunning model inference.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from time import monotonic
from typing import Any


PUBLIC_NVARC_COMMIT = "846d0198efa752534594e321fc3289fc0a06c657"
PUBLIC_NVARC_NOTEBOOK = "ARC-AGI1/002_ivan_arc1.ipynb"
PUBLIC_NVARC_NOTEBOOK_SHA256 = "24e57f6ae39e63d0041641ae2302a40f7afb9c2973420fb8db05054028537e52"


@dataclass(frozen=True)
class TurboDFSOptConfig:
    """Frozen decoder contract for one target-blind run.

    ``max_batch_forward_passes`` and ``max_complete_candidates_per_prompt``
    deliberately differ from the public notebook's time-only stopping rule;
    they are explicit bounded-search semantics used to protect an overnight
    experiment.  Any change must use a new decoder-config hash.
    """

    max_new_tokens: int
    max_cumulative_nll: float
    max_wall_seconds: float
    # ``None`` means use the wall-clock stopping rule alone, matching the
    # public notebook's decoder loop.  V1 supplied an integer safety cap.
    max_batch_forward_passes: int | None
    # ``None`` exactly matches the public notebook's unbounded completed-suffix
    # retention.  Integer values preserve the prior V1/V2 bounded experiment.
    max_complete_candidates_per_prompt: int | None
    top_k_trace: int = 8
    capture_full_arc_distribution: bool = True
    pad_token_id: int = 13
    branch_ordering: str = "cumulative_nll_ascending_then_token_id"

    def __post_init__(self) -> None:
        if self.max_new_tokens < 2:
            raise ValueError("max_new_tokens must be at least two")
        if self.max_cumulative_nll <= 0 or self.max_wall_seconds <= 0:
            raise ValueError("NLL and wall limits must be positive")
        if self.max_batch_forward_passes is not None and self.max_batch_forward_passes < 1:
            raise ValueError("batch-forward cap must be positive when supplied")
        if (
            self.max_complete_candidates_per_prompt is not None
            and self.max_complete_candidates_per_prompt < 1
        ):
            raise ValueError("candidate cap must be positive")
        if self.top_k_trace < 8:
            raise ValueError("top_k_trace must be at least eight")
        if self.branch_ordering != "cumulative_nll_ascending_then_token_id":
            raise ValueError("unsupported frozen branch ordering")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TurboDFSOptCandidate:
    candidate_id: int
    token_ids: tuple[int, ...]
    cumulative_nll: float
    terminal_node_id: int


@dataclass(frozen=True)
class TurboDFSOptResult:
    candidates: tuple[tuple[TurboDFSOptCandidate, ...], ...]
    nodes: tuple[dict[str, Any], ...]
    branch_probabilities: tuple[dict[str, Any], ...]
    batch_forward_passes: int
    tokens_advanced: int
    probability_pruned: int
    complete_candidates: int
    timed_out: bool
    branch_cap_reached: bool
    candidate_cap_reached: bool
    termination_reason: str


def allowed_arc_tokens(*, eos_token_id: int) -> tuple[int, ...]:
    """The notebook's digits/newline vocabulary plus native EOS."""
    return tuple(range(11)) + (int(eos_token_id),)


def _distribution(
    log_probs: Any,
    allowed: tuple[int, ...],
    config: TurboDFSOptConfig,
) -> dict[str, Any]:
    """Retain exactly the compact distribution used at a branch point."""
    values = [(token, float(log_probs[token].item())) for token in allowed]
    values.sort(key=lambda item: (-item[1], item[0]))
    payload: dict[str, Any] = {
        "normalization": "log_softmax_over_model_vocab_float32_cpu",
        "top_k": [{"token_id": token, "logprob": score} for token, score in values[: config.top_k_trace]],
    }
    if config.capture_full_arc_distribution:
        payload["full_arc_logprobs"] = [{"token_id": token, "logprob": score} for token, score in values]
    return payload


def turbodfs_opt(
    model: Any,
    *,
    input_ids: Any,
    eos_token_id: int,
    config: TurboDFSOptConfig,
) -> TurboDFSOptResult:
    """Run the public-style batched DFS with immutable, lossless provenance.

    Input rows must have equal prompt lengths, as in the public notebook's
    ``torch.tensor(prefix_tokens)`` construction.  A caller can partition
    heterogeneous prompts by length without changing per-prompt semantics.
    """
    import torch

    if input_ids.ndim != 2 or input_ids.shape[0] < 1:
        raise ValueError("input_ids must be a non-empty [batch, tokens] tensor")
    started = monotonic()
    allowed = allowed_arc_tokens(eos_token_id=eos_token_id)
    batch_size = int(input_ids.shape[0])
    device = input_ids.device
    batch_forward_passes = 0
    tokens_advanced = 0
    probability_pruned = 0
    completed = 0
    timed_out = False
    branch_cap_reached = False
    candidate_cap_reached = False
    next_node_id = 0
    next_candidate_id = 0
    nodes: list[dict[str, Any]] = []
    probabilities: list[dict[str, Any]] = []

    def new_node(**payload: Any) -> int:
        nonlocal next_node_id
        node_id = next_node_id
        next_node_id += 1
        nodes.append({"node_id": node_id, **payload})
        return node_id

    def expired() -> bool:
        return monotonic() - started >= config.max_wall_seconds

    def recurse(
        logits: Any,
        remaining_tokens: int,
        scores: list[float],
        position: int,
        cache: Any,
        parent_node_ids: list[int | None],
        suffix_prefixes: list[tuple[int, ...]],
    ) -> list[list[TurboDFSOptCandidate]]:
        nonlocal batch_forward_passes, tokens_advanced, probability_pruned
        nonlocal completed, timed_out, branch_cap_reached, candidate_cap_reached, next_candidate_id
        lanes = int(logits.shape[0])
        log_probs = logits.float().cpu().log_softmax(-1)
        suffixes: list[list[TurboDFSOptCandidate]] = [[] for _ in range(lanes)]
        candidates: list[list[tuple[float, int, int, tuple[int, ...]]]] = [[] for _ in range(lanes)]

        for lane in range(lanes):
            distribution = _distribution(log_probs[lane], allowed, config)
            probabilities.append(
                {
                    "lane": lane,
                    "parent_node_id": parent_node_ids[lane],
                    "branch_depth": len(suffix_prefixes[lane]),
                    "generated_position": position,
                    "cumulative_score_before": scores[lane],
                    **distribution,
                }
            )
            permitted: list[tuple[float, int]] = []
            for token in allowed:
                next_score = float(scores[lane] - log_probs[lane, token].item())
                if next_score >= config.max_cumulative_nll:
                    probability_pruned += 1
                    new_node(
                        parent_node_id=parent_node_ids[lane], lane=lane,
                        branch_depth=len(suffix_prefixes[lane]) + 1,
                        generated_position=position, token_id=int(token),
                        token_logprob=float(log_probs[lane, token].item()),
                        cumulative_score=next_score, branch_rank_at_parent=None,
                        state="pruned", prune_reason="cumulative_nll_threshold",
                        termination_reason=None, candidate_id=None,
                    )
                    continue
                permitted.append((next_score, int(token)))
            permitted.sort(key=lambda item: (item[0], item[1]))
            for rank, (next_score, token) in enumerate(permitted, 1):
                token_logprob = float(log_probs[lane, token].item())
                suffix = suffix_prefixes[lane] + (token,)
                if token == eos_token_id:
                    node_id = new_node(
                        parent_node_id=parent_node_ids[lane], lane=lane,
                        branch_depth=len(suffix), generated_position=position,
                        token_id=token, token_logprob=token_logprob,
                        cumulative_score=next_score, branch_rank_at_parent=rank,
                        state="completed", prune_reason=None,
                        termination_reason="eos", candidate_id=next_candidate_id,
                    )
                    if (
                        config.max_complete_candidates_per_prompt is None
                        or len(suffixes[lane]) < config.max_complete_candidates_per_prompt
                    ):
                        suffixes[lane].append(TurboDFSOptCandidate(next_candidate_id, suffix, next_score, node_id))
                        next_candidate_id += 1
                        completed += 1
                    else:
                        candidate_cap_reached = True
                    continue
                if remaining_tokens <= 1:
                    new_node(
                        parent_node_id=parent_node_ids[lane], lane=lane,
                        branch_depth=len(suffix), generated_position=position,
                        token_id=token, token_logprob=token_logprob,
                        cumulative_score=next_score, branch_rank_at_parent=rank,
                        state="pruned", prune_reason="max_new_tokens",
                        termination_reason="max_new_tokens", candidate_id=None,
                    )
                    continue
                candidates[lane].append((next_score, token, rank, suffix))

        while not expired() and not branch_cap_reached:
            batch_tokens: list[int] = []
            batch_scores: list[float] = []
            child_parent_ids: list[int | None] = []
            child_prefixes: list[tuple[int, ...]] = []
            alive = 0
            for lane in range(lanes):
                if not candidates[lane]:
                    batch_tokens.append(config.pad_token_id)
                    batch_scores.append(1000.0)
                    child_parent_ids.append(parent_node_ids[lane])
                    child_prefixes.append(suffix_prefixes[lane])
                    continue
                score, token, rank, suffix = candidates[lane].pop(0)
                node_id = new_node(
                    parent_node_id=parent_node_ids[lane], lane=lane,
                    branch_depth=len(suffix), generated_position=position,
                    token_id=token,
                    token_logprob=-(score - scores[lane]),
                    cumulative_score=score, branch_rank_at_parent=rank,
                    state="expanded", prune_reason=None,
                    termination_reason=None, candidate_id=None,
                )
                batch_tokens.append(token)
                batch_scores.append(score)
                child_parent_ids.append(node_id)
                child_prefixes.append(suffix)
                alive += 1
            if not alive:
                break
            if (
                config.max_batch_forward_passes is not None
                and batch_forward_passes >= config.max_batch_forward_passes
            ):
                branch_cap_reached = True
                break
            outputs = model(
                input_ids=torch.tensor(batch_tokens, device=device, dtype=torch.long).view(-1, 1),
                position_ids=torch.full((lanes, 1), position, device=device),
                past_key_values=cache,
                return_dict=True,
                use_cache=True,
            )
            batch_forward_passes += 1
            tokens_advanced += alive
            descendants = recurse(
                outputs.logits[:, -1], remaining_tokens - 1, batch_scores, position + 1,
                outputs.past_key_values, child_parent_ids, child_prefixes,
            )
            for lane, lane_candidates in enumerate(descendants):
                if not lane_candidates:
                    continue
                if config.max_complete_candidates_per_prompt is None:
                    suffixes[lane].extend(lane_candidates)
                else:
                    capacity = config.max_complete_candidates_per_prompt - len(suffixes[lane])
                    if capacity <= 0:
                        candidate_cap_reached = True
                        continue
                    suffixes[lane].extend(lane_candidates[:capacity])
                    if len(lane_candidates) > capacity:
                        candidate_cap_reached = True
            del outputs
        if expired():
            timed_out = True
        return suffixes

    with torch.inference_mode():
        root = model(input_ids=input_ids, return_dict=True, use_cache=True)
        batch_forward_passes += 1
        root_ids = [new_node(parent_node_id=None, lane=lane, branch_depth=0, generated_position=int(input_ids.shape[1]), token_id=None, token_logprob=None, cumulative_score=0.0, branch_rank_at_parent=None, state="root", prune_reason=None, termination_reason=None, candidate_id=None) for lane in range(batch_size)]
        raw = recurse(
            root.logits[:, -1], config.max_new_tokens, [0.0] * batch_size,
            int(input_ids.shape[1]), root.past_key_values, root_ids,
            [tuple() for _ in range(batch_size)],
        )
        del root
    ranked = tuple(tuple(sorted(lane, key=lambda item: (item.cumulative_nll, item.token_ids))) for lane in raw)
    if timed_out:
        reason = "wall_time"
    elif branch_cap_reached:
        reason = "max_batch_forward_passes"
    elif candidate_cap_reached:
        reason = "max_complete_candidates_per_prompt"
    else:
        reason = "search_exhausted"
    return TurboDFSOptResult(
        candidates=ranked, nodes=tuple(nodes), branch_probabilities=tuple(probabilities),
        batch_forward_passes=batch_forward_passes, tokens_advanced=tokens_advanced,
        probability_pruned=probability_pruned, complete_candidates=sum(len(lane) for lane in ranked),
        timed_out=timed_out, branch_cap_reached=branch_cap_reached,
        candidate_cap_reached=candidate_cap_reached, termination_reason=reason,
    )
