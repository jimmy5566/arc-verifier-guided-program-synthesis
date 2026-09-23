"""Bounded reproduction of the public NVARC TurboDFS decoder.

The public decoder expands every ARC-vocabulary token whose *cumulative* NLL
is below ``-log(0.2)`` and reuses ``past_key_values`` down each depth-first
branch.  This implementation intentionally keeps those semantics.  The only
additions are explicit experiment safety caps: wall time, expanded branches,
complete suffixes, and generated tokens.  It does not inspect ARC targets or
apply semantic/grid-prefix pruning; parsing remains a separate transport step.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import monotonic
from typing import Any


@dataclass(frozen=True)
class TurboDFSConfig:
    max_new_tokens: int
    max_cumulative_nll: float
    max_wall_seconds: float
    max_expanded_branches: int
    max_complete_candidates: int

    def __post_init__(self) -> None:
        if self.max_new_tokens < 2:
            raise ValueError("max_new_tokens must be at least two")
        if self.max_cumulative_nll <= 0 or self.max_wall_seconds <= 0:
            raise ValueError("NLL and wall limits must be positive")
        if self.max_expanded_branches < 1 or self.max_complete_candidates < 1:
            raise ValueError("branch and candidate limits must be positive")


@dataclass(frozen=True)
class TurboDFSCandidate:
    token_ids: tuple[int, ...]
    cumulative_nll: float


@dataclass(frozen=True)
class TurboDFSResult:
    candidates: tuple[TurboDFSCandidate, ...]
    expanded_branches: int
    probability_pruned: int
    complete_candidates: int
    generated_tokens: int
    timed_out: bool
    branch_cap_reached: bool
    candidate_cap_reached: bool


def allowed_arc_tokens(*, eos_token_id: int) -> tuple[int, ...]:
    """Official ARC vocabulary: digits, newline, then the native EOS token."""
    return tuple(range(11)) + (int(eos_token_id),)


def turbo_dfs(
    model: Any,
    *,
    input_ids: Any,
    attention_mask: Any | None,
    eos_token_id: int,
    config: TurboDFSConfig,
) -> TurboDFSResult:
    """Decode one native prompt using score-bounded public-style DFS.

    The exact public notebook batches independent prompts.  Here a single
    prompt is intentionally isolated so its experiment-level hard caps cannot
    be exceeded by another view.  Within the prompt, every child forward pass
    receives its parent KV cache exactly as the reference implementation does.
    """
    import torch

    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("TurboDFS smoke accepts one prompt per invocation")
    started = monotonic()
    tokens = allowed_arc_tokens(eos_token_id=eos_token_id)
    completed: list[TurboDFSCandidate] = []
    expanded = probability_pruned = generated_tokens = 0
    timed_out = branch_cap_reached = candidate_cap_reached = False
    position = int(input_ids.shape[1])
    device = input_ids.device

    def expired() -> bool:
        return monotonic() - started >= config.max_wall_seconds

    with torch.inference_mode():
        kwargs: dict[str, Any] = {"input_ids": input_ids, "return_dict": True, "use_cache": True}
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
        root = model(**kwargs)

        def visit(logits: Any, cache: Any, pos: int, suffix: tuple[int, ...], score: float) -> None:
            nonlocal expanded, probability_pruned, generated_tokens, timed_out, branch_cap_reached, candidate_cap_reached
            if len(completed) >= config.max_complete_candidates:
                candidate_cap_reached = True
                return
            if expired():
                timed_out = True
                return
            if len(suffix) >= config.max_new_tokens:
                return
            nll = score - torch.log_softmax(logits.float().cpu(), dim=-1)
            choices: list[tuple[float, int]] = []
            for token_id in tokens:
                next_score = float(nll[int(token_id)].item())
                if next_score < config.max_cumulative_nll:
                    choices.append((next_score, int(token_id)))
                else:
                    probability_pruned += 1
            for next_score, token_id in sorted(choices, key=lambda item: (item[0], item[1])):
                if len(completed) >= config.max_complete_candidates:
                    candidate_cap_reached = True
                    return
                if expired():
                    timed_out = True
                    return
                next_suffix = suffix + (token_id,)
                generated_tokens += 1
                if token_id == eos_token_id:
                    completed.append(TurboDFSCandidate(next_suffix, next_score))
                    continue
                if expanded >= config.max_expanded_branches:
                    branch_cap_reached = True
                    return
                expanded += 1
                child = model(
                    input_ids=torch.tensor([[token_id]], dtype=torch.long, device=device),
                    position_ids=torch.full((1, 1), pos, dtype=torch.long, device=device),
                    past_key_values=cache,
                    return_dict=True,
                    use_cache=True,
                )
                visit(child.logits[0, -1], child.past_key_values, pos + 1, next_suffix, next_score)
                del child

        visit(root.logits[0, -1], root.past_key_values, position, (), 0.0)
        del root
    ranked = tuple(sorted(completed, key=lambda item: (item.cumulative_nll, item.token_ids)))
    return TurboDFSResult(
        candidates=ranked,
        expanded_branches=expanded,
        probability_pruned=probability_pruned,
        complete_candidates=len(ranked),
        generated_tokens=generated_tokens,
        timed_out=timed_out,
        branch_cap_reached=branch_cap_reached,
        candidate_cap_reached=candidate_cap_reached,
    )
