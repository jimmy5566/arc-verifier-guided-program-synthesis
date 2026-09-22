"""Small grammar-constrained native-token DFS for local ARC ablations.

This module deliberately knows only the 16-token NVARC transport grammar.  It
does not inspect ARC targets, infer rules, or rank grids.  The caller supplies
an already rendered native prompt and may later parse/rank returned text using
the existing V8 components.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import inf
from time import monotonic
from typing import Any


_DIGITS = tuple(range(10))
_NEWLINE = 10
_EOS = 15


@dataclass(frozen=True)
class GridPrefix:
    """Sufficient state to reject continuations that cannot be an ARC grid."""

    completed_rows: int = 0
    width: int | None = None
    current_width: int = 0

    def allowed_tokens(self) -> tuple[int, ...]:
        if self.completed_rows >= 30:
            return (_EOS,) if self.current_width == 0 else ()
        allowed: list[int] = []
        if self.current_width < 30:
            allowed.extend(_DIGITS)
        if self.current_width:
            if self.width is None or self.current_width == self.width:
                allowed.append(_NEWLINE)
                allowed.append(_EOS)
        elif self.completed_rows:
            allowed.append(_EOS)
        return tuple(allowed)

    def consume(self, token_id: int) -> "GridPrefix | None":
        if token_id not in self.allowed_tokens():
            return None
        if token_id in _DIGITS:
            return GridPrefix(self.completed_rows, self.width, self.current_width + 1)
        if token_id == _NEWLINE:
            return GridPrefix(self.completed_rows + 1, self.width or self.current_width, 0)
        if token_id == _EOS:
            return self if self.terminal else None
        return None

    @property
    def terminal(self) -> bool:
        if self.current_width:
            return self.width is None or self.current_width == self.width
        return self.completed_rows > 0 and self.width is not None


@dataclass(frozen=True)
class ConstrainedDFSConfig:
    max_candidates: int = 4
    max_new_tokens: int = 1024
    max_cumulative_nll: float = 1.6094379124341003  # -log(0.2), NVARC-style conservative gate
    max_branch_tokens: int = 4
    max_wall_seconds: float = 90.0
    context_window: int = 16384

    def __post_init__(self) -> None:
        if not 1 <= self.max_candidates <= 8:
            raise ValueError("max_candidates must be in 1..8")
        if not 1 <= self.max_new_tokens <= 1024:
            raise ValueError("max_new_tokens must be in 1..1024")
        if not 1 <= self.max_branch_tokens <= 16:
            raise ValueError("max_branch_tokens must be in 1..16")
        if self.max_cumulative_nll <= 0 or self.max_wall_seconds <= 0 or self.context_window < 1:
            raise ValueError("DFS thresholds must be positive")


@dataclass(frozen=True)
class DFSCandidate:
    token_ids: tuple[int, ...]
    cumulative_nll: float
    elapsed_seconds: float


@dataclass(frozen=True)
class DFSResult:
    candidates: tuple[DFSCandidate, ...]
    expanded_nodes: int
    pruned_probability: int
    pruned_grammar: int
    timed_out: bool


def _allowed_ranked_tokens(logits: Any, prefix: GridPrefix, cumulative_nll: float, config: ConstrainedDFSConfig) -> tuple[list[tuple[float, int]], int]:
    """Score only grammar-legal native tokens and apply the frozen NLL bound."""
    import torch

    log_probs = torch.log_softmax(logits.float(), dim=-1)
    options: list[tuple[float, int]] = []
    rejected = 0
    for token_id in prefix.allowed_tokens():
        score = float(cumulative_nll - log_probs[int(token_id)].item())
        if score <= config.max_cumulative_nll:
            options.append((score, int(token_id)))
        else:
            rejected += 1
    return sorted(options, key=lambda item: (item[0], item[1]))[: config.max_branch_tokens], rejected


def constrained_token_dfs(provider: Any, messages: list[dict[str, str]], config: ConstrainedDFSConfig) -> DFSResult:
    """Enumerate up to K legal native-grid continuations by depth-first NLL search.

    ``provider`` is the existing :class:`NVARCNativeProvider`; no model/tokenizer
    configuration is created or modified here.  We reuse KV caches on every
    branch exactly as the public NVARC DFS does, but prune all impossible grid
    prefixes before scheduling another model forward pass.
    """
    provider.load()
    import torch

    model, tokenizer = provider.model, provider.tokenizer
    if model is None or tokenizer is None:
        raise RuntimeError("native provider did not load model/tokenizer")
    encoded = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_tensors="pt", return_dict=True)
    prompt_tokens = int(encoded["input_ids"].shape[-1])
    if prompt_tokens > config.context_window:
        raise ValueError(f"native prompt has {prompt_tokens} tokens, exceeds DFS context {config.context_window}")
    device = provider.device
    started = monotonic()
    terminal: list[DFSCandidate] = []
    expanded = probability_pruned = grammar_pruned = 0
    timed_out = False

    with torch.inference_mode():
        initial = {name: value.to(device) for name, value in encoded.items()}
        output = model(**initial, return_dict=True, use_cache=True)
        first_logits = output.logits[:, -1, :][0]
        first_cache = output.past_key_values
        del initial, output

        def visit(logits: Any, cache: Any, position: int, prefix: GridPrefix, tokens: tuple[int, ...], nll: float) -> None:
            nonlocal expanded, probability_pruned, grammar_pruned, timed_out
            if len(terminal) >= config.max_candidates:
                return
            if monotonic() - started >= config.max_wall_seconds:
                timed_out = True
                return
            if len(tokens) >= config.max_new_tokens:
                return
            options, rejected = _allowed_ranked_tokens(logits, prefix, nll, config)
            probability_pruned += rejected
            if not options:
                grammar_pruned += 1
                return
            for next_nll, token_id in options:
                if len(terminal) >= config.max_candidates or timed_out:
                    return
                next_prefix = prefix.consume(token_id)
                if next_prefix is None:
                    grammar_pruned += 1
                    continue
                next_tokens = tokens + (token_id,)
                if token_id == _EOS:
                    terminal.append(DFSCandidate(next_tokens, next_nll, monotonic() - started))
                    continue
                expanded += 1
                next_input = torch.tensor([[token_id]], device=device, dtype=torch.long)
                next_position = torch.full((1, 1), position, device=device, dtype=torch.long)
                child = model(input_ids=next_input, position_ids=next_position, past_key_values=cache, return_dict=True, use_cache=True)
                visit(child.logits[:, -1, :][0], child.past_key_values, position + 1, next_prefix, next_tokens, next_nll)
                del child, next_input, next_position

        visit(first_logits, first_cache, prompt_tokens, GridPrefix(), (), 0.0)
        del first_logits, first_cache
    return DFSResult(tuple(terminal), expanded, probability_pruned, grammar_pruned, timed_out)


def decode_dfs_candidate(provider: Any, candidate: DFSCandidate) -> str:
    """Decode the exact generated token sequence; special terminator is hidden."""
    if provider.tokenizer is None:
        raise RuntimeError("provider tokenizer is not loaded")
    return str(provider.tokenizer.decode(list(candidate.token_ids), skip_special_tokens=True))
