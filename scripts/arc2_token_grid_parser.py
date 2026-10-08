"""Target-blind, token-level ARC grid extraction for baseline evaluation."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

from src.inference.arc_native_io import ARCNativeOutputParser


@dataclass(frozen=True)
class TokenGridContract:
    digit_token_ids: tuple[int, ...]
    newline_token_id: int
    eos_token_id: int
    pad_token_id: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "digit_token_ids": list(self.digit_token_ids),
            "newline_token_id": self.newline_token_id,
            "eos_token_id": self.eos_token_id,
            "pad_token_id": self.pad_token_id,
        }


@dataclass(frozen=True)
class TokenParseResult:
    grid: list[list[int]] | None
    generated_token_ids: list[int]
    generated_length: int
    termination_status: str
    eos_observed: bool
    trailing_pad_count: int
    content_token_ids: list[int]
    parse_reason: str | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def tokenizer_token_contract(tokenizer: Any, *, eos_token_id: int, pad_token_id: int) -> TokenGridContract:
    """Derive and verify grid token IDs from the frozen runtime tokenizer."""
    if int(tokenizer.eos_token_id) != int(eos_token_id):
        raise RuntimeError("TOKENIZER_EOS_ID_MISMATCH")
    if int(tokenizer.pad_token_id) != int(pad_token_id):
        raise RuntimeError("TOKENIZER_PAD_ID_MISMATCH")
    digits: list[int] = []
    for digit in "0123456789":
        ids = list(tokenizer.encode(digit, add_special_tokens=False))
        if len(ids) != 1 or tokenizer.decode(ids, skip_special_tokens=False) != digit:
            raise RuntimeError(f"TOKENIZER_DIGIT_TOKEN_MISMATCH:{digit}")
        digits.append(int(ids[0]))
    newline_ids = list(tokenizer.encode("\n", add_special_tokens=False))
    if len(newline_ids) != 1 or tokenizer.decode(newline_ids, skip_special_tokens=False) != "\n":
        raise RuntimeError("TOKENIZER_NEWLINE_TOKEN_MISMATCH")
    contract = TokenGridContract(tuple(digits), int(newline_ids[0]), int(eos_token_id), int(pad_token_id))
    all_ids = [*contract.digit_token_ids, contract.newline_token_id, contract.eos_token_id, contract.pad_token_id]
    if len(set(all_ids)) != len(all_ids):
        raise RuntimeError("TOKENIZER_GRID_SPECIAL_TOKEN_COLLISION")
    return contract


def parse_generated_token_ids(token_ids: Iterable[int], contract: TokenGridContract) -> TokenParseResult:
    """Accept exactly a digit/newline grid plus optional terminal EOS and pads."""
    original = [int(value) for value in token_ids]
    content = list(original)
    trailing_pad_count = 0
    while content and content[-1] == contract.pad_token_id:
        content.pop()
        trailing_pad_count += 1
    eos_positions = [index for index, value in enumerate(content) if value == contract.eos_token_id]
    if len(eos_positions) > 1:
        return TokenParseResult(None, original, len(original), "INVALID_MULTIPLE_EOS", True, trailing_pad_count, content, "MULTIPLE_EOS")
    if eos_positions:
        if eos_positions[0] != len(content) - 1:
            return TokenParseResult(None, original, len(original), "INVALID_NONTERMINAL_EOS", True, trailing_pad_count, content, "NONTERMINAL_EOS")
        content.pop()
        termination = "EOS"
    else:
        termination = "MAX_LENGTH_OR_UNTERMINATED"
    allowed = set(contract.digit_token_ids) | {contract.newline_token_id}
    if not content:
        return TokenParseResult(None, original, len(original), termination, bool(eos_positions), trailing_pad_count, content, "EMPTY_CONTENT")
    if any(value not in allowed for value in content):
        return TokenParseResult(None, original, len(original), termination, bool(eos_positions), trailing_pad_count, content, "DISALLOWED_TOKEN")
    mapping = {token_id: str(index) for index, token_id in enumerate(contract.digit_token_ids)}
    mapping[contract.newline_token_id] = "\n"
    grid = ARCNativeOutputParser.parse("".join(mapping[value] for value in content))
    return TokenParseResult(grid, original, len(original), termination, bool(eos_positions), trailing_pad_count, content, None if grid is not None else "INVALID_ARC_GRID")
