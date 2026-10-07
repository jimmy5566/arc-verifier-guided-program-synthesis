"""Pure contracts for the generation capability audit.

The evaluation target is the final assistant turn in each already-frozen
sentinel episode.  Earlier assistant turns remain as demonstrations; the final
assistant payload and every later token are excluded from the model input.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
from typing import Any, Iterable, Sequence


SOURCE_COMMIT = "18c2034799085de7d551f39571470f741baa17f9"
MODEL_STATES = ("base", "tokens_500000", "tokens_1000000", "tokens_1500000", "tokens_2000000")
MAX_NEW_TOKENS = 932
CONTEXT_WINDOW = 8_704
ASSISTANT_PREFIX = (14, 10)
EOS_TOKEN_ID = 15
PAD_TOKEN_ID = 13
ALLOWED_GRID_TOKEN_IDS = frozenset(range(11))  # colors 0..9 plus newline 10
MAX_GRID_DIMENSION = 256  # Includes the frozen 1D-ARC long-row representation.


class GenerationAuditError(RuntimeError):
    """Raised when a frozen generation-evaluation invariant fails."""


@dataclass(frozen=True)
class GenerationExample:
    sample_id: str
    source: str
    family: str
    prompt_ids: tuple[int, ...]
    gold_ids: tuple[int, ...]
    gold_grid: tuple[tuple[int, ...], ...]
    target_span_start: int
    target_span_end: int
    original_sequence_length: int


@dataclass(frozen=True)
class ParsedGeneration:
    classification: str
    grid: tuple[tuple[int, ...], ...] | None
    terminated_by_eos: bool


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _supervised_spans(labels: Sequence[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(labels):
        supervised = int(value) != -100
        if supervised and start is None:
            start = index
        if start is not None and (not supervised or index == len(labels) - 1):
            end = index if supervised and index == len(labels) - 1 else index - 1
            spans.append((start, end))
            start = None
    return spans


def grid_from_native_tokens(tokens: Sequence[int]) -> tuple[tuple[int, ...], ...]:
    values = tuple(int(value) for value in tokens)
    if not values or any(value not in ALLOWED_GRID_TOKEN_IDS for value in values):
        raise GenerationAuditError("GOLD_TARGET_NOT_NATIVE_GRID_TOKENS")
    rows: list[list[int]] = [[]]
    for value in values:
        if value == 10:
            if not rows[-1]:
                raise GenerationAuditError("GOLD_TARGET_EMPTY_ROW")
            rows.append([])
        else:
            rows[-1].append(value)
    if not rows[-1] or not 1 <= len(rows) <= MAX_GRID_DIMENSION:
        raise GenerationAuditError("GOLD_TARGET_INVALID_HEIGHT")
    width = len(rows[0])
    if not 1 <= width <= MAX_GRID_DIMENSION or any(len(row) != width for row in rows):
        raise GenerationAuditError("GOLD_TARGET_INVALID_WIDTH")
    return tuple(tuple(row) for row in rows)


def build_generation_example(row: dict[str, Any], metadata: dict[str, Any]) -> GenerationExample:
    ids = tuple(int(value) for value in row["input_ids"])
    labels = tuple(int(value) for value in row["labels"])
    if not ids or len(ids) != len(labels):
        raise GenerationAuditError("TOKEN_LABEL_LENGTH_MISMATCH")
    spans = _supervised_spans(labels)
    if not spans:
        raise GenerationAuditError("NO_ASSISTANT_TARGET")
    start, end = spans[-1]
    if end != len(ids) - 1:
        raise GenerationAuditError("LATER_TOKENS_AFTER_TARGET_ASSISTANT")
    if tuple(ids[start:start + len(ASSISTANT_PREFIX)]) != ASSISTANT_PREFIX:
        raise GenerationAuditError("TARGET_ASSISTANT_PREFIX_MISMATCH")
    content_start = start + len(ASSISTANT_PREFIX)
    if ids[end] != EOS_TOKEN_ID or labels[end] != EOS_TOKEN_ID:
        raise GenerationAuditError("TARGET_ASSISTANT_EOS_MISSING")
    if tuple(labels[start:end + 1]) != tuple(ids[start:end + 1]):
        raise GenerationAuditError("TARGET_ASSISTANT_LABEL_MISMATCH")
    prompt = ids[:content_start]
    gold = ids[content_start:end]
    if not prompt or prompt[-len(ASSISTANT_PREFIX):] != ASSISTANT_PREFIX:
        raise GenerationAuditError("GENERATION_PROMPT_BOUNDARY_MISMATCH")
    return GenerationExample(
        sample_id=str(metadata["sample_id"]),
        source=str(metadata["source"]),
        family=str(metadata["family"]),
        prompt_ids=prompt,
        gold_ids=gold,
        gold_grid=grid_from_native_tokens(gold),
        target_span_start=content_start,
        target_span_end=end,
        original_sequence_length=len(ids),
    )


def context_audit_record(example: GenerationExample, row: dict[str, Any]) -> dict[str, Any]:
    ids = tuple(int(value) for value in row["input_ids"])
    labels = tuple(int(value) for value in row["labels"])
    start, end = example.target_span_start, example.target_span_end
    return {
        "sample_id": example.sample_id,
        "source": example.source,
        "family": example.family,
        "prompt_length": len(example.prompt_ids),
        "gold_length_without_eos": len(example.gold_ids),
        "original_sequence_length": len(ids),
        "prompt_sha256": canonical_sha256(example.prompt_ids),
        "gold_sha256": canonical_sha256(example.gold_ids),
        "prompt_is_exact_source_prefix": tuple(ids[:start]) == example.prompt_ids,
        "target_gold_is_exact_disjoint_source_suffix": tuple(ids[start:end]) == example.gold_ids,
        "target_eos_at_boundary": ids[end] == EOS_TOKEN_ID,
        "no_later_tokens": end == len(ids) - 1,
        "target_labels_never_passed_to_generation": True,
        "gold_target_tokens_in_inference_input_by_source_position": False,
        "prior_demo_assistant_turns_retained": len(_supervised_spans(labels)) - 1,
    }


def parse_generated_tokens(
    token_ids: Sequence[int], gold_grid: Sequence[Sequence[int]], *, hit_max_new_tokens: bool
) -> ParsedGeneration:
    raw = [int(value) for value in token_ids]
    eos_positions = [index for index, value in enumerate(raw) if value == EOS_TOKEN_ID]
    terminated = bool(eos_positions)
    if terminated:
        first_eos = eos_positions[0]
        if first_eos != len(raw) - 1:
            return ParsedGeneration("EXTRA_TEXT", None, True)
        body = raw[:first_eos]
    else:
        body = raw
        if hit_max_new_tokens:
            return ParsedGeneration("TRUNCATED", None, False)
    if not body:
        return ParsedGeneration("INVALID_FORMAT", None, terminated)
    if any(value in (11, 12, 13, 14, 15) for value in body):
        return ParsedGeneration("EXTRA_TEXT", None, terminated)
    if any(value not in ALLOWED_GRID_TOKEN_IDS for value in body):
        return ParsedGeneration("INVALID_COLOR", None, terminated)
    try:
        grid = grid_from_native_tokens(body)
    except GenerationAuditError:
        return ParsedGeneration("INVALID_FORMAT", None, terminated)
    expected = tuple(tuple(int(cell) for cell in row) for row in gold_grid)
    if len(grid) != len(expected) or len(grid[0]) != len(expected[0]):
        return ParsedGeneration("INVALID_DIMENSIONS", grid, terminated)
    return ParsedGeneration("VALID_GRID", grid, terminated)


def score_record(example: GenerationExample, parsed: ParsedGeneration) -> dict[str, Any]:
    exact = parsed.classification == "VALID_GRID" and parsed.grid == example.gold_grid
    dimensions = parsed.grid is not None and (
        len(parsed.grid) == len(example.gold_grid) and len(parsed.grid[0]) == len(example.gold_grid[0])
    )
    total_cells = sum(len(row) for row in example.gold_grid)
    correct_cells = 0
    if dimensions and parsed.grid is not None:
        correct_cells = sum(
            int(predicted == gold)
            for predicted_row, gold_row in zip(parsed.grid, example.gold_grid, strict=True)
            for predicted, gold in zip(predicted_row, gold_row, strict=True)
        )
    return {
        "sample_id": example.sample_id,
        "source": example.source,
        "family": example.family,
        "parse_classification": parsed.classification,
        "strict_valid_grid": parsed.classification == "VALID_GRID",
        "exact_grid": exact,
        "dimension_exact": dimensions,
        "correct_cells": correct_cells,
        "total_cells": total_cells,
        "cell_accuracy": correct_cells / total_cells,
    }


def aggregate_generation(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise GenerationAuditError("EMPTY_GENERATION_RECORDS")
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    parse_counts: dict[str, int] = defaultdict(int)
    for record in records:
        by_family[str(record["family"])].append(record)
        by_source[str(record["source"])].append(record)
        parse_counts[str(record["parse_classification"])] += 1
    per_family = {name: sum(bool(row["exact_grid"]) for row in rows) / len(rows) for name, rows in sorted(by_family.items())}
    per_source = {name: sum(bool(row["exact_grid"]) for row in rows) / len(rows) for name, rows in sorted(by_source.items())}
    total = len(records)
    return {
        "total": total,
        "exact_grid_correct": sum(bool(row["exact_grid"]) for row in records),
        "micro_exact_grid_accuracy": sum(bool(row["exact_grid"]) for row in records) / total,
        "macro_family_exact_grid_accuracy": sum(per_family.values()) / len(per_family),
        "source_macro_exact_grid_accuracy": sum(per_source.values()) / len(per_source),
        "valid_format_rate": sum(bool(row["strict_valid_grid"]) for row in records) / total,
        "dimension_accuracy": sum(bool(row["dimension_exact"]) for row in records) / total,
        "cell_accuracy": sum(float(row["cell_accuracy"]) for row in records) / total,
        "per_family_exact_accuracy": per_family,
        "per_source_exact_accuracy": per_source,
        "parse_classification_counts": dict(sorted(parse_counts.items())),
    }


def classify_novel_gain(base: float, trained: float) -> str:
    delta = trained - base
    if delta < 0:
        return "REGRESSION"
    if delta >= 0.20:
        return "STRONG_GENERATIVE_GAIN"
    if delta >= 0.10:
        return "CLEAR_GENERATIVE_GAIN"
    if delta >= 0.03:
        return "MODEST_GENERATIVE_GAIN"
    return "NO_CLEAR_GENERATIVE_GAIN"


def best_checkpoint(points: Iterable[dict[str, Any]]) -> dict[str, Any]:
    values = list(points)
    if not values:
        raise GenerationAuditError("NO_GENERATION_POINTS")
    return max(values, key=lambda row: (float(row["novel"]["macro_family_exact_grid_accuracy"]), -MODEL_STATES.index(str(row["checkpoint"]))))


def saturation_status(points: Sequence[dict[str, Any]]) -> str:
    lookup = {str(row["checkpoint"]): float(row["novel"]["macro_family_exact_grid_accuracy"]) for row in points}
    final = lookup["tokens_2000000"]
    if max(lookup["tokens_1000000"], lookup["tokens_1500000"]) >= final - 0.01:
        return "GENERATION_SATURATION_BEFORE_2M"
    return "NO_GENERATION_SATURATION_BEFORE_2M"


def select_calibration_examples(novel: Sequence[GenerationExample], replay: Sequence[GenerationExample]) -> list[tuple[str, GenerationExample, str]]:
    """Select a fixed 20-example length-stratified calibration cohort.

    Novel contributes min/median/max prompt lengths for every one of its four
    families (12). Replay contributes min/max for each of its four sources
    (8). Ties are broken by sample ID.
    """
    result: list[tuple[str, GenerationExample, str]] = []
    novel_groups: dict[str, list[GenerationExample]] = defaultdict(list)
    replay_groups: dict[str, list[GenerationExample]] = defaultdict(list)
    for item in novel:
        novel_groups[item.family].append(item)
    for item in replay:
        replay_groups[item.source].append(item)
    if len(novel_groups) != 4 or len(replay_groups) != 4:
        raise GenerationAuditError("CALIBRATION_STRATA_MISMATCH")
    for family, values in sorted(novel_groups.items()):
        ordered = sorted(values, key=lambda item: (len(item.prompt_ids), item.sample_id))
        for label, index in (("short", 0), ("median", len(ordered) // 2), ("long", len(ordered) - 1)):
            result.append(("novel", ordered[index], f"family:{family}:{label}"))
    for source, values in sorted(replay_groups.items()):
        ordered = sorted(values, key=lambda item: (len(item.prompt_ids), item.sample_id))
        for label, index in (("short", 0), ("long", len(ordered) - 1)):
            result.append(("replay", ordered[index], f"source:{source}:{label}"))
    if len(result) != 20 or len({item.sample_id for _cohort, item, _stratum in result}) != 20:
        raise GenerationAuditError("CALIBRATION_SELECTION_NOT_20_UNIQUE")
    return result
