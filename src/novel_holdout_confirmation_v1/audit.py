"""Pure contracts for the one-time Novel V1.1 holdout confirmation.

The pre-generation path deliberately extracts only demonstration pairs, the
final target input, and compositional metadata.  The final target output is
not decoded until the post-freeze scoring path calls :func:`target_output`.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
import random
from typing import Any, Iterable, Sequence


SEED = 2_000_031
EXPECTED_DATASET_FINGERPRINT = "3b040b3fcd45d84361aa5286c3b762e268a5f7c98dae5b3c87ddeb16c5105c82"
EXPECTED_FAMILY_SHA256 = "2d81c2257ea533a54fdef00b7f77140e98b640cecf8451da67a3f3de79aa2f4c"
EXPECTED_ADAPTER_SHA256 = "f0079dd399c1f0521c378a46bd7bbb737a0abc72ef0bb6c47a6c1ae33446afce"
FAMILIES = (
    "1d_arc:1d_mirror",
    "1d_arc:1d_move_2p",
    "compositional_arc:systematicity:grow+mirror+translation",
    "compositional_arc:systematicity:grow+rotation+translation",
)


class HoldoutAuditError(RuntimeError):
    """Raised when a frozen holdout invariant is violated."""


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def ranking_hash(family: str, sample_id: str, seed: int = SEED) -> str:
    return hashlib.sha256(f"{seed}:{family}:{sample_id}".encode("utf-8")).hexdigest()


def _skip_ws(text: str, index: int) -> int:
    while index < len(text) and text[index].isspace():
        index += 1
    return index


def _value_end(text: str, index: int) -> int:
    """Return the end of one JSON value without decoding its contents."""
    index = _skip_ws(text, index)
    if index >= len(text):
        raise HoldoutAuditError("JSON_VALUE_MISSING")
    if text[index] == '"':
        index += 1
        escaped = False
        while index < len(text):
            char = text[index]
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                return index + 1
            index += 1
        raise HoldoutAuditError("UNTERMINATED_JSON_STRING")
    if text[index] in "[{":
        stack = [text[index]]
        index += 1
        in_string = False
        escaped = False
        while index < len(text) and stack:
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
            elif char == '"':
                in_string = True
            elif char in "[{":
                stack.append(char)
            elif char in "]}":
                opener = stack.pop()
                if (opener, char) not in (("[", "]"), ("{", "}")):
                    raise HoldoutAuditError("JSON_BRACKET_MISMATCH")
            index += 1
        if stack:
            raise HoldoutAuditError("UNTERMINATED_JSON_CONTAINER")
        return index
    end = index
    while end < len(text) and text[end] not in ",]} \t\r\n":
        end += 1
    return end


def object_member_raw(text: str, wanted: str) -> str:
    decoder = json.JSONDecoder()
    index = _skip_ws(text, 0)
    if index >= len(text) or text[index] != "{":
        raise HoldoutAuditError("JSON_OBJECT_REQUIRED")
    index += 1
    while True:
        index = _skip_ws(text, index)
        if index < len(text) and text[index] == "}":
            break
        key, index = decoder.raw_decode(text, index)
        index = _skip_ws(text, index)
        if index >= len(text) or text[index] != ":":
            raise HoldoutAuditError("JSON_OBJECT_COLON_MISSING")
        start = _skip_ws(text, index + 1)
        end = _value_end(text, start)
        if key == wanted:
            return text[start:end]
        index = _skip_ws(text, end)
        if index < len(text) and text[index] == ",":
            index += 1
            continue
        if index < len(text) and text[index] == "}":
            break
        raise HoldoutAuditError("JSON_OBJECT_DELIMITER_INVALID")
    raise HoldoutAuditError(f"JSON_MEMBER_MISSING={wanted}")


def array_item_raws(text: str) -> list[str]:
    index = _skip_ws(text, 0)
    if index >= len(text) or text[index] != "[":
        raise HoldoutAuditError("JSON_ARRAY_REQUIRED")
    index += 1
    result: list[str] = []
    while True:
        index = _skip_ws(text, index)
        if index < len(text) and text[index] == "]":
            return result
        end = _value_end(text, index)
        result.append(text[index:end])
        index = _skip_ws(text, end)
        if index < len(text) and text[index] == ",":
            index += 1
            continue
        if index < len(text) and text[index] == "]":
            return result
        raise HoldoutAuditError("JSON_ARRAY_DELIMITER_INVALID")


def _pair(raw: str, *, allow_output: bool) -> dict[str, Any]:
    raw = raw.strip()
    if raw.startswith("["):
        items = array_item_raws(raw)
        if len(items) != 2:
            raise HoldoutAuditError("PAIR_ARITY_MISMATCH")
        result = {"input": json.loads(items[0])}
        if allow_output:
            result["output"] = json.loads(items[1])
        return result
    if raw.startswith("{"):
        result = {"input": json.loads(object_member_raw(raw, "input"))}
        if allow_output:
            result["output"] = json.loads(object_member_raw(raw, "output"))
        return result
    raise HoldoutAuditError("PAIR_CONTAINER_INVALID")


def _pairs(raw: str, *, final_output: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    items = array_item_raws(raw)
    if not items:
        raise HoldoutAuditError("EMPTY_TARGET_PAIRS")
    prior = [_pair(item, allow_output=True) for item in items[:-1]]
    target = _pair(items[-1], allow_output=final_output)
    return prior, target


def composition_family(text: str) -> str:
    meta = json.loads(object_member_raw(text, "meta_data"))
    values = []
    for item in meta.values():
        if not isinstance(item, dict) or not item.get("type"):
            raise HoldoutAuditError("INVALID_COMPOSITION_METADATA")
        values.append(str(item["type"]))
    if len(values) != 3:
        raise HoldoutAuditError("COMPOSITION_ARITY_MISMATCH")
    return "compositional_arc:systematicity:" + "+".join(sorted(values))


def prompt_pairs(text: str, source: str) -> tuple[list[dict[str, Any]], list[list[int]]]:
    """Return complete demonstrations and final input without decoding Gold."""
    if source == "1d_arc":
        demos = json.loads(object_member_raw(text, "train"))
        prior, target = _pairs(object_member_raw(text, "test"), final_output=False)
        return [*demos, *prior], target["input"]
    if source == "compositional_arc":
        demos: list[dict[str, Any]] = []
        for section in ("primitive_functions", "function_compositions"):
            mapping = json.loads(object_member_raw(text, section))
            for name in sorted(mapping):
                for pair in mapping[name]:
                    if isinstance(pair, dict):
                        demos.append({"input": pair["input"], "output": pair["output"]})
                    else:
                        demos.append({"input": pair[0], "output": pair[1]})
        prior, target = _pairs(object_member_raw(text, "queries"), final_output=False)
        return [*demos, *prior], target["input"]
    raise HoldoutAuditError(f"UNKNOWN_SOURCE={source}")


def target_output(text: str, source: str) -> list[list[int]]:
    """Decode the final target output.  This is post-freeze scoring only."""
    key = "test" if source == "1d_arc" else "queries"
    _prior, target = _pairs(object_member_raw(text, key), final_output=True)
    output = target.get("output")
    if not isinstance(output, list) or not output:
        raise HoldoutAuditError("TARGET_OUTPUT_INVALID")
    return output


def select_sentinel(candidates: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        family = str(candidate["family"])
        if family in FAMILIES:
            row = dict(candidate)
            row["ranking_sha256"] = ranking_hash(family, str(row["sample_id"]))
            grouped[family].append(row)
    if set(grouped) != set(FAMILIES):
        raise HoldoutAuditError("HOLDOUT_FAMILY_SET_MISMATCH")
    selected: list[dict[str, Any]] = []
    for family in FAMILIES:
        ordered = sorted(grouped[family], key=lambda row: (row["ranking_sha256"], row["sample_id"]))
        if family.startswith("1d_arc:") and len(ordered) != 50:
            raise HoldoutAuditError(f"ONE_D_HOLDOUT_COUNT_MISMATCH={family}:{len(ordered)}")
        if len(ordered) < 50:
            raise HoldoutAuditError(f"HOLDOUT_FAMILY_TOO_SMALL={family}:{len(ordered)}")
        selected.extend(ordered[:50])
    if len(selected) != 200 or len({row["sample_id"] for row in selected}) != 200:
        raise HoldoutAuditError("SENTINEL_NOT_200_UNIQUE")
    return selected


def paired_counts(base: Sequence[bool], trained: Sequence[bool]) -> dict[str, int]:
    if len(base) != len(trained) or not base:
        raise HoldoutAuditError("PAIRED_LENGTH_MISMATCH")
    result = {"n00": 0, "n01": 0, "n10": 0, "n11": 0}
    for left, right in zip(base, trained, strict=True):
        result[f"n{int(bool(left))}{int(bool(right))}"] += 1
    return result


def mcnemar_exact_p(n01: int, n10: int) -> float:
    discordant = n01 + n10
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(0, min(n01, n10) + 1)) / (2 ** discordant)
    return min(1.0, 2.0 * tail)


def stratified_bootstrap(
    rows: Sequence[dict[str, Any]], *, repetitions: int = 10_000, seed: int = SEED
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["family"])].append(row)
    if set(grouped) != set(FAMILIES) or {len(value) for value in grouped.values()} != {50}:
        raise HoldoutAuditError("BOOTSTRAP_STRATA_NOT_BALANCED_50")
    point = sum(
        sum(float(item["trained_exact"]) - float(item["base_exact"]) for item in grouped[family]) / 50
        for family in FAMILIES
    ) / 4
    rng = random.Random(seed)
    values: list[float] = []
    for _ in range(repetitions):
        family_deltas = []
        for family in FAMILIES:
            group = grouped[family]
            sample = [group[rng.randrange(50)] for _index in range(50)]
            family_deltas.append(sum(float(item["trained_exact"]) - float(item["base_exact"]) for item in sample) / 50)
        values.append(sum(family_deltas) / 4)
    values.sort()
    lower = values[int(0.025 * repetitions)]
    upper = values[min(repetitions - 1, int(0.975 * repetitions))]
    return {"seed": seed, "resamples": repetitions, "point_estimate": point, "percentile_95_ci": [lower, upper]}


def transfer_classification(point: float, lower: float, upper: float) -> str:
    if upper < 0:
        return "STRONG_REGRESSION"
    if point < 0:
        return "REGRESSION_WARNING"
    if point >= 0.10 and lower > 0:
        return "STRONG_TRANSFER_CONFIRMED"
    if point >= 0.03 and lower > 0:
        return "TRANSFER_CONFIRMED"
    if point > 0 and lower <= 0 <= upper:
        return "POSITIVE_BUT_INCONCLUSIVE"
    return "NO_CLEAR_TRANSFER"
