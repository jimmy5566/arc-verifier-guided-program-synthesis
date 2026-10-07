"""Pure scoring and aggregation for FOUNDATION_V2_CAPABILITY_DIAGNOSTIC_V1."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
from typing import Any, Iterable, Sequence


SOURCE_COMMIT = "9098a93d6d2c1234879b6576a2a0c29a5e35657f"
MODEL_STATES = ("base", "foundation_v2")
MAX_NEW_TOKENS = 932
CONTEXT_WINDOW = 8704
EOS_TOKEN_ID = 15
PAD_TOKEN_ID = 13
ALLOWED_GRID_TOKEN_IDS = frozenset(range(11))

INPUTS = {
    "capability": ("FOUNDATION_DIAGNOSTIC_BATTERY_V1_2.jsonl.gz", 2656, "10b84b70ae4b855f71e521333ef136af959c7e97798e8e06bfad733ffd5008c0"),
    "parameter": ("PARAMETER_GENERALIZATION_DIAGNOSTIC_V1_2.jsonl.gz", 288, "b36f2694b9e8e2892190f723cc7b99371b99a2fd2ae4fc4a96079e8e821b4b7b"),
    "composition": ("COMPOSITION_DIAGNOSTIC_DEV_V1_2.jsonl.gz", 56, "1479e4b62d76ba8d909a38222707464d33442f0e8cd7dc3f783544839989da79"),
}

COMPOSITION_CATEGORIES = (
    "GEOMETRY_TO_GEOMETRY", "SELECTOR_TO_ACTION", "RELATION_TO_SELECTOR_ACTION",
    "COUNTING_TO_CONSTRUCTION", "MASK_SET_TO_CONSTRUCTION",
    "STATE_PROGRESSION_TO_ACTION", "CONDITIONAL_TO_ACTION",
)

STAGE_CAPABILITIES = {
    "P1": {
        "above/below", "area", "color mapping", "copy", "copy reference color", "height",
        "holes/enclosures", "inside/contains", "largest", "left/right", "nearest/farthest",
        "orientation", "recolor", "relation-conditioned selector", "reflect LR", "reflect UD",
        "reflect diagonal", "rotate 180", "rotate 270", "rotate 90", "same color", "scale",
        "separators/subgrids", "width",
    },
    "P2": {
        "complete missing structure", "connected components", "difference", "overlay",
    },
    "P3": {
        "alternation", "branch-balanced conditional tasks", "compare quantities", "count components",
        "encode quantity geometrically", "if/else on attribute", "if/else on count",
        "if/else on relation", "iterative update", "periodic pattern", "progression", "propagation",
    },
}


class DiagnosticError(RuntimeError):
    pass


@dataclass(frozen=True)
class ParsedGeneration:
    classification: str
    grid: tuple[tuple[int, ...], ...] | None
    terminated_by_eos: bool


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def serialize_grid(grid: Sequence[Sequence[int]]) -> str:
    return "\n".join("".join(str(int(cell)) for cell in row) for row in grid)


def grid_from_tokens(tokens: Sequence[int]) -> tuple[tuple[int, ...], ...]:
    values = tuple(int(value) for value in tokens)
    if not values or any(value not in ALLOWED_GRID_TOKEN_IDS for value in values):
        raise DiagnosticError("NOT_NATIVE_GRID_TOKENS")
    rows: list[list[int]] = [[]]
    for value in values:
        if value == 10:
            if not rows[-1]:
                raise DiagnosticError("EMPTY_GRID_ROW")
            rows.append([])
        else:
            rows[-1].append(value)
    if not rows[-1] or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise DiagnosticError("NON_RECTANGULAR_GRID")
    return tuple(tuple(row) for row in rows)


def parse_generated_tokens(token_ids: Sequence[int], gold_grid: Sequence[Sequence[int]], *, hit_max_new_tokens: bool) -> ParsedGeneration:
    raw = [int(value) for value in token_ids]
    eos = [i for i, value in enumerate(raw) if value == EOS_TOKEN_ID]
    terminated = bool(eos)
    if terminated:
        if eos[0] != len(raw) - 1:
            return ParsedGeneration("EXTRA_TEXT", None, True)
        body = raw[:eos[0]]
    else:
        body = raw
        if hit_max_new_tokens:
            return ParsedGeneration("TRUNCATED", None, False)
    if not body:
        return ParsedGeneration("INVALID_FORMAT", None, terminated)
    if any(value not in ALLOWED_GRID_TOKEN_IDS for value in body):
        return ParsedGeneration("EXTRA_TEXT" if any(value in {11, 12, 13, 14, 15} for value in body) else "INVALID_COLOR", None, terminated)
    try:
        grid = grid_from_tokens(body)
    except DiagnosticError:
        return ParsedGeneration("INVALID_FORMAT", None, terminated)
    expected = tuple(tuple(int(cell) for cell in row) for row in gold_grid)
    if len(grid) != len(expected) or len(grid[0]) != len(expected[0]):
        return ParsedGeneration("INVALID_DIMENSIONS", grid, terminated)
    return ParsedGeneration("VALID_GRID", grid, terminated)


def score_prediction(sample_id: str, token_ids: Sequence[int], gold_grid: Sequence[Sequence[int]]) -> dict[str, Any]:
    parsed = parse_generated_tokens(token_ids, gold_grid, hit_max_new_tokens=len(token_ids) >= MAX_NEW_TOKENS and EOS_TOKEN_ID not in token_ids)
    expected = tuple(tuple(int(cell) for cell in row) for row in gold_grid)
    dimensions = parsed.grid is not None and len(parsed.grid) == len(expected) and len(parsed.grid[0]) == len(expected[0])
    correct = 0
    total = sum(len(row) for row in expected)
    if dimensions and parsed.grid is not None:
        correct = sum(int(a == b) for ar, br in zip(parsed.grid, expected, strict=True) for a, b in zip(ar, br, strict=True))
    return {
        "sample_id": sample_id,
        "parse_failure_class": parsed.classification,
        "valid_format": parsed.classification == "VALID_GRID",
        "dimension_exact": dimensions,
        "exact_grid": parsed.classification == "VALID_GRID" and parsed.grid == expected,
        "cell_accuracy": correct / total,
    }


def engineering_band(score: float, *, composite: bool = False) -> str:
    stem = "COMPOSITE_CHAIN_" if composite else ""
    if score < .40:
        return stem + "WEAK"
    if score < .75:
        return stem + "PARTIAL"
    if score < .95:
        return stem + "STRONG"
    return stem + "SATURATED"


def metric(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise DiagnosticError("EMPTY_METRIC_GROUP")
    return {
        "total": len(rows),
        "exact_correct": sum(bool(row["exact_grid"]) for row in rows),
        "exact_accuracy": sum(bool(row["exact_grid"]) for row in rows) / len(rows),
        "valid_format_rate": sum(bool(row["valid_format"]) for row in rows) / len(rows),
        "dimension_exact_rate": sum(bool(row["dimension_exact"]) for row in rows) / len(rows),
        "cell_accuracy": sum(float(row["cell_accuracy"]) for row in rows) / len(rows),
    }


def aggregate_capabilities(rows_by_state: dict[str, list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    direct, minimal, composite = [], [], []
    for state in MODEL_STATES:
        by_cap: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows_by_state[state]:
            by_cap[str(row["capability"])].append(row)
        for capability, rows in sorted(by_cap.items()):
            kind = str(rows[0]["measurement_type"])
            if kind == "MINIMAL_CONTRAST":
                pairs: dict[str, list[dict[str, Any]]] = defaultdict(list)
                for row in rows:
                    pairs[str(row["pair_id"])].append(row)
                if len(pairs) != 16 or any(len(pair) != 2 for pair in pairs.values()):
                    raise DiagnosticError(f"MINIMAL_PAIR_CONTRACT={capability}")
                pair_success = sum(all(bool(member["exact_grid"]) for member in pair) for pair in pairs.values())
                result = {"model_state": state, "capability": capability, "measurement_type": kind,
                          "contrast_pairs_correct": pair_success, "contrast_pair_total": 16,
                          "contrast_success_rate": pair_success / 16, "individual_task": metric(rows)}
                minimal.append(result)
            else:
                result = {"model_state": state, "capability": capability, "measurement_type": kind, **metric(rows)}
                (direct if kind == "DIRECT_ATOMIC" else composite).append(result)
    return direct, minimal, composite


def pair_models(rows: list[dict[str, Any]], score_key: str, *, composite: bool = False) -> list[dict[str, Any]]:
    lookup = {(row["model_state"], row["capability"]): row for row in rows}
    capabilities = sorted({row["capability"] for row in rows})
    result = []
    for capability in capabilities:
        base, fv2 = lookup[("base", capability)], lookup[("foundation_v2", capability)]
        bs, fs = float(base[score_key]), float(fv2[score_key])
        observation = 1 / (16 if fv2["measurement_type"] == "MINIMAL_CONTRAST" else 32)
        borderline = any(abs(fs - boundary) <= observation + 1e-12 for boundary in (0.40, 0.75, 0.95))
        result.append({"capability": capability, "measurement_type": fv2["measurement_type"],
                       "base_primary_score": bs, "foundation_v2_primary_score": fs, "delta": fs - bs,
                       "engineering_status": engineering_band(fs, composite=composite),
                       "borderline": borderline, "borderline_observation_width": observation,
                       "base": base, "foundation_v2": fv2})
    return result


def parameter_results(rows_by_state: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    result = []
    for axis in sorted({row["axis"] for rows in rows_by_state.values() for row in rows}):
        item: dict[str, Any] = {"axis": axis}
        for state in MODEL_STATES:
            base = metric([row for row in rows_by_state[state] if row["axis"] == axis and row["domain"] == "base"])
            extra = metric([row for row in rows_by_state[state] if row["axis"] == axis and row["domain"] == "diagnostic"])
            item[state] = {"base_domain": base, "extrapolation_domain": extra,
                           "generalization_gap": extra["exact_accuracy"] - base["exact_accuracy"]}
        item["foundation_v2_extrapolation_gain"] = item["foundation_v2"]["extrapolation_domain"]["exact_accuracy"] - item["base"]["extrapolation_domain"]["exact_accuracy"]
        result.append(item)
    return result


def composition_results(rows_by_state: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    result = []
    for category in COMPOSITION_CATEGORIES:
        item: dict[str, Any] = {"category": category, "development_exposed": True}
        for state in MODEL_STATES:
            selected = [row for row in rows_by_state[state] if row["category"] == category]
            item[state] = metric(selected)
        item["delta"] = item["foundation_v2"]["exact_accuracy"] - item["base"]["exact_accuracy"]
        result.append(item)
    return result


def stage_summary(profile: Sequence[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    assigned: set[str] = set()
    for stage, names in STAGE_CAPABILITIES.items():
        rows = [row for row in profile if row["capability"] in names]
        assigned.update(row["capability"] for row in rows)
        scores = sorted(float(row["foundation_v2_primary_score"]) for row in rows)
        counts = {band: sum(str(row["engineering_status"]).endswith(band) for row in rows) for band in ("WEAK", "PARTIAL", "STRONG", "SATURATED")}
        result[stage] = {"direct_or_minimal_capability_count": len(rows), **{f"{k.lower()}_count": v for k, v in counts.items()},
                         "borderline_count": sum(bool(row.get("borderline")) for row in rows),
                         "median_primary_score": (scores[(len(scores)-1)//2] + scores[len(scores)//2]) / 2,
                         "mean_base_to_foundation_v2_delta": sum(float(row["delta"]) for row in rows) / len(rows),
                         "capabilities": [row["capability"] for row in rows]}
    missing = sorted({row["capability"] for row in profile} - assigned)
    if missing:
        raise DiagnosticError(f"UNMAPPED_STAGE_CAPABILITIES={missing}")
    return result


def priority_map(profile: Sequence[dict[str, Any]], parameters: Sequence[dict[str, Any]], composites: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    weak_parameter = {row["axis"] for row in parameters if row["foundation_v2"]["generalization_gap"] < 0}
    weak_composites = {row["capability"] for row in composites if row["foundation_v2_primary_score"] < .40}
    result = []
    for row in profile:
        suffix = str(row["engineering_status"]).split("_")[-1]
        priority = {"WEAK": "HIGH", "PARTIAL": "MEDIUM", "STRONG": "LOW", "SATURATED": "REPLAY_ONLY"}[suffix]
        name = str(row["capability"])
        result.append({"capability": name, "measurement_type": row["measurement_type"], "priority": priority,
                       "foundation_v2_primary_score": row["foundation_v2_primary_score"], "delta": row["delta"],
                       "flags": {"BORDERLINE": bool(row.get("borderline")), "REGRESSION_VS_BASE": row["delta"] < 0,
                                 "PARAMETER_GENERALIZATION_GAP": bool(weak_parameter) and any(token in name for token in ("color", "position", "count", "size", "orientation", "displacement")),
                                 "COMPOSITION_PREREQUISITE": any(name in cap for cap in weak_composites)}})
    return result

