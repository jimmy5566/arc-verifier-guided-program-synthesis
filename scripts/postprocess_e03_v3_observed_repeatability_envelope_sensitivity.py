#!/usr/bin/env python3
"""CPU-only, post-hoc E03 observed-repeatability envelope sensitivity.

This diagnostic never loads a model or recomputes gradients.  It propagates the
largest observed repeat relative-L2 discrepancy as a conservative per-example
norm-ball envelope through the stored E03 Gram matrix.  The output is an
observed-envelope sensitivity bound, not a confidence interval or a global
numerical-error bound.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

EXPECTED_RAW_SHA256 = "2f0245bea7467c498c325942396f17459ccd675d23a4db920968804427b4cb6f"
EXPECTED_MANIFEST_SHA256 = "2ff3fa4f71eaf4900fcfc75ba418c188048b76cd6287d92bdb1f1efcdbd8b762"
WEAK = ("connected_components", "inside_contains", "width", "difference", "orientation")
PROTECTED = ("same_color", "color_mapping")
FAMILY_ORDER = WEAK + PROTECTED + ("object_selection_then_action", "novel_two_operation_order")
TOLERANCE = 1e-12


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("NONFINITE_VALUE")
    return value


def validate(raw: dict, manifest: dict) -> tuple[list[list[float]], list[str], list[int]]:
    if raw.get("status") != "COMPLETE_NO_UPDATE" or raw.get("optimizer_steps") != 0 or raw.get("parameter_updates") != 0 or raw.get("generation_calls") != 0 or raw.get("final_audit_opened") is not False:
        raise ValueError("RAW_PROCESS_CONTRACT_INVALID")
    expected_ids = [m["episode_id"] for family in manifest["families"] for m in family["members"]]
    expected_families = [family["canonical_family"] for family in manifest["families"] for _ in family["members"]]
    families = list(raw.get("families", []))
    if list(raw.get("episode_ids", [])) != expected_ids or families != expected_families:
        raise ValueError("RAW_COHORT_IDENTITY_INVALID")
    if len(families) != 72 or set(families) != set(FAMILY_ORDER) or any(families.count(name) != 8 for name in FAMILY_ORDER):
        raise ValueError("RAW_FAMILY_IDENTITY_INVALID")
    tokens = list(raw.get("supervised_token_counts", []))
    if len(tokens) != 72 or any(not isinstance(value, int) or value <= 0 for value in tokens):
        raise ValueError("RAW_TOKEN_COUNTS_INVALID")
    gram = raw.get("gram_matrix")
    if not isinstance(gram, list) or len(gram) != 72 or any(not isinstance(row, list) or len(row) != 72 for row in gram):
        raise ValueError("GRAM_DIMENSION_INVALID")
    matrix = [[finite(value) for value in row] for row in gram]
    for i in range(72):
        if matrix[i][i] <= 0:
            raise ValueError("GRAM_ZERO_NORM")
        for j in range(72):
            if abs(matrix[i][j] - matrix[j][i]) > 1e-8:
                raise ValueError("GRAM_NOT_SYMMETRIC")
    return matrix, families, tokens


def coefficients(indices: list[int], n: int, weights: list[float] | None = None) -> list[float]:
    if not indices:
        raise ValueError("EMPTY_AGGREGATE")
    weights = weights if weights is not None else [1.0] * len(indices)
    if len(weights) != len(indices) or any(finite(weight) <= 0 for weight in weights):
        raise ValueError("AGGREGATE_WEIGHTS_INVALID")
    total = sum(float(weight) for weight in weights)
    vector = [0.0] * n
    for index, weight in zip(indices, weights):
        vector[index] += float(weight) / total
    return vector


def inner(left: list[float], right: list[float], gram: list[list[float]]) -> float:
    return sum(left[i] * gram[i][j] * right[j] for i in range(len(left)) for j in range(len(right)))


def norm(vector: list[float], gram: list[list[float]]) -> float:
    squared = inner(vector, vector, gram)
    if not math.isfinite(squared) or squared <= 0:
        raise ValueError("AGGREGATE_ZERO_NORM")
    return math.sqrt(squared)


def cosine(left: list[float], right: list[float], gram: list[list[float]]) -> float:
    return inner(left, right, gram) / (norm(left, gram) * norm(right, gram))


def perturbation_radius(indices: list[int], weights: list[float], gram: list[list[float]], epsilon: float) -> float:
    """Worst-case aggregate perturbation under ||delta_i|| <= epsilon ||g_i||."""
    total = sum(weights)
    return epsilon * sum(weight * math.sqrt(gram[index][index]) for index, weight in zip(indices, weights)) / total


def conservative_cosine_interval(left: list[float], left_indices: list[int], left_weights: list[float], right: list[float], right_indices: list[int], right_weights: list[float], gram: list[list[float]], epsilon: float) -> dict:
    left_norm, right_norm = norm(left, gram), norm(right, gram)
    left_radius = perturbation_radius(left_indices, left_weights, gram, epsilon)
    right_radius = perturbation_radius(right_indices, right_weights, gram, epsilon)
    point = cosine(left, right, gram)
    numerator_radius = left_radius * right_norm + left_norm * right_radius + left_radius * right_radius
    numerator = point * left_norm * right_norm
    lower = (numerator - numerator_radius) / ((left_norm + left_radius) * (right_norm + right_radius))
    if left_radius >= left_norm or right_radius >= right_norm:
        upper = 1.0
        lower = -1.0
    else:
        upper = (numerator + numerator_radius) / ((left_norm - left_radius) * (right_norm - right_radius))
    lower, upper = max(-1.0, lower), min(1.0, upper)
    if lower > point + 1e-12 or upper < point - 1e-12 or lower > upper:
        raise ValueError("INVALID_BOUND_CONSTRUCTION")
    return {
        "point_cosine": point,
        "lower": lower,
        "upper": upper,
        "zero_reachable": lower <= 0.0 <= upper,
        "left_mean_norm": left_norm,
        "right_mean_norm": right_norm,
        "left_aggregate_perturbation_radius": left_radius,
        "right_aggregate_perturbation_radius": right_radius,
    }


def analyse(raw_path: Path, published_path: Path, manifest_path: Path, expected_raw_sha256: str = EXPECTED_RAW_SHA256, expected_manifest_sha256: str = EXPECTED_MANIFEST_SHA256) -> dict:
    if sha256(raw_path) != expected_raw_sha256:
        raise ValueError("RAW_SHA256_MISMATCH")
    if sha256(manifest_path) != expected_manifest_sha256:
        raise ValueError("MANIFEST_SHA256_MISMATCH")
    raw, published, manifest = (json.loads(path.read_text(encoding="utf-8")) for path in (raw_path, published_path, manifest_path))
    matrix, families, token_counts = validate(raw, manifest)
    if published.get("status") != "COMPLETE_NO_UPDATE" or published.get("decision") != "INCONCLUSIVE":
        raise ValueError("PUBLISHED_E03_STATUS_INVALID")
    repeatability = raw.get("repeatability")
    epsilon = finite((repeatability or {}).get("max_relative_l2"))
    if epsilon != 0.008617405731011414:
        raise ValueError("OBSERVED_REPEATABILITY_ENVELOPE_IDENTITY_INVALID")
    by_family = {name: [index for index, value in enumerate(families) if value == name] for name in FAMILY_ORDER}
    protected_indices = by_family["same_color"] + by_family["color_mapping"]
    protected_weights = [1.0] * len(protected_indices)
    protected = coefficients(protected_indices, 72, protected_weights)
    contrasts = {}
    reconstructed = {}
    for name in WEAK:
        left_indices = by_family[name]
        left_weights = [1.0] * len(left_indices)
        left = coefficients(left_indices, 72, left_weights)
        interval = conservative_cosine_interval(left, left_indices, left_weights, protected, protected_indices, protected_weights, matrix, epsilon)
        expected = finite(published["point_cosines"][name])
        if abs(interval["point_cosine"] - expected) > TOLERANCE:
            raise ValueError("PUBLISHED_POINT_RECONSTRUCTION_FAILED:" + name)
        reconstructed[name] = interval["point_cosine"]
        contrasts[name] = interval
    weak_indices = [index for name in WEAK for index in by_family[name]]
    weak_weights = [float(token_counts[index]) for index in weak_indices]
    weighted_protected = [float(token_counts[index]) for index in protected_indices]
    secondary = conservative_cosine_interval(coefficients(weak_indices, 72, weak_weights), weak_indices, weak_weights, coefficients(protected_indices, 72, weighted_protected), protected_indices, weighted_protected, matrix, epsilon)
    expected_secondary = finite(published["point_cosines"]["secondary_combined"])
    if abs(secondary["point_cosine"] - expected_secondary) > TOLERANCE:
        raise ValueError("PUBLISHED_POINT_RECONSTRUCTION_FAILED:secondary_combined")
    reconstructed["secondary_combined"] = secondary["point_cosine"]
    contrasts["secondary_combined"] = secondary
    decision = "DIRECTIONAL_PATTERN_ROBUST_UNDER_OBSERVED_ENVELOPE" if all(item["lower"] > 0.0 for item in contrasts.values()) else "DIRECTIONAL_PATTERN_SENSITIVE_TO_OBSERVED_ENVELOPE"
    return {
        "schema_version": 1,
        "diagnostic_id": "CPU_ONLY_E03_OBSERVED_REPEATABILITY_ENVELOPE_SENSITIVITY_V1",
        "status": "COMPLETE_CPU_ONLY_POST_HOC",
        "decision": decision,
        "frozen_e03_result_status": "INCONCLUSIVE_REPEATABILITY_GATE_FAILED_PRESERVED",
        "post_hoc_label": "OBSERVED_ENVELOPE_SENSITIVITY_NOT_CONFIDENCE_INTERVAL_NOT_GLOBAL_NUMERICAL_ERROR_BOUND",
        "input_identity": {"raw_gram_sha256": expected_raw_sha256, "manifest_sha256": expected_manifest_sha256, "published_postprocessor_sha256": sha256(published_path)},
        "observed_per_example_relative_l2_envelope": epsilon,
        "reconstructed_point_cosines": reconstructed,
        "contrasts": contrasts,
        "all_six_lower_bounds_strictly_positive": all(item["lower"] > 0.0 for item in contrasts.values()),
        "interpretation_limit": "This post-hoc TRAIN-only sensitivity result neither validates the frozen E03 repeatability contract nor establishes absence of gradient interference, forgetting, fitting, generalization, or ARC transfer.",
        "forbidden_actions_observed": {"model_loaded": False, "gpu_used": False, "gradient_recomputed": False, "optimizer_constructed": False, "training": False, "generation": False, "sealed_data_opened": False},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--published", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("OUTPUT_ALREADY_EXISTS")
    try:
        result = analyse(args.raw, args.published, args.manifest)
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise SystemExit("INVALID_NOT_INTERPRETABLE:" + str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
