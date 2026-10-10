#!/usr/bin/env python3
"""CPU-only inference from E03 V3 sufficient Gram statistics."""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

WEAK = ("connected_components", "inside_contains", "width", "difference", "orientation")
PROTECTED = ("same_color", "color_mapping")
# The 72-row frozen cohort has nine equal strata.  The last two are not part
# of the primary contrast, but sampling them in this declared order keeps the
# complete stratified resample process-bound and auditable.
OTHER_FAMILIES = ("object_selection_then_action", "novel_two_operation_order")
FAMILY_ORDER = WEAK + PROTECTED + OTHER_FAMILIES
FROZEN_REPLICATES = 10000
FROZEN_SEED = 20261010


def cosine(gram, left: list[int], right: list[int]) -> float:
    numerator = sum(gram[i][j] for i in left for j in right) / (len(left) * len(right))
    left_norm = sum(gram[i][j] for i in left for j in left) / (len(left) * len(left))
    right_norm = sum(gram[i][j] for i in right for j in right) / (len(right) * len(right))
    if left_norm <= 0 or right_norm <= 0:
        raise ValueError("E03_V3_ZERO_NORM")
    return numerator / math.sqrt(left_norm * right_norm)


def weighted_cosine(gram, left: list[int], left_weights: list[float], right: list[int], right_weights: list[float]) -> float:
    def bilinear(a, aw, b, bw):
        return sum(weight_i * weight_j * gram[i][j] for i, weight_i in zip(a, aw) for j, weight_j in zip(b, bw))
    left_total, right_total = sum(left_weights), sum(right_weights)
    if left_total <= 0 or right_total <= 0:
        raise ValueError("E03_V3_ZERO_TOKEN_WEIGHT")
    numerator = bilinear(left, left_weights, right, right_weights) / (left_total * right_total)
    left_norm = bilinear(left, left_weights, left, left_weights) / (left_total * left_total)
    right_norm = bilinear(right, right_weights, right, right_weights) / (right_total * right_total)
    if left_norm <= 0 or right_norm <= 0:
        raise ValueError("E03_V3_ZERO_NORM")
    return numerator / math.sqrt(left_norm * right_norm)


def bootstrap(gram, families: list[str], token_counts: list[int], replicates: int, seed: int) -> dict[str, list[float]]:
    if len(families) != 72 or set(families) != set(FAMILY_ORDER):
        raise ValueError("E03_V3_BOOTSTRAP_FAMILY_IDENTITY_INVALID")
    if len(token_counts) != len(families) or any(not isinstance(value, int) or value <= 0 for value in token_counts):
        raise ValueError("E03_V3_BOOTSTRAP_TOKEN_COUNTS_INVALID")
    family_indices = {name: [i for i, value in enumerate(families) if value == name] for name in FAMILY_ORDER}
    if any(len(family_indices[name]) != 8 for name in FAMILY_ORDER):
        raise ValueError("E03_V3_BOOTSTRAP_FAMILY_ROWS_INVALID")
    rng = random.Random(seed)
    output = {name: [] for name in WEAK + ("secondary_combined",)}
    for _ in range(replicates):
        sampled = {name: [rng.choice(indices) for _ in range(8)] for name, indices in family_indices.items()}
        protected = sampled["same_color"] + sampled["color_mapping"]
        for name in WEAK:
            output[name].append(cosine(gram, sampled[name], protected))
        weak = [item for name in WEAK for item in sampled[name]]
        protected_weights = [float(token_counts[item]) for item in protected]
        weak_weights = [float(token_counts[item]) for item in weak]
        output["secondary_combined"].append(weighted_cosine(gram, weak, weak_weights, protected, protected_weights))
    return output


def percentile(values: list[float], fraction: float) -> float:
    values = sorted(values)
    index = fraction * (len(values) - 1)
    lo, hi = int(index), min(int(index) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (index - lo)


def classify(point: dict[str, float], samples: dict[str, list[float]], repeatability: dict) -> str:
    intervals = {name: [percentile(samples[name], 0.025), percentile(samples[name], 0.975)] for name in WEAK}
    supported = [name for name in WEAK if point[name] <= -0.05 and intervals[name][1] < 0]
    secondary = [percentile(samples["secondary_combined"], 0.025), percentile(samples["secondary_combined"], 0.975)]
    if repeatability.get("status") != "PASS":
        return "INCONCLUSIVE"
    if len(supported) >= 3 or (point["secondary_combined"] <= -0.05 and secondary[1] < 0):
        return "LOCAL_GRADIENT_INTERFERENCE_SUPPORTED"
    if all(intervals[name][0] >= 0 for name in WEAK) and secondary[0] >= 0:
        return "GRADIENT_INTERFERENCE_DEPRIORITIZED"
    return "INCONCLUSIVE"


def validate_raw_identity(raw: dict, manifest_path: Path) -> tuple[list[str], list[int]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_ids = [member["episode_id"] for family in manifest["families"] for member in family["members"]]
    expected_families = [family["canonical_family"] for family in manifest["families"] for _ in family["members"]]
    families = list(raw.get("families", []))
    episode_ids = list(raw.get("episode_ids", []))
    token_counts = list(raw.get("supervised_token_counts", []))
    if episode_ids != expected_ids or families != expected_families:
        raise ValueError("E03_V3_RAW_COHORT_IDENTITY_INVALID")
    if len(token_counts) != len(expected_ids) or any(not isinstance(value, int) or value <= 0 for value in token_counts):
        raise ValueError("E03_V3_RAW_TOKEN_COUNTS_INVALID")
    return families, token_counts


def main() -> None:
    import numpy as np
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261010)
    args = parser.parse_args()
    if args.replicates != FROZEN_REPLICATES or args.seed != FROZEN_SEED:
        raise SystemExit("E03_V3_BOOTSTRAP_CONTRACT_INVALID")
    raw = json.loads(args.raw.read_text(encoding="utf-8"))
    if raw.get("status") != "COMPLETE_NO_UPDATE" or raw.get("optimizer_steps") != 0 or raw.get("generation_calls") != 0:
        raise SystemExit("E03_V3_RAW_NOT_INTERPRETABLE")
    gram = np.asarray(raw["gram_matrix"], dtype=np.float64)
    families, token_counts = validate_raw_identity(raw, args.manifest)
    if gram.shape != (72, 72) or not np.allclose(gram, gram.T, rtol=0, atol=1e-8) or not np.isfinite(gram).all():
        raise SystemExit("E03_V3_GRAM_INVALID")
    matrix = gram.tolist()
    index = {name: [i for i, value in enumerate(families) if value == name] for name in FAMILY_ORDER}
    protected = index["same_color"] + index["color_mapping"]
    point = {name: cosine(matrix, index[name], protected) for name in WEAK}
    weak = [item for name in WEAK for item in index[name]]
    point["secondary_combined"] = weighted_cosine(matrix, weak, [float(token_counts[item]) for item in weak], protected, [float(token_counts[item]) for item in protected])
    repeatability = raw.get("repeatability")
    if not isinstance(repeatability, dict):
        raise SystemExit("E03_V3_REPEATABILITY_EVIDENCE_MISSING")
    samples = bootstrap(matrix, families, token_counts, args.replicates, args.seed)
    intervals = {name: [percentile(samples[name], 0.025), percentile(samples[name], 0.975)] for name in WEAK}
    intervals["secondary_combined"] = [percentile(samples["secondary_combined"], 0.025), percentile(samples["secondary_combined"], 0.975)]
    args.output.write_text(json.dumps({"status": "COMPLETE_NO_UPDATE", "point_cosines": point, "bootstrap_95_ci": intervals, "repeatability": repeatability, "decision": classify(point, samples, repeatability), "bootstrap_replicates": args.replicates, "bootstrap_seed": args.seed, "interpretation_limit": "local gradient disagreement is not causal proof of forgetting"}, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
