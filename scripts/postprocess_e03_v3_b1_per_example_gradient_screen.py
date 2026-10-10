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


def cosine(gram, left: list[int], right: list[int]) -> float:
    numerator = sum(gram[i][j] for i in left for j in right) / (len(left) * len(right))
    left_norm = sum(gram[i][j] for i in left for j in left) / (len(left) * len(left))
    right_norm = sum(gram[i][j] for i in right for j in right) / (len(right) * len(right))
    if left_norm <= 0 or right_norm <= 0:
        raise ValueError("E03_V3_ZERO_NORM")
    return numerator / math.sqrt(left_norm * right_norm)


def bootstrap(gram, families: list[str], replicates: int, seed: int) -> dict[str, list[float]]:
    family_indices = {name: [i for i, value in enumerate(families) if value == name] for name in set(families)}
    if any(len(family_indices[name]) != 8 for name in WEAK + PROTECTED):
        raise ValueError("E03_V3_BOOTSTRAP_FAMILY_ROWS_INVALID")
    rng = random.Random(seed)
    output = {name: [] for name in WEAK}
    for _ in range(replicates):
        sampled = {name: [rng.choice(indices) for _ in range(8)] for name, indices in family_indices.items()}
        protected = sampled["same_color"] + sampled["color_mapping"]
        for name in WEAK:
            output[name].append(cosine(gram, sampled[name], protected))
    return output


def percentile(values: list[float], fraction: float) -> float:
    values = sorted(values)
    index = fraction * (len(values) - 1)
    lo, hi = int(index), min(int(index) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (index - lo)


def classify(point: dict[str, float], samples: dict[str, list[float]]) -> str:
    intervals = {name: [percentile(samples[name], 0.025), percentile(samples[name], 0.975)] for name in WEAK}
    supported = [name for name in WEAK if point[name] <= -0.05 and intervals[name][1] < 0]
    if len(supported) >= 3:
        return "LOCAL_GRADIENT_INTERFERENCE_SUPPORTED"
    if all(intervals[name][0] >= 0 for name in WEAK):
        return "GRADIENT_INTERFERENCE_DEPRIORITIZED"
    return "INCONCLUSIVE"


def main() -> None:
    import numpy as np
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261010)
    args = parser.parse_args()
    raw = json.loads(args.raw.read_text(encoding="utf-8"))
    if raw.get("status") != "COMPLETE_NO_UPDATE" or raw.get("optimizer_steps") != 0 or raw.get("generation_calls") != 0:
        raise SystemExit("E03_V3_RAW_NOT_INTERPRETABLE")
    gram = np.asarray(raw["gram_matrix"], dtype=np.float64)
    families = list(raw["families"])
    if gram.shape != (72, 72) or not np.allclose(gram, gram.T, rtol=0, atol=1e-8) or not np.isfinite(gram).all():
        raise SystemExit("E03_V3_GRAM_INVALID")
    matrix = gram.tolist()
    index = {name: [i for i, value in enumerate(families) if value == name] for name in set(families)}
    protected = index["same_color"] + index["color_mapping"]
    point = {name: cosine(matrix, index[name], protected) for name in WEAK}
    samples = bootstrap(matrix, families, args.replicates, args.seed)
    intervals = {name: [percentile(samples[name], 0.025), percentile(samples[name], 0.975)] for name in WEAK}
    args.output.write_text(json.dumps({"status": "COMPLETE_NO_UPDATE", "point_cosines": point, "bootstrap_95_ci": intervals, "decision": classify(point, samples), "bootstrap_replicates": args.replicates, "bootstrap_seed": args.seed, "interpretation_limit": "local gradient disagreement is not causal proof of forgetting"}, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
