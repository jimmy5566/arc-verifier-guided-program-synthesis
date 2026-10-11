#!/usr/bin/env python3
"""Execute the Director-approved, CPU-only E05 V2 residual decomposition."""
from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/capability_repair_baseline_v1/e05_v3_semantically_identifiable_demonstrations_and_adapter_storage_recovery"
SPEC_PATH = OUT / "E05_V3_NONORACLE_STRUCTURAL_RESIDUAL_DECOMPOSITION_PREFREEZE_SPEC_V2.json"
BRIEF_PATH = OUT / "E05_V3_NONORACLE_STRUCTURAL_RESIDUAL_DECOMPOSITION_PREEXECUTION_REVIEW_BRIEF_V2.json"
RESPONSE_PATH = OUT / "E05_V3_NONORACLE_STRUCTURAL_RESIDUAL_DECOMPOSITION_PREEXECUTION_REVIEW_BRIEF_V2_RESPONSE.json"
EVIDENCE = OUT / "RUN_004_REMOTE_EVIDENCE"
RESULT_ROOT = OUT / "e05_v3_nonoracle_structural_residual_decomposition_v2"
ARMS = ("RAW_V7_NO_E04E_UPDATE", "E04E_RUN003_CONTROL_EQUAL_SLOT", "E04E_RUN003_TREATMENT_EQUAL_SLOT")
RAW_BY_ARM = {
    "RAW_V7_NO_E04E_UPDATE": "RAW_V7_NO_E04E_UPDATE_RAW_GENERATIONS.jsonl",
    "E04E_RUN003_CONTROL_EQUAL_SLOT": "E04E_RUN003_CONTROL_EQUAL_SLOT_RAW_GENERATIONS.jsonl",
    "E04E_RUN003_TREATMENT_EQUAL_SLOT": "E04E_RUN003_TREATMENT_EQUAL_SLOT_RAW_GENERATIONS.jsonl",
}
CATEGORIES = (
    "EXACT_INPUT_COPY",
    "TARGET_FOREGROUND_GEOMETRY_COLOR_ONLY_ERROR",
    "TARGET_CONGRUENT_GEOMETRY_WRONG_PLACEMENT",
    "TARGET_CELL_COUNT_PRESERVED_NONCONGRUENT_GEOMETRY",
    "DIMENSION_OR_CELL_COUNT_FAILURE",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def rectangular(grid: object) -> bool:
    return isinstance(grid, list) and bool(grid) and all(isinstance(row, list) and row and len(row) == len(grid[0]) for row in grid)


def shape(grid: list[list[int]]) -> tuple[int, int]:
    return len(grid), len(grid[0])


def unique_modal_color(grid: list[list[int]]) -> int | None:
    counts = Counter(cell for row in grid for cell in row)
    maximum = max(counts.values())
    candidates = [color for color, count in counts.items() if count == maximum]
    return candidates[0] if len(candidates) == 1 else None


def support(grid: list[list[int]], background: int) -> set[tuple[int, int]]:
    return {(row, col) for row, cells in enumerate(grid) for col, color in enumerate(cells) if color != background}


def translated(source: set[tuple[int, int]], target: set[tuple[int, int]]) -> bool:
    if not source or not target or len(source) != len(target):
        return False
    source_anchor = min(source)
    target_anchor = min(target)
    dr, dc = target_anchor[0] - source_anchor[0], target_anchor[1] - source_anchor[1]
    return (dr != 0 or dc != 0) and {(row + dr, col + dc) for row, col in source} == target


def classify(parsed: list[list[int]], test_input: list[list[int]], target: list[list[int]]) -> str:
    """Strict, mutually exclusive V2 taxonomy for one already-eligible view."""
    if parsed == test_input:
        return CATEGORIES[0]
    if shape(parsed) != shape(target):
        return CATEGORIES[4]
    background = unique_modal_color(target)
    if background is None:
        raise ValueError("TIED_TARGET_BACKGROUND")
    target_support, parsed_support = support(target, background), support(parsed, background)
    background_ok = all(parsed[row][col] == background for row in range(len(target)) for col in range(len(target[0])) if (row, col) not in target_support)
    if parsed_support == target_support and background_ok and any(parsed[row][col] != target[row][col] for row, col in target_support):
        return CATEGORIES[1]
    if translated(target_support, parsed_support) and all(parsed[row][col] == background for row in range(len(target)) for col in range(len(target[0])) if (row, col) not in parsed_support):
        return CATEGORIES[2]
    if len(parsed_support) == len(target_support):
        return CATEGORIES[3]
    return CATEGORIES[4]


def percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        raise ValueError("EMPTY_PERCENTILE")
    index = (len(sorted_values) - 1) * p
    low, high = int(index), min(int(index) + 1, len(sorted_values) - 1)
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (index - low)


def bootstrap(vectors: list[dict], seed: int, replicates: int) -> dict:
    rng = random.Random(seed)
    samples = {category: [] for category in CATEGORIES}
    denominator_zero = False
    count = len(vectors)
    for _ in range(replicates):
        draw = [vectors[rng.randrange(count)] for _ in range(count)]
        denominator = sum(item["eligible_count"] for item in draw)
        if denominator == 0:
            denominator_zero = True
            continue
        for category in CATEGORIES:
            samples[category].append(sum(item["category_counts"][category] for item in draw) / denominator)
    if denominator_zero:
        return {"status": "INCONCLUSIVE_ZERO_ELIGIBLE_DENOMINATOR", "replicates": replicates}
    return {"status": "VALID", "replicates": replicates, "category_ci95": {category: [percentile(sorted(values), .025), percentile(sorted(values), .975)] for category, values in samples.items()}}


def main() -> None:
    if RESULT_ROOT.exists():
        raise RuntimeError("RESULT_ROOT_ALREADY_EXISTS_NO_RERUN")
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8-sig"))
    response = json.loads(RESPONSE_PATH.read_text(encoding="utf-8-sig"))
    if sha(SPEC_PATH) != "8292f942676af5627e231e5081e7d4a2559792362b8f531d46823bb9f1eb4814":
        raise RuntimeError("SPECIFICATION_SHA256_MISMATCH")
    if response.get("reviewed_brief_sha256") != sha(BRIEF_PATH) or response.get("decision") != "CONTINUE_CONTROLLER":
        raise RuntimeError("DIRECTOR_RESPONSE_BINDING_MISMATCH")
    if not response.get("execution_authorization", {}).get("cpu_only_residual_decomposition"):
        raise RuntimeError("CPU_EXECUTION_NOT_AUTHORIZED")
    if any(response.get("execution_authorization", {}).get(key) for key in ("model_loading", "gpu_work", "training", "inference", "generation", "gradient_work", "target_expansion", "gold_dgold_final_audit")):
        raise RuntimeError("FORBIDDEN_EXECUTION_AUTHORIZATION_PRESENT")
    if tuple(spec["taxonomy_order"]) != CATEGORIES or spec["bootstrap"]["seed"] != 20261013 or spec["bootstrap"]["replicates"] != 10000:
        raise RuntimeError("FROZEN_SPEC_CONTRACT_MISMATCH")

    sidecar, prompts = read_jsonl(OUT / "TARGET_SCORER_SIDECAR.jsonl"), read_jsonl(OUT / "INPUT_PROMPTS.jsonl")
    identities = {name: sha(OUT / name) for name in ("MANIFEST.json", "INPUT_PROMPTS.jsonl", "TARGET_SCORER_SIDECAR.jsonl")}
    if any(spec["reused_artifacts_sha256"].get(name) != digest for name, digest in identities.items()):
        raise RuntimeError("REUSED_ARTIFACT_IDENTITY_MISMATCH")
    side_by_id, prompt_by_id = {row["episode_id"]: row for row in sidecar}, {row["episode_id"]: row for row in prompts}
    if len(side_by_id) != 768 or len(prompt_by_id) != 768 or set(side_by_id) != set(prompt_by_id):
        raise RuntimeError("EPISODE_IDENTITY_OR_COMPLETENESS_MISMATCH")
    required_conditions = tuple(spec["population"]["scope"])
    required_ids = {episode_id for episode_id, row in side_by_id.items() if row["condition"] in required_conditions}
    if len(required_ids) != 384:
        raise RuntimeError("REQUIRED_CONDITION_CARDINALITY_MISMATCH")

    raw_hashes = {name: sha(EVIDENCE / name) for name in RAW_BY_ARM.values()}
    if raw_hashes != spec["raw_evidence_sha256"]:
        raise RuntimeError("RAW_EVIDENCE_SHA256_MISMATCH")
    rows_by_arm = {arm: read_jsonl(EVIDENCE / filename) for arm, filename in RAW_BY_ARM.items()}
    vectors, invalidity = [], []
    for arm, rows in rows_by_arm.items():
        by_id = {row.get("episode_id"): row for row in rows}
        if len(rows) != 768 or len(by_id) != 768 or set(by_id) != set(side_by_id):
            raise RuntimeError("RAW_ARM_EPISODE_COMPLETENESS_MISMATCH:" + arm)
        if any(row.get("arm_id") != arm for row in rows):
            raise RuntimeError("RAW_ARM_ID_MISMATCH:" + arm)
        for episode_id in sorted(required_ids):
            raw, target_row, prompt_row = by_id[episode_id], side_by_id[episode_id], prompt_by_id[episode_id]
            condition, base, turn = target_row["condition"], target_row["canonical_base_id"], target_row["turn"]
            target = target_row.get("target")
            try:
                test = prompt_row["prompt"]["test"]
                if len(test) != 1 or not rectangular(test[0]["input"]) or not rectangular(target):
                    raise ValueError("MALFORMED_TEST_OR_TARGET")
                test_input = test[0]["input"]
                parsed = raw.get("parsed_grid")
                if not raw.get("parser_valid"):
                    vector = {"arm": arm, "condition": condition, "canonical_base_id": base, "turn": turn, "episode_id": episode_id, "status": "PARSER_INVALID", "category": None}
                elif not rectangular(parsed):
                    raise ValueError("PARSER_VALID_GRID_MALFORMED")
                elif parsed == target:
                    vector = {"arm": arm, "condition": condition, "canonical_base_id": base, "turn": turn, "episode_id": episode_id, "status": "EXACT_TARGET_MAPPING", "category": None}
                else:
                    category = classify(parsed, test_input, target)
                    if category not in CATEGORIES:
                        raise ValueError("UNCLASSIFIABLE_ELIGIBLE_VIEW")
                    vector = {"arm": arm, "condition": condition, "canonical_base_id": base, "turn": turn, "episode_id": episode_id, "status": "ELIGIBLE", "category": category}
                vectors.append(vector)
            except (KeyError, TypeError, ValueError) as exc:
                invalidity.append({"arm": arm, "episode_id": episode_id, "error": str(exc)})
    if invalidity:
        result = {"status": "INVALID_NOT_INTERPRETABLE", "invalidity": invalidity, "execution_authorization": "CPU_ONLY", "source_identities": identities, "raw_evidence_sha256": raw_hashes}
        atomic(RESULT_ROOT / "RESULT_V2.json", result)
        atomic(RESULT_ROOT / "EXECUTION_RECEIPT_V2.json", {"status": result["status"], "result_sha256": sha(RESULT_ROOT / "RESULT_V2.json"), "model_loaded": False, "gpu_used": False, "training": False, "generation": False, "final_audit_opened": False})
        print(result["status"])
        return
    grouped = defaultdict(list)
    for vector in vectors:
        grouped[(vector["arm"], vector["condition"], vector["canonical_base_id"])].append(vector)
    per_base, aggregates = [], {}
    for arm in ARMS:
        for condition in required_conditions:
            bases = sorted({base for candidate_arm, candidate_condition, base in grouped if candidate_arm == arm and candidate_condition == condition})
            if len(bases) != 48:
                raise RuntimeError("CANONICAL_BASE_COUNT_MISMATCH:" + arm + ":" + condition)
            base_vectors = []
            for base in bases:
                views = sorted(grouped[(arm, condition, base)], key=lambda row: row["turn"])
                if [row["turn"] for row in views] != [0, 1, 2, 3]:
                    raise RuntimeError("TURN_COMPLETENESS_MISMATCH:" + arm + ":" + condition + ":" + base)
                counts = Counter(row["category"] for row in views if row["status"] == "ELIGIBLE")
                item = {"canonical_base_id": base, "eligible_count": sum(counts.values()), "parser_invalid_count": sum(row["status"] == "PARSER_INVALID" for row in views), "exact_target_count": sum(row["status"] == "EXACT_TARGET_MAPPING" for row in views), "category_counts": {category: counts[category] for category in CATEGORIES}, "four_view_vector": views}
                base_vectors.append(item)
                per_base.append({"arm": arm, "condition": condition, **item})
            denominator = sum(item["eligible_count"] for item in base_vectors)
            aggregate = {"canonical_bases": 48, "eligible_view_count": denominator, "parser_invalid_view_count": sum(item["parser_invalid_count"] for item in base_vectors), "exact_target_view_count": sum(item["exact_target_count"] for item in base_vectors), "primary_conditional_category_share": ({category: sum(item["category_counts"][category] for item in base_vectors) / denominator for category in CATEGORIES} if denominator else None), "unconditional_four_view_prevalence": {category: sum(item["category_counts"][category] / 4 for item in base_vectors) / 48 for category in CATEGORIES}, "eligible_view_coverage": sum(item["eligible_count"] / 4 for item in base_vectors) / 48, "bootstrap": bootstrap(base_vectors, 20261013, 10000)}
            aggregates[f"{arm}|{condition}"] = aggregate
    for condition in required_conditions:
        treatment = aggregates[f"E04E_RUN003_TREATMENT_EQUAL_SLOT|{condition}"]
        if treatment["bootstrap"]["status"] == "VALID":
            lower = {category: treatment["bootstrap"]["category_ci95"][category][0] for category in CATEGORIES}
            dominant = [category for category, value in lower.items() if value > .5]
            treatment["prospective_dominance"] = "MECHANISM_DOMINANT:" + dominant[0] if len(dominant) == 1 else "INCONCLUSIVE"
        else:
            treatment["prospective_dominance"] = "INCONCLUSIVE"
    result = {"schema_version": 2, "protocol_id": spec["protocol_id"], "stage": "CPU_ONLY_E05_V3_NONORACLE_STRUCTURAL_RESIDUAL_DECOMPOSITION_EXECUTION_V2", "status": "COMPLETE_VALID_DESCRIPTIVE_ONLY", "specification_sha256": sha(SPEC_PATH), "director_response_sha256": sha(RESPONSE_PATH), "source_identities": identities, "raw_evidence_sha256": raw_hashes, "vector_count": len(vectors), "aggregate": aggregates, "prohibitions_observed": {"model_loaded": False, "gpu_used": False, "training": False, "inference": False, "generation": False, "gradient_work": False, "final_audit_opened": False}, "interpretation_limit": "Reused synthetic development evidence only; descriptive morphology, not causal mechanism, checkpoint promotion, intervention selection, or independent ARC generalization."}
    atomic(RESULT_ROOT / "PER_BASE_VECTORS_V2.json", {"schema_version": 1, "vectors": per_base})
    atomic(RESULT_ROOT / "RESULT_V2.json", result)
    receipt = {"status": result["status"], "result_sha256": sha(RESULT_ROOT / "RESULT_V2.json"), "per_base_vectors_sha256": sha(RESULT_ROOT / "PER_BASE_VECTORS_V2.json"), "model_loaded": False, "gpu_used": False, "training": False, "generation": False, "final_audit_opened": False}
    atomic(RESULT_ROOT / "EXECUTION_RECEIPT_V2.json", receipt)
    brief = {"brief_type": "DIRECTOR_E05_V3_NONORACLE_STRUCTURAL_RESIDUAL_DECOMPOSITION_RESULT_REVIEW_V2", "request": "Interpret this frozen CPU-only descriptive result only; do not infer causal mechanism, select intervention, promote checkpoint, or authorize model/GPU work.", "result_sha256": receipt["result_sha256"], "vectors_sha256": receipt["per_base_vectors_sha256"], "execution_receipt_sha256": sha(RESULT_ROOT / "EXECUTION_RECEIPT_V2.json"), "allowed_outcomes": ["CONTINUE_CONTROLLER", "REQUIRE_CHANGES", "PAUSED", "TERMINAL"], "boundaries": spec["prohibitions"]}
    atomic(RESULT_ROOT / "RESULT_REVIEW_BRIEF_V2.json", brief)
    print(result["status"])


if __name__ == "__main__":
    main()
