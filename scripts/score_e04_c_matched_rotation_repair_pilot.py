"""CPU-only target-side scoring for E04-C after both raw files are frozen."""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.e04_c_matched_rotation_execution import ARMS, E04CFailure, ROOT, atomic_json, fail, sha_path
from scripts.e04_v3_full768_b1_remeasurement import load_config, read_jsonl, validate_full_raw


def percentile(values: list[float], fraction: float) -> float:
    values = sorted(values)
    return values[int(round((len(values) - 1) * fraction))]


def _scored(raw_path: Path, prompts: list[dict[str, Any]], targets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    raw = validate_full_raw(raw_path, prompts)
    rows: list[dict[str, Any]] = []
    for prediction, truth in zip(raw, targets, strict=True):
        rows.append({"row_index": prediction["row_index"], "canonical_base_id": truth["canonical_base_id"],
                     "condition": truth["condition"], "reserved_cell": truth["reserved_cell"],
                     "parser_valid": prediction["parser_valid"], "exact_grid": prediction["parsed_grid"] == truth["target"]})
    return rows


def _summarize(rows: list[dict[str, Any]], reference: list[dict[str, Any]]) -> dict[str, Any]:
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    ref_by_index = {row["row_index"]: row for row in reference}
    by_base: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    cells: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        ref = ref_by_index.get(row["row_index"])
        if ref is None or ref["condition"] != row["condition"] or ref["canonical_base_id"] != row["canonical_base_id"]:
            fail("E04C_REFERENCE_ALIGNMENT")
        by_condition[row["condition"]].append(row)
        by_base[row["canonical_base_id"]][row["condition"]].append(float(row["exact_grid"]) - float(ref["exact_grid"]))
        cells[json.dumps(row["reserved_cell"], sort_keys=True)].add(row["canonical_base_id"])
    conditions = ("ROTATION_TARGET", "FIXED_TURN_ROTATION_CONTROL", "MARKER_BINDING_CONTROL", "NO_TRANSFORM_RETENTION_CONTROL")
    if set(by_condition) != set(conditions) or any(len(by_condition[key]) != 192 for key in conditions):
        fail("E04C_EVALUATION_CONDITION_COVERAGE")
    units: dict[str, dict[str, float]] = {}
    for base, values in by_base.items():
        if set(values) != set(conditions) or any(len(values[key]) != 4 for key in conditions):
            fail("E04C_EVALUATION_BASE_COVERAGE")
        units[base] = {key: sum(values[key]) / 4.0 for key in conditions}
    if len(units) != 48 or any(len(value) != 12 for value in cells.values()):
        fail("E04C_EVALUATION_CELL_COVERAGE")
    accuracy = {key: sum(float(row["exact_grid"]) for row in by_condition[key]) / 192.0 for key in conditions}
    parser = {key: sum(float(row["parser_valid"]) for row in by_condition[key]) / 192.0 for key in conditions}
    rng = random.Random(20261011); cell_units = [sorted(values) for _, values in sorted(cells.items())]
    delta: dict[str, dict[str, float]] = {}
    for condition in conditions:
        point = sum(units[base][condition] for base in units) / len(units)
        draws = []
        for _ in range(10_000):
            selected = [rng.choice(cell) for cell in cell_units for _ in range(len(cell))]
            draws.append(sum(units[base][condition] for base in selected) / len(selected))
        delta[condition] = {"accuracy_delta_vs_v7": point, "episode_delta_vs_v7": point * 192.0,
                            "ci95_low": percentile(draws, .025), "ci95_high": percentile(draws, .975)}
    return {"condition_exact_grid_accuracy": accuracy, "condition_parser_validity": parser,
            "delta_vs_v7_by_condition": delta, "canonical_bases": len(units)}


def score(control_raw: Path, treatment_raw: Path, freeze_receipt: Path, output: Path, *, binding_path: Path | None = None) -> dict[str, Any]:
    if output.exists(): fail("E04C_SCORE_OUTPUT_FRESH")
    receipt = json.loads(freeze_receipt.read_text(encoding="utf-8"))
    if receipt.get("status") != "TRAINING_AND_RAW_GENERATIONS_FROZEN_NO_TARGETS" or receipt.get("target_sidecar_accessed") is not False:
        fail("E04C_RAW_FREEZE_REQUIRED")
    if receipt.get("raw_sha256") != {ARMS[0]: sha_path(control_raw), ARMS[1]: sha_path(treatment_raw)}:
        fail("E04C_RAW_FREEZE_IDENTITY")
    config = load_config(); prompts = read_jsonl(ROOT / config["prompt_path"])
    # The three raw files are fully validated before this code reads a target.
    if binding_path is None: fail("E04C_BINDING_REQUIRED")
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    reference_path = ROOT / binding["v7_reference_raw_path"]
    if sha_path(reference_path) != binding["v7_reference_raw_sha256"]:
        fail("E04C_V7_REFERENCE_IDENTITY")
    sidecar = ROOT / config["target_sidecar_path"]
    if sha_path(sidecar) != config["target_sidecar_sha256"]:
        fail("E04C_TARGET_SIDECAR_IDENTITY")
    targets = read_jsonl(sidecar)
    if len(targets) != 768 or [row.get("row_index") for row in targets] != list(range(768)):
        fail("E04C_TARGET_MAPPING")
    reference = _scored(reference_path, prompts, targets)
    control = _summarize(_scored(control_raw, prompts, targets), reference)
    treatment = _summarize(_scored(treatment_raw, prompts, targets), reference)
    primary = treatment["delta_vs_v7_by_condition"]["FIXED_TURN_ROTATION_CONTROL"]
    protected = (treatment["delta_vs_v7_by_condition"]["MARKER_BINDING_CONTROL"], treatment["delta_vs_v7_by_condition"]["NO_TRANSFORM_RETENTION_CONTROL"])
    valid_primary = primary["episode_delta_vs_v7"] >= 8 and primary["ci95_low"] > 0
    protection_failure = any(row["episode_delta_vs_v7"] < -8 for row in protected)
    rotation = treatment["delta_vs_v7_by_condition"]["ROTATION_TARGET"]
    if protection_failure: decision = "PROTECTION_FAILURE"
    elif valid_primary and rotation["ci95_low"] > 0: decision = "IMPROVED"
    elif valid_primary: decision = "ATOMIC_GAIN_WITHOUT_TRANSFER"
    else: decision = "INCONCLUSIVE"
    result = {"schema_version": 1, "protocol_id": binding["protocol_id"], "status": "SCORED_DEVELOPMENT_EVIDENCE",
              "scientific_scope": "REUSED_E04_DEVELOPMENT_COHORT_NOT_INDEPENDENT_ARC_GENERALIZATION_CONFIRMATION",
              "binding_sha256": sha_path(binding_path), "raw_freeze_receipt_sha256": sha_path(freeze_receipt),
              "control_raw_sha256": sha_path(control_raw), "treatment_raw_sha256": sha_path(treatment_raw),
              "v7_reference_raw_sha256": sha_path(reference_path), "target_scorer_sidecar_sha256": sha_path(sidecar),
              "target_sidecar_opened_only_after_all_raw_freeze_validation": True, "rows": 192, "bootstrap_replicates": 10000,
              "bootstrap_seed": 20261011, "control_vs_v7": control, "treatment_vs_v7": treatment,
              "decision": decision, "primary_rule": {"condition": "FIXED_TURN_ROTATION_CONTROL", "minimum_episode_gain": 8,
              "bootstrap_ci_lower_gt_zero": primary["ci95_low"] > 0, "observed": primary},
              "protected_rule": {"maximum_episode_decline": 8, "observed": {"MARKER_BINDING_CONTROL": protected[0], "NO_TRANSFORM_RETENTION_CONTROL": protected[1]}},
              "final_audit_opened": False}
    atomic_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--control-raw", type=Path, required=True); parser.add_argument("--treatment-raw", type=Path, required=True)
    parser.add_argument("--raw-freeze-receipt", type=Path, required=True); parser.add_argument("--binding", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); print(json.dumps(score(args.control_raw, args.treatment_raw, args.raw_freeze_receipt, args.output, binding_path=args.binding), sort_keys=True))


if __name__ == "__main__":
    main()
