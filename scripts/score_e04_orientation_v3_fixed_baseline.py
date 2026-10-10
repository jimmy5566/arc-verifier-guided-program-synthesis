"""CPU scorer for E04 V3 after target-blind raw generation is frozen."""
from __future__ import annotations
import argparse, json, random, sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.e04_orientation_v3_no_update import (
    CONDITIONS, CONTRACT, E04ExecutionFailure, OUT, atomic_json, read_jsonl, sha_path, validate_contract, validate_raw_generation,
)

SIDECAR = OUT / "TARGET_SCORER_SIDECAR.jsonl"
MANIFEST = OUT / "MANIFEST.json"

def fail(code: str) -> None:
    raise E04ExecutionFailure(code)

def percentile(values: list[float], p: float) -> float:
    values = sorted(values)
    index = int(round((len(values) - 1) * p))
    return values[index]

def score(raw_path: Path, b1_path: Path, raw_receipt_path: Path, destination: Path) -> dict[str, Any]:
    contract_validation = validate_contract(CONTRACT)
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    receipt = json.loads(raw_receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "RAW_GENERATIONS_FROZEN_NO_TARGETS" or receipt.get("target_sidecar_accessed") is not False:
        fail("E04_RAW_FREEZE_RECEIPT_REQUIRED")
    if receipt.get("primary_raw_sha256") != sha_path(raw_path) or receipt.get("batch1_raw_sha256") != sha_path(b1_path):
        fail("E04_RAW_FREEZE_IDENTITY")
    prompts = read_jsonl(ROOT / contract["input_prompts_path"])
    primary = validate_raw_generation(raw_path, list(range(768)))
    subset = json.loads((ROOT / contract["batch1_sensitivity_manifest_path"]).read_text(encoding="utf-8"))
    b1 = validate_raw_generation(b1_path, subset["row_indexes"], expected_input_sha=subset["row_input_sha256"])
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if sha_path(SIDECAR) != manifest["target_scorer_sidecar_sha256"]:
        fail("E04_TARGET_SIDECAR_IDENTITY")
    sidecar = read_jsonl(SIDECAR)
    if len(sidecar) != 768 or [row.get("row_index") for row in sidecar] != list(range(768)):
        fail("E04_TARGET_SIDECAR_MAPPING")
    primary_by_index = {row["row_index"]: row for row in primary}
    b1_by_index = {row["row_index"]: row for row in b1}
    rows: list[dict[str, Any]] = []
    b1_disagreements: list[int] = []
    for truth in sidecar:
        index = truth["row_index"]
        prediction = primary_by_index[index]
        exact = prediction["parsed_grid"] == truth["target"]
        item = {
            "row_index": index, "canonical_base_id": truth["canonical_base_id"], "condition": truth["condition"],
            "reserved_cell": truth["reserved_cell"], "control_marker_turn": truth["control_marker_turn"],
            "parser_valid": prediction["parser_valid"], "exact_grid": exact,
        }
        if index in b1_by_index:
            reference = b1_by_index[index]
            b1_exact = reference["parsed_grid"] == truth["target"]
            same = (prediction["parser_valid"] == reference["parser_valid"]
                    and prediction["parsed_grid"] == reference["parsed_grid"] and exact == b1_exact)
            item["batch1_agreement"] = same
            if not same:
                b1_disagreements.append(index)
        rows.append(item)
    if len(rows) != 768:
        fail("E04_SCORING_ROW_COUNT")
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_base: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    by_cell: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        by_condition[row["condition"]].append(row)
        by_base[row["canonical_base_id"]][row["condition"]].append(row)
        by_cell[json.dumps(row["reserved_cell"], sort_keys=True)].add(row["canonical_base_id"])
    if set(by_condition) != set(CONDITIONS) or any(len(value) != 192 for value in by_condition.values()):
        fail("E04_CONDITION_COVERAGE")
    if any(len(bases) < 8 for bases in by_cell.values()):
        fail("E04_CONDITIONAL_SUPPORT")
    # All four marker turns are the matched observations for every base/condition.
    unit: dict[str, dict[str, float]] = {}
    for base, conditions in by_base.items():
        if set(conditions) != set(CONDITIONS) or any(len(values) != 4 for values in conditions.values()):
            fail("E04_MATCHED_UNIT_COVERAGE")
        unit[base] = {condition: sum(float(row["exact_grid"]) for row in values) / 4.0
                      for condition, values in conditions.items()}
    accuracy = {condition: sum(float(row["exact_grid"]) for row in values) / len(values)
                for condition, values in by_condition.items()}
    parser_validity = {condition: sum(float(row["parser_valid"]) for row in values) / len(values)
                       for condition, values in by_condition.items()}
    # Stratified bootstrap resamples canonical bases within the four held-out cells.
    rng = random.Random(20261010)
    units_by_cell: list[list[str]] = [sorted(bases) for _, bases in sorted(by_cell.items())]
    contrasts: dict[str, dict[str, float]] = {}
    for control in ("FIXED_TURN_ROTATION_CONTROL", "MARKER_BINDING_CONTROL", "NO_TRANSFORM_RETENTION_CONTROL"):
        point = sum(unit[base]["ROTATION_TARGET"] - unit[base][control] for base in unit) / len(unit)
        draws: list[float] = []
        for _ in range(10_000):
            sample = [rng.choice(cell) for cell in units_by_cell for _ in range(len(cell))]
            draws.append(sum(unit[base]["ROTATION_TARGET"] - unit[base][control] for base in sample) / len(sample))
        contrasts[control] = {"point": point, "ci95_low": percentile(draws, .025), "ci95_high": percentile(draws, .975)}
    result = {
        "schema_version": 1,
        "protocol_id": contract["protocol_id"],
        "status": "NUMERICAL_OR_BATCH_SENSITIVITY_INCONCLUSIVE" if b1_disagreements else "SCORED_DEVELOPMENT_EVIDENCE",
        "scientific_scope": "REUSED_DEVELOPMENT_COHORT_NOT_INDEPENDENT_ARC_GENERALIZATION_CONFIRMATION",
        "raw_freeze_receipt_sha256": sha_path(raw_receipt_path),
        "primary_raw_sha256": sha_path(raw_path), "batch1_raw_sha256": sha_path(b1_path),
        "target_scorer_sidecar_sha256": sha_path(SIDECAR), "target_sidecar_opened_only_after_raw_freeze": True,
        "rows": 768, "canonical_bases": 48, "minimum_support_per_reserved_cell": min(len(value) for value in by_cell.values()),
        "condition_exact_grid_accuracy": accuracy, "condition_parser_validity": parser_validity,
        "matched_rotation_target_minus_control_bootstrap": contrasts,
        "batch1_sensitivity_rows": 64, "batch1_disagreement_row_indexes": b1_disagreements,
        "batch1_gate": "PASS" if not b1_disagreements else "FAIL_INCONCLUSIVE",
        "forbidden_operations": {"optimizer_steps": 0, "parameter_updates": 0, "backward_calls": 0, "final_audit_opened": False},
        "contract_validation": contract_validation,
    }
    if destination.exists():
        fail("E04_SCORE_OUTPUT_NOT_FRESH")
    atomic_json(destination, result)
    return result

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-raw", type=Path, required=True)
    parser.add_argument("--batch1-raw", type=Path, required=True)
    parser.add_argument("--raw-freeze-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(score(args.primary_raw, args.batch1_raw, args.raw_freeze_receipt, args.output), sort_keys=True))

if __name__ == "__main__":
    main()
