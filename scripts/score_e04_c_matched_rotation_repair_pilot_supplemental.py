"""CPU-only supplemental paired scoring for the immutable E04-C Run003 raw outputs."""
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
from scripts.e04_c_matched_rotation_execution import ARMS, E04CFailure, atomic_json, fail, sha_path
from scripts.e04_v3_full768_b1_remeasurement import load_config, read_jsonl, validate_full_raw
from scripts.score_e04_c_matched_rotation_repair_pilot import percentile

CONDITIONS = ("FIXED_TURN_ROTATION_CONTROL", "MARKER_BINDING_CONTROL", "NO_TRANSFORM_RETENTION_CONTROL", "ROTATION_TARGET")
SEED = 20261011
REPLICATES = 10_000
ROWS_PER_CONDITION = 192


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validated_rows(raw_path: Path, prompts: list[dict[str, Any]], targets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    raw = validate_full_raw(raw_path, prompts)
    if len(raw) != len(targets):
        fail("E04C_SUPPLEMENTAL_ROW_COUNT")
    rows: list[dict[str, Any]] = []
    for prediction, truth in zip(raw, targets, strict=True):
        if prediction["row_index"] != truth.get("row_index"):
            fail("E04C_SUPPLEMENTAL_TARGET_ALIGNMENT")
        rows.append({"row_index": prediction["row_index"], "canonical_base_id": truth["canonical_base_id"],
                     "condition": truth["condition"], "exact_grid": prediction["parsed_grid"] == truth["target"],
                     "parser_valid": prediction["parser_valid"]})
    return rows


def paired_contrasts(treatment: list[dict[str, Any]], control: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Canonical-base paired, percentile-bootstrap contrasts: treatment minus control."""
    def collect(rows: list[dict[str, Any]]) -> dict[str, dict[str, list[float]]]:
        grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for row in rows:
            if row["condition"] not in CONDITIONS:
                fail("E04C_SUPPLEMENTAL_CONDITION")
            grouped[row["canonical_base_id"]][row["condition"]].append(float(row["exact_grid"]))
        if len(grouped) != 48:
            fail("E04C_SUPPLEMENTAL_CANONICAL_BASE_COUNT")
        for values in grouped.values():
            if set(values) != set(CONDITIONS) or any(len(values[c]) != 4 for c in CONDITIONS):
                fail("E04C_SUPPLEMENTAL_CANONICAL_BASE_COVERAGE")
        return grouped
    t, c = collect(treatment), collect(control)
    if set(t) != set(c):
        fail("E04C_SUPPLEMENTAL_PAIRED_BASES")
    bases = sorted(t)
    unit = {base: {condition: sum(t[base][condition]) / 4.0 - sum(c[base][condition]) / 4.0 for condition in CONDITIONS} for base in bases}
    rng = random.Random(SEED)
    draws = {condition: [] for condition in CONDITIONS}
    for _ in range(REPLICATES):
        selected = [rng.choice(bases) for _ in bases]
        for condition in CONDITIONS:
            draws[condition].append(sum(unit[base][condition] for base in selected) / len(selected))
    return {condition: {"accuracy_delta_treatment_minus_control": sum(unit[base][condition] for base in bases) / len(bases),
                        "episode_delta_treatment_minus_control": sum(unit[base][condition] for base in bases) / len(bases) * ROWS_PER_CONDITION,
                        "ci95_low": percentile(draws[condition], .025), "ci95_high": percentile(draws[condition], .975)}
            for condition in CONDITIONS}


def score(control_raw: Path, treatment_raw: Path, freeze_receipt: Path, binding: Path, original_score: Path, original_package: Path, output: Path) -> dict[str, Any]:
    if output.exists(): fail("E04C_SUPPLEMENTAL_OUTPUT_FRESH")
    receipt = _read(freeze_receipt)
    if receipt.get("status") != "TRAINING_AND_RAW_GENERATIONS_FROZEN_NO_TARGETS" or receipt.get("target_sidecar_accessed") is not False:
        fail("E04C_SUPPLEMENTAL_RAW_FREEZE")
    expected = {ARMS[0]: sha_path(control_raw), ARMS[1]: sha_path(treatment_raw)}
    if receipt.get("raw_sha256") != expected:
        fail("E04C_SUPPLEMENTAL_RAW_IDENTITY")
    config = load_config(); prompts = read_jsonl(ROOT / config["prompt_path"])
    # Validate all prediction bytes before loading the sealed target-sidecar bytes.
    validate_full_raw(control_raw, prompts); validate_full_raw(treatment_raw, prompts)
    if sha_path(binding) != _read(original_score).get("binding_sha256"):
        fail("E04C_SUPPLEMENTAL_BINDING_IDENTITY")
    original = _read(original_score); package = _read(original_package)
    if original.get("decision") != "PROTECTION_FAILURE" or package.get("score_decision") != "PROTECTION_FAILURE":
        fail("E04C_SUPPLEMENTAL_ORIGINAL_PROTECTION_DECISION")
    sidecar = ROOT / config["target_sidecar_path"]
    if sha_path(sidecar) != config["target_sidecar_sha256"]:
        fail("E04C_SUPPLEMENTAL_TARGET_SIDECAR_IDENTITY")
    targets = read_jsonl(sidecar)
    if len(targets) != 768 or [row.get("row_index") for row in targets] != list(range(768)):
        fail("E04C_SUPPLEMENTAL_TARGET_MAPPING")
    contrasts = paired_contrasts(_validated_rows(treatment_raw, prompts, targets), _validated_rows(control_raw, prompts, targets))
    primary = contrasts["FIXED_TURN_ROTATION_CONTROL"]
    primary_pass = primary["episode_delta_treatment_minus_control"] >= 8 and primary["ci95_low"] > 0
    protection = original["protected_rule"]
    result = {"schema_version": 1, "protocol_id": "E04_C_MATCHED_FIXED_TURN_ROTATION_REPAIR_PILOT_V1",
              "status": "CORRECTED_SUPPLEMENTAL_CPU_SCORE", "scientific_scope": "REUSED_E04_DEVELOPMENT_COHORT_NOT_INDEPENDENT_ARC_GENERALIZATION_CONFIRMATION",
              "control_raw_sha256": sha_path(control_raw), "treatment_raw_sha256": sha_path(treatment_raw),
              "raw_freeze_receipt_sha256": sha_path(freeze_receipt), "binding_sha256": sha_path(binding),
              "original_cpu_score_sha256": sha_path(original_score), "original_result_package_sha256": sha_path(original_package),
              "target_scorer_sidecar_sha256": sha_path(sidecar), "target_sidecar_opened_only_after_all_raw_validation": True,
              "bootstrap": {"canonical_base_pairs": 48, "seed": SEED, "replicates": REPLICATES, "interval": "percentile_95"},
              "treatment_minus_control": contrasts,
              "primary_rule": {"condition": "FIXED_TURN_ROTATION_CONTROL", "minimum_additional_correct_episodes": 8,
                               "lower_bound_greater_than_zero": primary["ci95_low"] > 0, "observed": primary, "passes_primary_rule": primary_pass},
              "frozen_treatment_minus_v7_protection_rule_unchanged": protection,
              "final_protocol_decision": "PROTECTION_FAILURE", "primary_cannot_override_protection_failure": True,
              "final_audit_opened": False}
    atomic_json(output, result)
    return result


def self_test() -> None:
    rows_t=[]; rows_c=[]
    for base in range(48):
        for condition in CONDITIONS:
            for repeat in range(4):
                rows_c.append({"canonical_base_id": str(base), "condition": condition, "exact_grid": False, "parser_valid": True})
                rows_t.append({"canonical_base_id": str(base), "condition": condition, "exact_grid": condition == "FIXED_TURN_ROTATION_CONTROL", "parser_valid": True})
    a=paired_contrasts(rows_t, rows_c); b=paired_contrasts(rows_t, rows_c)
    assert a == b and a["FIXED_TURN_ROTATION_CONTROL"]["episode_delta_treatment_minus_control"] == 192.0
    assert all(a[c]["episode_delta_treatment_minus_control"] == 0.0 for c in CONDITIONS[1:])
    print("PASS_E04C_SUPPLEMENTAL_SELF_TEST")


def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument("--control-raw", type=Path); ap.add_argument("--treatment-raw", type=Path); ap.add_argument("--raw-freeze-receipt", type=Path); ap.add_argument("--binding", type=Path); ap.add_argument("--original-score", type=Path); ap.add_argument("--original-package", type=Path); ap.add_argument("--output", type=Path); ap.add_argument("--self-test", action="store_true")
    a=ap.parse_args()
    if a.self_test: self_test(); return
    if not all([a.control_raw,a.treatment_raw,a.raw_freeze_receipt,a.binding,a.original_score,a.original_package,a.output]): raise SystemExit("E04C_SUPPLEMENTAL_ARGUMENTS")
    print(json.dumps(score(a.control_raw,a.treatment_raw,a.raw_freeze_receipt,a.binding,a.original_score,a.original_package,a.output),sort_keys=True))
if __name__ == "__main__": main()