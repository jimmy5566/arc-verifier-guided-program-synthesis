"""Deterministic post-result E4-C error decomposition.

This is intentionally a CPU-only consumer of the frozen E4-C raw prediction
records and the sidecar opened after their raw freeze.  It imports the
independent connected-component semantic oracle, never imports a model or a
generator, and does not write targets outside the existing sidecar boundary.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.connected_components_object_extraction_semantic_oracle import (
    connected_component_count,
    validate_demonstrations,
)

ARMS = ("V7", "EQUAL_SLOT_5E5", "QUERY_LOCAL_CODE_PAIR_5E5")
OOD_SPLIT = "OOD_INDEPENDENT"


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def rectangular_int_grid(value: object) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(row, list) and bool(row) for row in value)
        and len({len(row) for row in value}) == 1
        and all(isinstance(cell, int) and 0 <= cell <= 9 for row in value for cell in row)
    )


def count_output_signature(grid: list[list[int]], query_color: int) -> dict:
    """Independent signature for the observable Count-output construction.

    A semantically constructed Count answer is a 3x3 grid containing the
    query color as a left-aligned horizontal prefix on row 1.  Its prefix
    length is the component-count estimate.  This defines construction
    separately from whether that estimate is the true count.
    """
    if not rectangular_int_grid(grid):
        return {"well_formed_grid": False, "construction_valid": False, "predicted_count": None}
    nonzero = [(y, x, value) for y, row in enumerate(grid) for x, value in enumerate(row) if value]
    predicted_count = len(nonzero)
    valid = (
        len(grid) == 3
        and len(grid[0]) == 3
        and 1 <= predicted_count <= 3
        and all(value == query_color for _, _, value in nonzero)
        and {(y, x) for y, x, _ in nonzero} == {(1, x) for x in range(predicted_count)}
    )
    return {
        "well_formed_grid": True,
        "construction_valid": valid,
        "predicted_count": predicted_count,
    }


def classify_prediction(raw: dict, target: list[list[int]], query_color: int, true_count: int) -> dict:
    """Give every record an exhaustive primary class and independently check exactness."""
    if raw.get("parser_valid") is not True or not rectangular_int_grid(raw.get("parsed_grid")):
        primary = "PARSER_OR_FORMAT_FAILURE"
        signature = {"well_formed_grid": False, "construction_valid": False, "predicted_count": None}
        exact = False
    else:
        signature = count_output_signature(raw["parsed_grid"], query_color)
        exact = raw["parsed_grid"] == target
        if exact:
            primary = "EXACT"
        elif signature["construction_valid"] and signature["predicted_count"] != true_count:
            primary = "WRONG_COMPONENT_COUNT_VALID_CONSTRUCTION"
        elif signature["predicted_count"] == true_count:
            primary = "CORRECT_COMPONENT_COUNT_WRONG_CONSTRUCTION"
        else:
            # This residual is necessary to keep the required classifications
            # exhaustive: it has both a wrong count and malformed construction.
            primary = "WRONG_COMPONENT_COUNT_WRONG_CONSTRUCTION"
    return {
        "primary": primary,
        "exact": exact,
        **signature,
        "true_count": true_count,
        "query_color": query_color,
    }


def independent_oracle_checks(records: list[dict], targets: dict[str, list[list[int]]]) -> dict:
    checked = 0
    for row in records:
        task = row["prompt"]
        validate_demonstrations(task, "CONNECTED_COMPONENT_COUNT")
        expected = connected_component_count(task["test"][0]["input"])
        if expected != targets.get(row["record_id"]):
            raise ValueError(f"SIDECAR_ORACLE_MISMATCH:{row['record_id']}")
        if expected[1].count(row["query_color"]) != row["query_count"]:
            raise ValueError(f"QUERY_METADATA_ORACLE_MISMATCH:{row['record_id']}")
        checked += 1
    return {
        "status": "PASS_INDEPENDENT_SEMANTIC_ORACLE",
        "records": checked,
        "demonstrations_validated": checked * 3,
        "oracle": "scripts.connected_components_object_extraction_semantic_oracle.connected_component_count",
    }


def summarize(rows: list[dict]) -> dict:
    primary = Counter(row["classification"]["primary"] for row in rows)
    return {
        "records": len(rows),
        "primary_classes": {key: primary.get(key, 0) for key in (
            "EXACT",
            "WRONG_COMPONENT_COUNT_VALID_CONSTRUCTION",
            "CORRECT_COMPONENT_COUNT_WRONG_CONSTRUCTION",
            "WRONG_COMPONENT_COUNT_WRONG_CONSTRUCTION",
            "PARSER_OR_FORMAT_FAILURE",
        )},
    }


def transition(left: list[dict], right: list[dict]) -> dict:
    left_by_id = {row["record_id"]: row for row in left}
    right_by_id = {row["record_id"]: row for row in right}
    if set(left_by_id) != set(right_by_id):
        raise ValueError("ARM_RECORD_IDENTITY_MISMATCH")
    values = Counter()
    for record_id in sorted(left_by_id):
        before = left_by_id[record_id]
        after = right_by_id[record_id]
        values[f"{before['classification']['primary']}->{after['classification']['primary']}"] += 1
    return dict(sorted(values.items()))


def unit_order_summary(rows: list[dict]) -> dict:
    by_unit: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        by_unit[row["canonical_unit_id"]][row["demo_order"]] = row
    if len(by_unit) != 48 or any(set(pair) != {"A", "B"} for pair in by_unit.values()):
        raise ValueError("CANONICAL_ORDER_GRID_INVALID")
    unstable = []
    for unit, pair in sorted(by_unit.items()):
        if pair["A"]["classification"]["exact"] != pair["B"]["classification"]["exact"]:
            unstable.append(unit)
    return {
        "canonical_units": len(by_unit),
        "order_conditional_instability_units": len(unstable),
        "stable_units": len(by_unit) - len(unstable),
        "unit_ids": unstable,
    }


def grouped_summary(rows: list[dict], field: str) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row[field])].append(row)
    return {key: summarize(value) for key, value in sorted(groups.items())}


def grouped_transition(left: list[dict], right: list[dict], field: str) -> dict:
    left_group: dict[str, list[dict]] = defaultdict(list)
    right_group: dict[str, list[dict]] = defaultdict(list)
    for row in left:
        left_group[str(row[field])].append(row)
    for row in right:
        right_group[str(row[field])].append(row)
    if set(left_group) != set(right_group):
        raise ValueError("TRANSITION_GROUP_IDENTITY_MISMATCH")
    return {key: transition(left_group[key], right_group[key]) for key in sorted(left_group)}


def mechanism_selection(arm_results: dict[str, list[dict]], order: dict[str, dict]) -> dict:
    """Apply the Director-provided decision rule deterministically.

    The three categories are counted only among exact-grid failures.  Order
    instability is a canonical-unit property, so it is compared on the 48
    units rather than double-counted as two record failures.
    """
    category_totals = Counter()
    failure_records = 0
    for rows in arm_results.values():
        for row in rows:
            category = row["classification"]["primary"]
            if category != "EXACT":
                failure_records += 1
                category_totals[category] += 1
    unstable_units = sum(value["order_conditional_instability_units"] for value in order.values())
    # The intervention selection compares the specified evidence types.
    wrong_count_valid = category_totals["WRONG_COMPONENT_COUNT_VALID_CONSTRUCTION"]
    correct_count_bad_construction = category_totals["CORRECT_COMPONENT_COUNT_WRONG_CONSTRUCTION"]
    if unstable_units > max(wrong_count_valid, correct_count_bad_construction):
        selected = "RULE_GROUNDING_ORDER_ROBUSTNESS"
        rationale = "order-conditional instability has the largest specified failure signal across the three arms"
    elif wrong_count_valid > correct_count_bad_construction:
        selected = "OBJECT_CENTRIC_COMPONENT_REPRESENTATION_GROUNDING"
        rationale = "wrong component counts with otherwise valid Count construction predominate over construction-only errors"
    else:
        selected = "OUTPUT_CONSTRUCTION_INTERVENTION"
        rationale = "correct component count with wrong output construction is at least as prevalent as the competing specified count signal"
    return {
        "exact_grid_failure_records": failure_records,
        "specified_category_totals": {
            "wrong_component_count_valid_construction": wrong_count_valid,
            "correct_component_count_wrong_construction": correct_count_bad_construction,
            "parser_or_format_failure": category_totals["PARSER_OR_FORMAT_FAILURE"],
            "order_conditional_instability_canonical_units_across_arms": unstable_units,
        },
        "selected_mechanism": selected,
        "selection_rationale": rationale,
        "selection_rule": "Director response: wrong-count-valid-construction -> object-centric; correct-count-wrong-construction -> output construction; order instability dominates -> rule-grounding/order robustness.",
    }


def self_test() -> dict:
    target = [[0, 0, 0], [4, 4, 0], [0, 0, 0]]
    base = {"parser_valid": True}
    assert classify_prediction({**base, "parsed_grid": target}, target, 4, 2)["primary"] == "EXACT"
    assert classify_prediction({**base, "parsed_grid": [[0, 0, 0], [4, 0, 0], [0, 0, 0]]}, target, 4, 2)["primary"] == "WRONG_COMPONENT_COUNT_VALID_CONSTRUCTION"
    assert classify_prediction({**base, "parsed_grid": [[4, 4, 0], [0, 0, 0], [0, 0, 0]]}, target, 4, 2)["primary"] == "CORRECT_COMPONENT_COUNT_WRONG_CONSTRUCTION"
    assert classify_prediction({**base, "parsed_grid": [[4, 0, 0], [0, 0, 0], [0, 0, 0]]}, target, 4, 2)["primary"] == "WRONG_COMPONENT_COUNT_WRONG_CONSTRUCTION"
    assert classify_prediction({"parser_valid": False, "parsed_grid": None}, target, 4, 2)["primary"] == "PARSER_OR_FORMAT_FAILURE"
    return {"status": "PASS_E4C_ERROR_DECOMPOSITION_UNIT_TEST", "model_loaded": False, "gpu_used": False}


def run(evidence_dir: Path, ood_path: Path, output_path: Path) -> dict:
    sidecar_path = evidence_dir / "SYNTHETIC_TARGET_SIDECAR.json"
    required = [sidecar_path, ood_path, *[evidence_dir / f"{arm}_RAW.jsonl" for arm in ARMS]]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError("REQUIRED_FROZEN_INPUT_MISSING:" + ",".join(missing))
    sidecar = read_json(sidecar_path)
    if sidecar.get("status") != "DERIVED_AFTER_RAW_FREEZE" or sidecar.get("opened_after_raw_freeze") is not True:
        raise ValueError("SIDECAR_FREEZE_BOUNDARY_INVALID")
    ood = read_json(ood_path)
    records = [row for row in ood.get("records", []) if row.get("record_id", "").endswith((":ORDER_A", ":ORDER_B"))]
    if len(records) != 96 or len({row["canonical_unit_id"] for row in records}) != 48:
        raise ValueError("OOD_RECORD_GRID_INVALID")
    oracle = independent_oracle_checks(records, sidecar["targets"])
    metadata = {row["record_id"]: row for row in records}
    arms: dict[str, list[dict]] = {}
    inputs = {"sidecar": sidecar_path, "ood_inputs": ood_path}
    for arm in ARMS:
        raw_path = evidence_dir / f"{arm}_RAW.jsonl"
        inputs[f"{arm}_raw"] = raw_path
        raw = {row["record_id"]: row for row in read_jsonl(raw_path) if row.get("split") == OOD_SPLIT}
        if set(raw) != set(metadata) or len(raw) != 96:
            raise ValueError(f"RAW_OOD_IDENTITY_INVALID:{arm}")
        values = []
        for record_id in sorted(metadata):
            source = metadata[record_id]
            prediction = raw[record_id]
            if prediction.get("prompt_sha256") != source.get("prompt_sha256"):
                raise ValueError(f"RAW_PROMPT_IDENTITY_INVALID:{arm}:{record_id}")
            target = sidecar["targets"][record_id]
            values.append({
                "record_id": record_id,
                "canonical_unit_id": source["canonical_unit_id"],
                "demo_order": source["demo_order"],
                "true_count": source["query_count"],
                "topology": source["topology"],
                "classification": classify_prediction(prediction, target, source["query_color"], source["query_count"]),
            })
        arms[arm] = values
    order = {arm: unit_order_summary(rows) for arm, rows in arms.items()}
    result = {
        "schema_version": "E4C_POST_RESULT_PAIRED_ERROR_DECOMPOSITION_V1",
        "status": "PASS_CPU_ONLY_FROZEN_RAW_SIDECAR_DECOMPOSITION",
        "scope": {
            "model_loaded": False,
            "gpu_used": False,
            "training": False,
            "inference": False,
            "target_generation": False,
            "checkpoint_promotion": False,
            "sealed_data_access": False,
            "sidecar_already_opened_after_raw_freeze": True,
        },
        "inputs": {name: {"path": str(path).replace("\\", "/"), "sha256": sha256_file(path)} for name, path in sorted(inputs.items())},
        "independent_semantic_oracle": oracle,
        "classification_definition": {
            "valid_count_construction": "3x3 grid, only the query color, and a left-aligned nonzero prefix on row 1; prefix length is the predicted component count.",
            "primary_classes": [
                "EXACT",
                "WRONG_COMPONENT_COUNT_VALID_CONSTRUCTION",
                "CORRECT_COMPONENT_COUNT_WRONG_CONSTRUCTION",
                "WRONG_COMPONENT_COUNT_WRONG_CONSTRUCTION",
                "PARSER_OR_FORMAT_FAILURE",
            ],
            "order_conditional_instability": "canonical unit has exactly one of demonstration orders exact; reported separately to avoid double-counting record-level failures.",
        },
        "by_arm": {
            arm: {
                "overall": summarize(rows),
                "by_true_count": grouped_summary(rows, "true_count"),
                "by_topology": grouped_summary(rows, "topology"),
                "order": order[arm],
            }
            for arm, rows in arms.items()
        },
        "transitions": {
            "V7_TO_EQUAL_SLOT_5E5": {
                "overall": transition(arms["V7"], arms["EQUAL_SLOT_5E5"]),
                "by_true_count": grouped_transition(arms["V7"], arms["EQUAL_SLOT_5E5"], "true_count"),
                "by_topology": grouped_transition(arms["V7"], arms["EQUAL_SLOT_5E5"], "topology"),
            },
            "EQUAL_SLOT_5E5_TO_QUERY_LOCAL_CODE_PAIR_5E5": {
                "overall": transition(arms["EQUAL_SLOT_5E5"], arms["QUERY_LOCAL_CODE_PAIR_5E5"]),
                "by_true_count": grouped_transition(arms["EQUAL_SLOT_5E5"], arms["QUERY_LOCAL_CODE_PAIR_5E5"], "true_count"),
                "by_topology": grouped_transition(arms["EQUAL_SLOT_5E5"], arms["QUERY_LOCAL_CODE_PAIR_5E5"], "topology"),
            },
        },
        "mechanism_selection": mechanism_selection(arms, order),
        "record_classifications": arms,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(canonical(result) + b"\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-dir", type=Path)
    parser.add_argument("--ood-inputs", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), sort_keys=True))
        return
    if not args.evidence_dir or not args.ood_inputs or not args.output:
        raise SystemExit("--evidence-dir, --ood-inputs, and --output are required")
    print(json.dumps(run(args.evidence_dir, args.ood_inputs, args.output)["mechanism_selection"], sort_keys=True))


if __name__ == "__main__":
    main()
