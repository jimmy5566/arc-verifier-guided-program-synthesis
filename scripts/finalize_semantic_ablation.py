"""Local-only deterministic scoring for frozen semantic-ablation artifacts.

This is the sole scoring boundary for Tracks A--C.  It must never be packed
into the Kaggle source attachment because it accepts private oracle IR.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluation.semantic_ir_scorer import score


METRIC_KEYS = (
    "primary_family_correct", "core_operation_correct", "object_role_f1", "relation_f1",
    "selection_criteria_correct", "iteration_detection_correct", "iteration_semantic_correct",
    "conditional_logic_correct", "composition_correct", "output_construction_correct",
    "semantic_parameter_correct", "full_exact", "semantic_success", "slot_f1",
)


def _records(path: Path, condition: str) -> Mapping[str, Mapping[str, Any]]:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if artifact.get("status") != "PREDICTIONS_FROZEN_BEFORE_GOLD_SCORING":
        raise ValueError(f"{path} is not a frozen prediction artifact")
    records = artifact["records"]
    if condition in records and isinstance(records[condition], Mapping):
        return records[condition]
    # V1's historical artifact contains only one condition at the top level.
    if all(isinstance(value, Mapping) and "prediction" in value for value in records.values()):
        return records
    raise ValueError(f"condition {condition} absent from {path}")


def score_condition(records: Mapping[str, Mapping[str, Any]], gold: Mapping[str, Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    if set(records) != set(gold):
        raise ValueError("prediction/gold task sets differ")
    scored = {task_id: score(record.get("prediction"), gold[task_id]) for task_id, record in records.items()}
    total = len(scored)
    rate = lambda key: sum(bool(item.get(key)) for item in scored.values()) / total
    metrics = {
        "schema_validity": rate("schema_valid"),
        "primary_family_accuracy": rate("primary_family_correct"),
        "core_operation_accuracy": rate("core_operation_correct"),
        "object_role_f1": sum(float(item.get("object_role_f1", 0)) for item in scored.values()) / total,
        "relation_f1": sum(float(item.get("relation_f1", 0)) for item in scored.values()) / total,
        "selection_criteria_accuracy": rate("selection_criteria_correct"),
        "iteration_detection_accuracy": rate("iteration_detection_correct"),
        "iteration_semantic_accuracy": rate("iteration_semantic_correct"),
        "conditional_logic_accuracy": rate("conditional_logic_correct"),
        "composition_accuracy": rate("composition_correct"),
        "output_construction_accuracy": rate("output_construction_correct"),
        "semantic_parameter_accuracy": rate("semantic_parameter_correct"),
        "full_semantic_ir_exact": rate("full_exact"),
        "semantic_ir_success": rate("semantic_success"),
        "slot_f1": sum(float(item.get("slot_f1", 0)) for item in scored.values()) / total,
    }
    failure = Counter(name for item in scored.values() for name in item.get("failure_taxonomy", ()))
    failure["IR_SCHEMA_FAILURE"] += sum(not bool(item.get("schema_valid")) for item in scored.values())
    families: dict[str, dict[str, Any]] = {}
    for family in sorted({item["primary_family"] for item in gold.values()}):
        items = [scored[task_id] for task_id in gold if gold[task_id]["primary_family"] == family]
        families[family] = {
            "tasks": len(items),
            "semantic_success": sum(bool(item.get("semantic_success")) for item in items),
            "primary_family_correct": sum(bool(item.get("primary_family_correct")) for item in items),
            "relation_f1": sum(float(item.get("relation_f1", 0)) for item in items) / len(items),
            "iteration_detection": sum(bool(item.get("iteration_detection_correct")) for item in items),
            "iteration_semantics": sum(bool(item.get("iteration_semantic_correct")) for item in items),
            "conditional_detection": sum(bool(item.get("conditional_logic_correct")) for item in items),
        }
    return {"task_count": total, "metrics": metrics, "failure_taxonomy": dict(sorted(failure.items())), "generation_seconds": sum(float(record.get("generation_seconds", 0)) for record in records.values()), "request_count": sum(int(record.get("request_count", 1)) for record in records.values()), "family_metrics": families}, scored


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    gold = json.loads(args.gold.read_text(encoding="utf-8"))["gold"]
    conditions: dict[str, dict[str, Any]] = {}
    private_scores: dict[str, dict[str, Any]] = {}
    for item in args.condition:
        name, delimiter, raw_path = item.partition("=")
        if not delimiter or not name or not raw_path:
            raise ValueError("--condition requires NAME=PATH")
        path = Path(raw_path)
        records = _records(path, name)
        result, scored = score_condition(records, gold)
        conditions[name] = {**result, "prediction_freeze_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        private_scores[name] = {task_id: {"semantic_success": bool(value.get("semantic_success")), "schema_valid": bool(value.get("schema_valid"))} for task_id, value in scored.items()}
    output = {
        "status": "SCORED_AFTER_PREDICTION_FREEZE",
        "task_count": len(gold),
        "oracle_gold_sha256": hashlib.sha256(args.gold.read_bytes()).hexdigest(),
        "conditions": conditions,
        "leakage_audit": "gold opened by this local-only finalizer after frozen artifacts; aggregate excludes task IDs, grids, targets, oracle IR, and raw model responses",
    }
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    # This private sidecar enables deterministic interface-recovery calculations
    # in the later track finalizer, while keeping individual task identifiers out
    # of the public result.
    sidecar = args.output.with_name(args.output.stem + "_private_scores.json")
    sidecar.write_text(json.dumps(private_scores, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"conditions": sorted(conditions), "task_count": len(gold)}, sort_keys=True))


if __name__ == "__main__":
    main()
