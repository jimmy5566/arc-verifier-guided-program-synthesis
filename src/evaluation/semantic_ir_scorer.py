"""Deterministic, non-LLM scoring for ARCSEMANTICIRV1."""
from __future__ import annotations

from collections import Counter
from typing import Any, Mapping

from recognition.arc_semantic_ir import canonical, validate


def _f1(predicted: set[str], expected: set[str]) -> tuple[float, float, float]:
    precision = len(predicted & expected) / len(predicted) if predicted else (1.0 if not expected else 0.0)
    recall = len(predicted & expected) / len(expected) if expected else 1.0
    return precision, recall, 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def score(prediction: Mapping[str, Any] | None, oracle: Mapping[str, Any]) -> dict[str, Any]:
    if prediction is None or not validate(prediction)[0]:
        return {"schema_valid": False, "semantic_success": False, "full_exact": False, "error": "IR_SCHEMA_FAILURE", "slot_precision": 0.0, "slot_recall": 0.0, "slot_f1": 0.0}
    predicted, expected = canonical(prediction), canonical(oracle)
    operations, expected_operations = set(predicted["operations"]), set(expected["operations"])
    relations = _f1(set(predicted["relations"]) - {"NONE"}, set(expected["relations"]) - {"NONE"})
    roles = _f1(set(predicted["object_roles"]) - {"NONE"}, set(expected["object_roles"]) - {"NONE"})
    critical = [predicted["primary_family"] == expected["primary_family"], bool(operations & expected_operations), predicted["iteration"]["enabled"] == expected["iteration"]["enabled"], predicted["conditional_logic"]["enabled"] == expected["conditional_logic"]["enabled"]]
    if expected["iteration"]["enabled"]: critical += [predicted["iteration"][key] == expected["iteration"][key] for key in ("mode", "direction", "step_rule", "termination")]
    if expected["conditional_logic"]["enabled"]: critical += [predicted["conditional_logic"][key] == expected["conditional_logic"][key] for key in ("condition", "action")]
    fields = [predicted["primary_family"] == expected["primary_family"], operations == expected_operations, set(predicted["object_roles"]) == set(expected["object_roles"]), set(predicted["relations"]) == set(expected["relations"]), set(predicted["selection_criteria"]) == set(expected["selection_criteria"]), predicted["spatial_transform"] == expected["spatial_transform"], predicted["iteration"] == expected["iteration"], predicted["conditional_logic"] == expected["conditional_logic"], set(predicted["composition"]) == set(expected["composition"]), predicted["output_construction"] == expected["output_construction"], predicted["semantic_parameters"] == expected["semantic_parameters"]]
    failures = []
    if not fields[0]: failures.append("FAMILY_WRONG")
    if roles[2] < 1: failures.append("OBJECT_ROLE_WRONG")
    if relations[2] < 1: failures.append("RELATION_WRONG")
    if expected["iteration"]["enabled"] and not predicted["iteration"]["enabled"]: failures.append("ITERATION_MISSED")
    elif expected["iteration"]["enabled"] and predicted["iteration"] != expected["iteration"]: failures.append("ITERATION_PARAMETER_WRONG")
    if expected["conditional_logic"]["enabled"] and not predicted["conditional_logic"]["enabled"]: failures.append("CONDITION_MISSED")
    elif expected["conditional_logic"]["enabled"] and predicted["conditional_logic"] != expected["conditional_logic"]: failures.append("CONDITION_WRONG")
    if not fields[8]: failures.append("COMPOSITION_INCOMPLETE")
    if not fields[9]: failures.append("OUTPUT_CONSTRUCTION_WRONG")
    if not failures and not all(fields): failures.append("OTHER_SEMANTIC_FAILURE")
    return {"schema_valid": True, "primary_family_correct": fields[0], "core_operation_correct": bool(operations & set(expected["operations"])), "object_role_f1": roles[2], "relation_f1": relations[2], "selection_criteria_correct": fields[4], "iteration_detection_correct": predicted["iteration"]["enabled"] == expected["iteration"]["enabled"], "iteration_semantic_correct": fields[6], "conditional_logic_correct": fields[7], "composition_correct": fields[8], "output_construction_correct": fields[9], "semantic_parameter_correct": fields[10], "full_exact": all(fields), "semantic_success": all(critical), "slot_precision": sum(fields) / len(fields), "slot_recall": sum(fields) / len(fields), "slot_f1": sum(fields) / len(fields), "failure_taxonomy": failures}
