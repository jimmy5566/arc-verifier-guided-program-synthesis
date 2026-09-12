"""Deterministic skeleton derivation and target-blind scoring helpers."""
from __future__ import annotations

from typing import Any, Mapping

from .skeleton_ir import canonical, validate


def _operation(op: str, **args: str) -> dict[str, Any]:
    return {"op": op, "args": args}


def derive_skeleton(oracle: Mapping[str, Any]) -> dict[str, Any]:
    """Deterministically strip all semantic parameter values from frozen IR."""
    operations: list[dict[str, Any]] = []
    source_roles = set(oracle["object_roles"])
    relation_present = set(oracle["relations"]) != {"NONE"}
    for operation in oracle["operations"]:
        if operation == "SELECT":
            operations.append(_operation("SELECT_OBJECT", output="$OBJECT"))
        elif operation == "COPY":
            if relation_present:
                operations.append(_operation("COPY_RELATIVE_TO", source="$SOURCE", reference="$REFERENCE", direction="$DIRECTION", distance="$DISTANCE"))
            else:
                operations.append(_operation("COPY", input="$OBJECT", output="$OBJECT"))
        elif operation == "REPEAT":
            operations.append(_operation("REPEAT", input="$OBJECT", direction="$DIRECTION", step="$STEP", termination="$TERMINATION"))
        elif operation == "RECOLOR":
            operations.append(_operation("RECOLOR", input="$OBJECT", color="$COLOR"))
        elif operation == "EXTRACT":
            operations.append(_operation("CROP", input="$OBJECT"))
        elif operation == "TRANSFORM":
            transform = oracle["spatial_transform"]["transform"]
            if transform == "ROTATE":
                operations.append(_operation("ROTATE", input="$OBJECT"))
            elif transform == "REFLECT":
                operations.append(_operation("REFLECT", input="$OBJECT"))
            elif transform == "TRANSLATE":
                operations.append(_operation("MOVE", input="$OBJECT", direction="$DIRECTION", distance="$DISTANCE"))
            elif relation_present or {"SOURCE_OBJECT", "REFERENCE_OBJECT"} <= source_roles:
                operations.append(_operation("TRANSFORM_RELATIVE_TO", source="$SOURCE", reference="$REFERENCE"))
            else:
                operations.append(_operation("MOVE", input="$OBJECT", direction="$DIRECTION", distance="$DISTANCE"))
        elif operation == "EXPAND":
            operations.append(_operation("EXTEND", input="$OBJECT", direction="$DIRECTION"))
        elif operation == "FILL":
            operations.append(_operation("FILL", input="$OBJECT", color="$COLOR"))
        elif operation == "CONNECT":
            operations.append(_operation("CONNECT", source="$SOURCE", target="$TARGET"))
        elif operation == "REARRANGE":
            operations.append(_operation("ALIGN", input="$OBJECT", reference="$REFERENCE"))
        elif operation == "CONSTRUCT":
            operations.append(_operation("COMPOSE", input="$OBJECT", output="$OBJECT"))
        else:
            raise ValueError(f"SKELETON_ONTOLOGY_GAP:{operation}")
    if oracle["conditional_logic"]["enabled"]:
        operations.append(_operation("CONDITIONAL_APPLY", input="$OBJECT", condition="$CONDITION"))
    holes = sorted({value for item in operations for value in item["args"].values()})
    skeleton = {"schema_id": "ARCSKELETONIRV1", "family": oracle["primary_family"], "operations": operations, "required_holes": holes}
    valid, reason = validate(skeleton)
    if not valid:
        raise ValueError(f"derived skeleton invalid: {reason}")
    return canonical(skeleton)


def score(prediction: Mapping[str, Any] | None, gold: Mapping[str, Any]) -> dict[str, Any]:
    if prediction is None or not validate(prediction)[0]:
        return {"schema_valid": False, "family_correct": False, "operation_set_correct": False, "operation_sequence_exact": False, "composition_structure_correct": False, "required_hole_set_correct": False, "full_skeleton_exact": False, "skeleton_success": False}
    prediction, gold = canonical(prediction), canonical(gold)
    pred_ops = [item["op"] for item in prediction["operations"]]
    gold_ops = [item["op"] for item in gold["operations"]]
    sequence = pred_ops == gold_ops
    family = prediction["family"] == gold["family"]
    op_set = set(pred_ops) == set(gold_ops)
    holes = set(prediction["required_holes"]) == set(gold["required_holes"])
    # Operation order and each finite op's typed argument signature are the
    # composition structure; parameter values cannot enter this representation.
    structure = sequence and [item["args"] for item in prediction["operations"]] == [item["args"] for item in gold["operations"]]
    exact = family and structure and holes
    return {"schema_valid": True, "family_correct": family, "operation_set_correct": op_set, "operation_sequence_exact": sequence, "composition_structure_correct": structure, "required_hole_set_correct": holes, "full_skeleton_exact": exact, "skeleton_success": family and sequence and holes}
