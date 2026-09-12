from __future__ import annotations

import inspect
from pathlib import Path

from arc.task import ARCExample, ARCGrid, ARCTask
from evaluation.semantic_ir_scorer import score
from recognition.arc_grid_feature_extractor import task_payload
from recognition.arc_semantic_ir import SCHEMA, validate
from recognition.arc_semantic_recognizer import parse_response, prompt_for


def _ir() -> dict:
    return {"primary_family": "ITERATION_REPEAT", "operations": ["COPY", "REPEAT"], "object_roles": ["MOTIF"], "relations": ["RELATIVE_POSITION"], "selection_criteria": ["NONE"], "spatial_transform": {"direction": "NONE", "transform": "NONE"}, "iteration": {"enabled": True, "mode": "REPEAT_UNTIL_BOUNDARY", "direction": "RIGHT", "step_rule": "FIXED_INTERVAL", "termination": "GRID_BOUNDARY"}, "conditional_logic": {"enabled": False, "condition": "NONE", "action": "NONE"}, "composition": [], "output_construction": "CONSTRUCTED_PATTERN", "semantic_parameters": {"color_policy": "PRESERVE", "repeat_object": "MOTIF"}}


def test_schema_is_finite_and_validator_rejects_unknown_semantics() -> None:
    assert SCHEMA["schema_id"] == "ARCSEMANTICIRV1"
    valid, _ = validate(_ir()); assert valid
    broken = _ir(); broken["primary_family"] = "TASK_123_SPECIAL"
    assert not validate(broken)[0]


def test_train_feature_payload_and_prompt_do_not_include_test_or_oracle() -> None:
    task = ARCTask("fixture", (ARCExample(ARCGrid([[0, 1]]), ARCGrid([[0, 2]])),), (ARCExample(ARCGrid([[9, 9]]), None),))
    payload = task_payload(task); prompt = prompt_for(task)
    assert "test" not in payload and "9, 9" not in prompt
    assert "semantic_ir_schema" in prompt and "macro" not in prompt.lower()


def test_parsing_and_deterministic_semantic_scoring() -> None:
    parsed, status = parse_response(__import__("json").dumps(_ir()))
    assert status == "SUCCESS" and parsed is not None
    exact = score(parsed, _ir()); assert exact["semantic_success"] and exact["full_exact"]
    wrong = _ir(); wrong["iteration"] = {**wrong["iteration"], "direction": "LEFT"}
    result = score(wrong, _ir()); assert not result["semantic_success"] and "ITERATION_PARAMETER_WRONG" in result["failure_taxonomy"]


def test_recognition_modules_have_no_downstream_execution_dependency() -> None:
    root = Path(__file__).parents[1]
    text = "\n".join((root / path).read_text(encoding="utf-8") for path in ("src/recognition/arc_semantic_ir.py", "src/recognition/arc_grid_feature_extractor.py", "src/recognition/arc_semantic_recognizer.py"))
    for forbidden in ("macro_compiler", "primitive_registry", "CapabilityProgramExecutor", "HardVerifier", "program_search"):
        assert forbidden not in text
    runner = (root / "scripts/run_grid_semantic_recognition.py").read_text(encoding="utf-8")
    assert "solutions" not in runner and "semantic_ir_scorer" not in runner
