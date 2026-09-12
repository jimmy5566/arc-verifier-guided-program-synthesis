from __future__ import annotations

import ast
import hashlib
import inspect
import json
from pathlib import Path

from evaluation.skeleton_ir import SCHEMA_ID, validate
from evaluation.skeleton_scorer import derive_skeleton, score
from inference.grid_skeleton_recognizer import parse_response, prompt_for
from recognition.ablation_inputs import RAW_RELATION_GRAPH, payload_for_condition
from arc.task import ARCExample, ARCGrid, ARCTask


ROOT = Path(__file__).parents[1]


def _oracle() -> dict:
    return {
        "primary_family": "ITERATION_REPEAT", "operations": ["SELECT", "COPY", "REPEAT"],
        "object_roles": ["MOTIF", "SOURCE_OBJECT"], "relations": ["RELATIVE_POSITION"],
        "selection_criteria": ["NONE"], "spatial_transform": {"direction": "RIGHT", "transform": "NONE"},
        "iteration": {"enabled": True, "mode": "REPEAT_UNTIL_BOUNDARY", "direction": "RIGHT", "step_rule": "FIXED_INTERVAL", "termination": "GRID_BOUNDARY"},
        "conditional_logic": {"enabled": False, "condition": "NONE", "action": "NONE"},
        "composition": ["COPY", "REPEAT"], "output_construction": "CONSTRUCTED_PATTERN",
        "semantic_parameters": {"color_policy": "PRESERVE", "repeat_object": "MOTIF"},
    }


def test_frozen_cohort_hash_matches_semantic_v1() -> None:
    frozen = json.loads((ROOT / "configs/GRID_SKELETON_RECOGNITION_V1_FROZEN_CONFIG.json").read_text(encoding="utf-8"))
    prior = json.loads((ROOT / "configs/GRID_SEMANTIC_RECOGNITION_V1_FROZEN_CONFIG.json").read_text(encoding="utf-8"))
    assert frozen["cohort"]["task_ids_hash"] == prior["cohort"]["task_ids_hash"] == "a2f8fb66af4b83ec09bd2f3f2bf3fc5e948a6f7a6839898b5dbf26c6f77d1fc8"
    assert frozen["cohort"]["task_count"] == 30 and not frozen["cohort"]["re_sampling"]


def test_oracle_skeleton_derivation_is_deterministic_and_strips_parameters() -> None:
    first, second = derive_skeleton(_oracle()), derive_skeleton(_oracle())
    assert first == second and first["schema_id"] == SCHEMA_ID
    assert first["operations"][-1]["op"] == "REPEAT"
    assert first["operations"][-1]["args"] == {"input": "$OBJECT", "direction": "$DIRECTION", "step": "$STEP", "termination": "$TERMINATION"}
    assert all(value.startswith("$") for operation in first["operations"] for value in operation["args"].values())
    assert "RIGHT" not in json.dumps(first) and "GRID_BOUNDARY" not in json.dumps(first)


def test_private_frozen_oracle_derives_all_30_without_ontology_gaps() -> None:
    private_gold = ROOT / "artifacts/oracle_arc_semantic_ir_v1_private.json"
    if not private_gold.exists():
        return
    payload = json.loads(private_gold.read_text(encoding="utf-8"))
    derived = {task_id: derive_skeleton(oracle) for task_id, oracle in payload["gold"].items()}
    assert len(derived) == 30
    assert all(validate(value)[0] for value in derived.values())
    assert all(value.startswith("$") for skeleton in derived.values() for operation in skeleton["operations"] for value in operation["args"].values())


def test_typed_holes_reject_concrete_parameters_and_unknown_operations() -> None:
    skeleton = derive_skeleton(_oracle())
    assert validate(skeleton)[0]
    broken = json.loads(json.dumps(skeleton)); broken["operations"][-1]["args"]["direction"] = "RIGHT"
    assert not validate(broken)[0]
    broken = json.loads(json.dumps(skeleton)); broken["operations"][0]["op"] = "TASK_123_SPECIAL"
    assert not validate(broken)[0]


def test_sequence_and_hole_scorers_are_independent() -> None:
    gold = derive_skeleton(_oracle())
    assert score(gold, gold)["skeleton_success"]
    reordered = {**gold, "operations": list(reversed(gold["operations"]))}
    result = score(reordered, gold)
    assert result["operation_set_correct"] and not result["operation_sequence_exact"] and not result["skeleton_success"]
    holes = {**gold, "required_holes": ["$OBJECT"]}
    assert not score(holes, gold)["required_hole_set_correct"]


def test_a3_prompt_is_train_only_and_no_downstream_dependency() -> None:
    task = ARCTask("fixture", (ARCExample(ARCGrid([[0, 1]]), ARCGrid([[0, 2]])),), (ARCExample(ARCGrid([[9, 9]]), None),))
    payload = payload_for_condition(task, RAW_RELATION_GRAPH)
    prompt = prompt_for(task)
    assert "test" not in json.dumps(payload).lower() and "9, 9" not in prompt
    assert "$DIRECTION" in prompt and "RIGHT" not in prompt
    runner = (ROOT / "scripts/run_grid_skeleton_recognition.py").read_text(encoding="utf-8")
    imported = {node.module for node in ast.walk(ast.parse(runner)) if isinstance(node, ast.ImportFrom) and node.module}
    assert "evaluation.skeleton_scorer" not in imported and "evaluation.semantic_ir_scorer" not in imported
    source = inspect.getsource(__import__("inference.grid_skeleton_recognizer", fromlist=["*"]))
    for forbidden in ("CapabilityProgramExecutor", "HardVerifier", "macro_compiler", "parameter_solver"):
        assert forbidden not in source


def test_parser_and_frozen_old_artifact_integrity() -> None:
    gold = derive_skeleton(_oracle())
    parsed, status = parse_response(json.dumps(gold))
    assert status == "SUCCESS" and parsed == gold
    old = ROOT / "artifacts/grid_semantic_kernel_output_v26/GRID_SEMANTIC_RECOGNITION_V1_PREDICTIONS_FROZEN.json"
    if old.exists():
        before = hashlib.sha256(old.read_bytes()).hexdigest()
        assert before == "b8df3ae0e79a20b6e39ce36ab8117eda8ef45fbd9ea2bdee88454ccbbe5dfb5a"
        assert hashlib.sha256(old.read_bytes()).hexdigest() == before


def test_finalizer_freeze_gate_precedes_oracle_and_recovery_is_task_level() -> None:
    finalizer = (ROOT / "scripts/finalize_grid_skeleton_recognition.py").read_text(encoding="utf-8")
    assert finalizer.index("raw_predictions = args.predictions.read_bytes()") < finalizer.index("oracle_payload = json.loads(args.oracle")
    old_success = {"a": False, "b": True, "c": True, "d": False}
    skeleton_success = {"a": True, "b": True, "c": False, "d": False}
    recovery = {
        "old_full_semantic_fail_to_skeleton_success": sum(not old_success[key] and skeleton_success[key] for key in old_success),
        "old_semantic_success_to_skeleton_success": sum(old_success[key] and skeleton_success[key] for key in old_success),
        "old_semantic_success_to_skeleton_fail": sum(old_success[key] and not skeleton_success[key] for key in old_success),
        "both_fail": sum(not old_success[key] and not skeleton_success[key] for key in old_success),
    }
    assert recovery == {"old_full_semantic_fail_to_skeleton_success": 1, "old_semantic_success_to_skeleton_success": 1, "old_semantic_success_to_skeleton_fail": 1, "both_fail": 1}
