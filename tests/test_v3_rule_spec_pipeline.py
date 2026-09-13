from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from arc.task import ARCExample, ARCGrid, ARCTask
from v3.evidence.cross_pair import intersect_candidates
from v3.evidence.extractor import extract_evidence, extract_task_evidence
from v3.parameters.joint_solver import infer_parameters
from v3.pipeline import train_consistent_rule_specs
from v3.recognition.recognizer_interface import parse_hypotheses, recognition_prompt
from v3.schema.rule_skeleton import OperationId, ParameterSlot, RuleSkeleton
from v3.schema.rule_spec import RuleSpec
from v3.verification.verifier import HardVerifier
from v3.execution.rule_executor import RuleExecutor
from v3.diagnostics.backend_forensics import primary_failure_attribution, recolor_candidate_semantic_recall, semantic_sufficiency


ROOT = Path(__file__).parents[1]


def _skeleton(family: str, *operations: OperationId) -> RuleSkeleton:
    return RuleSkeleton.from_operations(family, operations)


def _run(skeleton: RuleSkeleton, pairs: list[tuple[np.ndarray, np.ndarray]]):
    evidence = extract_evidence(pairs)
    candidates = train_consistent_rule_specs((skeleton,), evidence, pairs)
    assert candidates
    assert HardVerifier().verify(candidates[0], pairs).passed
    return candidates[0]


def test_evidence_bundle_reuses_a3_graph_and_excludes_test_data() -> None:
    task = ARCTask("fixture", (ARCExample(ARCGrid([[0, 1], [0, 0]]), ARCGrid([[0, 2], [0, 0]])),), (ARCExample(ARCGrid([[9, 9]]), None),))
    bundle = extract_task_evidence(task)
    assert bundle.invariants["pair_count"] == 1
    assert bundle.pairs[0].a3_relation_graph is not None
    assert 9 not in bundle.pairs[0].input_grid and 9 not in bundle.pairs[0].output_grid


def test_upstream_prompt_normalizes_train_derived_numpy_facts_and_parser_rejects_bad_slots() -> None:
    task = ARCTask(
        "fixture",
        (ARCExample(ARCGrid([[0, 1], [0, 0]]), ARCGrid([[0, 2], [0, 0]])),),
        (ARCExample(ARCGrid([[9, 9]]), None),),
    )
    evidence = extract_task_evidence(task)
    from v3.evidence.cross_pair import derive_cross_pair_evidence
    prompt = recognition_prompt(task, evidence, derive_cross_pair_evidence(evidence), top_k=1)
    assert json.loads(prompt)["train_grids"]["train_pairs"][0]["output_rows"] == ["02", "00"]
    assert "9" not in prompt
    raw = json.dumps({"hypotheses": [{"family": "X", "operations": ["RECOLOR"], "required_slots": "$TARGET_COLOR"}]})
    assert parse_hypotheses(raw, limit=1)[0] == ()


def test_every_exposed_operation_has_real_executor_semantics_and_no_noop() -> None:
    audit = RuleExecutor.operation_audit()
    assert len(audit["exposed_operations"]) == 16
    assert audit["exposed_operations"] == audit["fully_executable_operations"]
    assert audit["unsupported_exposed_operations"] == () and audit["no_op_operations"] == ()


def test_cross_pair_joint_intersection_and_train_consistency() -> None:
    cross = intersect_candidates([
        {"DIRECTION": frozenset({(0, 1), (1, 0)}), "STEP": frozenset({2})},
        {"DIRECTION": frozenset({(0, 1)}), "STEP": frozenset({2, 3})},
        {"DIRECTION": frozenset({(0, 1), (0, -1)}), "STEP": frozenset({2})},
    ])
    assert cross.candidates == {"DIRECTION": frozenset({(0, 1)}), "STEP": frozenset({2})}
    assert cross.candidate_frequency["DIRECTION"][(0, 1)] == 3
    assert cross.invariants["all_pair_candidate_intersection_nonempty"]
    pairs = []
    for row in (0, 1, 2):
        source = np.zeros((3, 7), dtype=int); source[row, 1] = 1
        target = source.copy(); target[row, 3] = target[row, 5] = 1
        pairs.append((source, target))
    spec = _run(_skeleton("ITERATION", OperationId.SELECT, OperationId.REPEAT), pairs)
    assert spec.value(ParameterSlot.DIRECTION) == (0, 1) and spec.value(ParameterSlot.STEP) == 2
    assert spec.value(ParameterSlot.TERMINATION) in {"BOUNDARY", "FIXED_COUNT"}


def test_pairwise_termination_and_selector_candidates_are_evidence_derived() -> None:
    source = np.zeros((3, 7), dtype=int); source[1, 1] = 4
    target = source.copy(); target[1, 3] = target[1, 5] = 4
    evidence = extract_evidence([(source, target)]).pairs[0]
    assert evidence.parameter_candidates["TERMINATION"] <= {"BOUNDARY", "FIXED_COUNT"}
    assert evidence.parameter_candidates["COUNT"] and evidence.parameter_candidates["STEP"]
    skeleton = _skeleton("REPEAT", OperationId.SELECT, OperationId.REPEAT)
    inferred = infer_parameters(skeleton, extract_evidence([(source, target)]))
    assert "COLOR:4" in inferred.candidates[ParameterSlot.SELECTOR]
    assert "ALWAYS" not in repr(inferred.candidates)


def test_vertical_slice_recolor() -> None:
    pairs = []
    for row in (1, 2):
        source = np.zeros((4, 4), dtype=int); source[row, 1] = 1
        target = source.copy(); target[row, 1] = 2
        pairs.append((source, target))
    spec = _run(_skeleton("RECOLOR", OperationId.SELECT, OperationId.RECOLOR), pairs)
    assert spec.value(ParameterSlot.TARGET_COLOR) == 2


def test_background_recolor_is_a_generic_color_selector_and_target_candidate() -> None:
    pairs = []
    for source in (
        np.array([[7, 1, 7], [1, 7, 7]], dtype=int),
        np.array([[2, 7, 7], [7, 2, 7]], dtype=int),
    ):
        target = source.copy(); target[target == 7] = 5
        pairs.append((source, target))
    evidence = extract_evidence(pairs)
    inferred = infer_parameters(_skeleton("RECOLOR", OperationId.SELECT, OperationId.RECOLOR), evidence)
    assert "COLOR:7" in inferred.candidates[ParameterSlot.SELECTOR]
    assert 5 in inferred.candidates[ParameterSlot.TARGET_COLOR]
    _run(_skeleton("RECOLOR", OperationId.SELECT, OperationId.RECOLOR), pairs)


def test_backend_forensics_attributes_missing_skeleton_semantics_without_task_ids() -> None:
    repeat_oracle = {
        "iteration": {"mode": "PROGRESSIVE_EXPANSION", "step_rule": "UNIT_STEP"},
        "semantic_parameters": {"color_policy": "SEQUENCE_FROM_MARKERS"},
        "selection_criteria": [], "conditional_logic": {"enabled": False},
    }
    repeat = _skeleton("ITERATION", OperationId.SELECT, OperationId.REPEAT)
    assert primary_failure_attribution(repeat_oracle, repeat, "VERIFICATION_FAILURE") == "SKELETON_LOST_RULE_SEMANTICS"
    assert semantic_sufficiency(repeat_oracle, repeat) == (False, "SKELETON_LOST_RULE_SEMANTICS")
    assert primary_failure_attribution({**repeat_oracle, "conditional_logic": {"enabled": True}}, None, "EXECUTION_CAPABILITY_FAILURE") == "MISSING_CANONICAL_OPERATION_CONDITIONAL_ROLE_LOGIC"


def test_train_derived_candidate_semantic_recall_for_uniform_recolor() -> None:
    source = np.array([[7, 1], [7, 7]], dtype=int)
    target = np.array([[5, 1], [5, 5]], dtype=int)
    skeleton = _skeleton("RECOLOR", OperationId.SELECT, OperationId.RECOLOR)
    audit = recolor_candidate_semantic_recall(skeleton, extract_evidence([(source, target)]))
    assert audit["assessable"] and audit["semantic_recall"] == audit["selector_recall"] == audit["target_color_recall"] == 1.0


def test_vertical_slice_copy_move_and_instance_binding() -> None:
    pairs = []
    for row in (1, 2):
        source = np.zeros((5, 6), dtype=int); source[row, 1] = 1
        target = source.copy(); target[row, 3] = 1; target[row, 1] = 0
        pairs.append((source, target))
    _run(_skeleton("MOVE", OperationId.SELECT, OperationId.MOVE), pairs)
    skeleton = _skeleton("RECOLOR", OperationId.SELECT, OperationId.RECOLOR)
    fixed = RuleSpec(skeleton, {ParameterSlot.SELECTOR: "SMALLEST_OBJECT", ParameterSlot.TARGET_COLOR: 7})
    first = np.array([[0, 1, 0, 2, 2], [0, 0, 0, 2, 2]], dtype=int)
    second = np.array([[2, 2, 0, 0, 1], [2, 2, 0, 0, 0]], dtype=int)
    expected_first, expected_second = first.copy(), second.copy(); expected_first[0, 1] = expected_second[0, 4] = 7
    assert HardVerifier().verify(fixed, ((first, expected_first), (second, expected_second))).passed


def test_vertical_slice_relational_copy_and_composition() -> None:
    pairs = []
    for row in (1, 2):
        source = np.zeros((5, 6), dtype=int); source[row, 1] = 1; source[row, 3] = 2
        target = source.copy(); target[row, 4] = 1
        pairs.append((source, target))
    evidence = extract_evidence(pairs)
    forced = tuple(replace(pair, parameter_candidates={**pair.parameter_candidates, "DIRECTION": frozenset({(0, 1)}), "DISTANCE": frozenset({1}), "REFERENCE_COLOR": frozenset({2})}) for pair in evidence.pairs)
    specs = train_consistent_rule_specs((_skeleton("RELATIONAL", OperationId.SELECT, OperationId.RELATIONAL_COPY),), replace(evidence, pairs=forced), pairs)
    assert specs and HardVerifier().verify(specs[0], pairs).passed
    source = np.zeros((4, 6), dtype=int); source[1, 1] = 1
    target = source.copy(); target[1, 1] = 0; target[1, 3] = 3
    _run(_skeleton("COMPOSITION", OperationId.SELECT, OperationId.MOVE, OperationId.RECOLOR), [(source, target)])


def test_parameter_and_execution_layers_have_no_llm_or_macro_compiler_dependency() -> None:
    for relative in ("src/v3/parameters/candidate_generator.py", "src/v3/parameters/joint_solver.py", "src/v3/execution/rule_executor.py"):
        tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
        imported = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
        assert all("llm" not in module.lower() and "macro_compiler" not in module.lower() for module in imported)
    verifier = (ROOT / "src/v3/verification/verifier.py").read_text(encoding="utf-8")
    assert "repair" in verifier and "def repair" not in verifier
    imports = {node.module for node in ast.walk(ast.parse((ROOT / "src/v3/execution/rule_executor.py").read_text(encoding="utf-8"))) if isinstance(node, ast.ImportFrom) and node.module}
    assert "compiler" not in imports


def test_v3_dependency_direction_and_legacy_artifacts_are_isolated() -> None:
    forbidden = {
        "evidence": {"recognition", "parameters", "execution", "verification"},
        "recognition": {"parameters", "execution", "verification"},
        "parameters": {"recognition", "execution", "verification"},
        "execution": {"recognition", "parameters", "verification"},
        "verification": {"recognition", "parameters"},
    }
    for path in (ROOT / "src/v3").rglob("*.py"):
        layer = next((name for name in forbidden if f"{name}" in path.parts), None)
        if layer is None:
            continue
        imports = {node.module for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))) if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("v3.")}
        for module in imports:
            dependency = module.split(".")[1]
            assert dependency not in forbidden[layer]
    legacy = ROOT / "experiments/results/GRID_SEMANTIC_RECOGNITION_V1.json"
    assert legacy.exists()
    assert hashlib.sha256(legacy.read_bytes()).hexdigest() == "f176fd363d475007639d7094810c6ac9b533ed2a58c93575d7245b4d73188aba"


def test_phase_a_audit_is_train_only_and_phase_b_c_are_stopped_on_gate_failure() -> None:
    runner = (ROOT / "scripts/run_v3_backend_audit.py").read_text(encoding="utf-8")
    tree = ast.parse(runner)
    imports = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
    assert "llm" not in " ".join(imports).lower() and "solutions" not in " ".join(imports).lower()
    result = json.loads((ROOT / "experiments/results/ARC2_V3_BACKEND_AUDIT_V1.json").read_text(encoding="utf-8"))
    assert result["status"] == "BACKEND_NOT_READY"
    assert result["phase_b"] == result["phase_c"] == "NOT_RUN_PHASE_A_BACKEND_NOT_READY"
    assert result["oracle_skeleton_test_exact"] == "NOT_RUN_PHASE_A_GATE_FAILED"
