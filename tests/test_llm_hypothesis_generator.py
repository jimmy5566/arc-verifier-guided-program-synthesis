import numpy as np
import pytest

from arc.task import ARCExample, ARCGrid, ARCTask
from llm.catalog import build_capability_catalog
from llm.comprehension import build_comprehension_benchmark, score_comprehension
from llm.context import build_task_context
from llm.direct_grid import DirectGridStatus, validate_direct_grid
from llm.models import CandidateStatus, HypothesisStep, LLMHypothesis
from llm.protocol import blocked_result, candidate_metrics
from llm.providers import NoBackendProvider, provider_audit
from llm.schema import execute_hypothesis, hypothesis_json_schema, parse_hypotheses, validate_hypothesis
from primitives.registry import REGISTRY
from scripts.finalize_llm_condition import score_predictions


RING = np.array([[0, 0, 0, 0, 0], [0, 2, 2, 2, 0], [0, 2, 0, 2, 0], [0, 2, 2, 2, 0], [0, 0, 0, 0, 0]], dtype=np.int16)


def _task():
    target = RING.copy(); target[2, 2] = 3
    return ARCTask("never_prompted", (ARCExample(ARCGrid(RING), ARCGrid(target)),), (ARCExample(ARCGrid(RING)),))


def test_catalogue_and_comprehension_are_registry_derived_and_complete():
    catalog = build_capability_catalog()
    assert catalog["frozen_registry_count"] == len(REGISTRY)
    assert {entry["primitive_id"] for entry in catalog["capabilities"]} == set(REGISTRY)
    assert all({"primitive_id", "input_types", "output_types", "parameters", "semantics", "preconditions", "failure_conditions", "does_not_do", "complexity_cost"} <= set(entry) for entry in catalog["capabilities"])
    benchmark = build_comprehension_benchmark()
    assert benchmark["question_count"] >= 40
    answers = {question["question_id"]: question["answer"] for question in benchmark["questions"]}
    assert score_comprehension(benchmark, answers)["accuracy"] == 1.0


def test_context_excludes_task_id_and_hidden_test_output():
    context = build_task_context(_task())
    text = str(context)
    assert "never_prompted" not in text
    assert "test_output" not in context
    assert context["test_input"] == RING.tolist()


def test_strict_schema_type_check_and_symbolic_hard_verification():
    parsed = parse_hypotheses({"hypotheses": [{"hypothesis_id": "fill", "steps": [{"primitive_id": "REG_FIND_ENCLOSED_REGIONS_V1", "params": {}}, {"primitive_id": "REG_FILL_INTERIOR_V1", "params": {"color": 3}}]}]})
    result = execute_hypothesis(parsed[0], _task())
    assert result.status == CandidateStatus.TRAIN_CONSISTENT
    assert np.array_equal(result.test_predictions[0], _task().train[0].output.values)
    invalid = LLMHypothesis("bad", (HypothesisStep("NOT_A_PRIMITIVE", {}),))
    assert validate_hypothesis(invalid).status == CandidateStatus.SCHEMA_INVALID
    wrong_chain = LLMHypothesis("bad_type", (HypothesisStep("GRAPH_FIND_ENDPOINTS_V1", {}),))
    assert validate_hypothesis(wrong_chain).status == CandidateStatus.TYPE_INVALID
    with pytest.raises(ValueError):
        parse_hypotheses({"hypotheses": [{"hypothesis_id": "x", "steps": [{"primitive_id": "REG_FILL_INTERIOR_V1", "params": {}, "extra": 1}]}]})


def test_structured_output_schema_enumerates_current_registry_ids():
    step = hypothesis_json_schema()["properties"]["hypotheses"]["items"]["properties"]["steps"]["items"]
    assert step["properties"]["primitive_id"]["enum"] == sorted(REGISTRY)


def test_direct_grid_and_no_backend_protocol_are_explicit():
    assert validate_direct_grid([[1, 2]], expected_shape=(1, 2)).status == DirectGridStatus.VALID
    assert validate_direct_grid([[10]]).status == DirectGridStatus.INVALID_COLORS
    assert validate_direct_grid([[1], [2]], expected_shape=(1, 2)).status == DirectGridStatus.WRONG_DIMENSIONS
    with pytest.raises(RuntimeError, match="BLOCKED_NO_LLM_BACKEND"):
        NoBackendProvider().generate_hypotheses({}, {}, object())
    result = blocked_result(provider_audit(), registry_count=len(REGISTRY), comprehension_questions=40)
    assert result["status"] == "BLOCKED_NO_LLM_BACKEND"
    metrics = candidate_metrics([])
    assert metrics["schema_valid_rate"] is None


def test_condition_scoring_requires_exact_all_test_outputs():
    target = RING.copy(); target[2, 2] = 3
    task = ARCTask("scored_fixture", (ARCExample(ARCGrid(RING), ARCGrid(target)),), (ARCExample(ARCGrid(RING), ARCGrid(target)),))
    correct = {task.task_id: {"prediction": [target.tolist()]}}
    exact, wrong = score_predictions(correct, {task.task_id: task})
    assert exact == [task.task_id] and wrong == []
    incorrect = {task.task_id: {"prediction": [[[0]]]}}
    exact, wrong = score_predictions(incorrect, {task.task_id: task})
    assert exact == [] and wrong == [task.task_id]
