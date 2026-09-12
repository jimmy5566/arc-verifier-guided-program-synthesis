from __future__ import annotations

from pathlib import Path

from arc.task import ARCExample, ARCGrid, ARCTask
from recognition.ablation_inputs import (
    FEATURES_ONLY,
    RAW_GRID_ONLY,
    RAW_RELATION_GRAPH,
    TRACK_A_CONDITIONS,
    no_test_oracle_terms,
    payload_for_condition,
)
from recognition.semantic_interfaces import (
    SLOT_ORDER,
    candidate_ontology,
    normalize_slots,
    parse_slot_response,
)
from scripts.run_parallel_semantic_ablation import _infer_one
from llm.transformers_provider import TransformersProvider


def _task() -> ARCTask:
    return ARCTask(
        "fixture",
        (
            ARCExample(ARCGrid([[0, 1, 0], [0, 1, 0]]), ARCGrid([[0, 2, 0], [0, 2, 0]])),
        ),
        (ARCExample(ARCGrid([[9, 9, 9]]), None),),
    )


def test_track_a_payloads_are_deterministic_and_gold_blind() -> None:
    task = _task()
    assert len(TRACK_A_CONDITIONS) == 4
    for condition in TRACK_A_CONDITIONS:
        left, right = payload_for_condition(task, condition), payload_for_condition(task, condition)
        assert left == right
        assert no_test_oracle_terms(left)
    assert "input" in payload_for_condition(task, RAW_GRID_ONLY)["train_pairs"][0]
    assert "objects" in payload_for_condition(task, FEATURES_ONLY)["train_pair_features"][0]["input"]
    assert payload_for_condition(task, FEATURES_ONLY)["train_pair_features"][0]["input"]["shape"] == [2, 3]
    graph = payload_for_condition(task, RAW_RELATION_GRAPH)["relation_graph"]["train_pair_relation_graphs"][0]
    assert {"nodes", "edges"} <= set(graph["input_graph"])
    assert "candidate_correspondences" in graph


def test_relation_graph_shape_signature_is_bounded_without_losing_identity() -> None:
    grid = ARCGrid([[0] * 30 for _ in range(29)] + [[2] * 30])
    task = ARCTask("fixture", (ARCExample(grid, grid),), (ARCExample(grid),))
    graph = payload_for_condition(task, RAW_RELATION_GRAPH)["relation_graph"]
    signature = graph["train_pair_relation_graphs"][0]["input_graph"]["nodes"][0]["shape_signature"]
    assert len(signature) < 40
    assert ";" not in signature


def test_dense_grid_feature_and_graph_budgets_are_deterministic() -> None:
    values = [[(row + col) % 2 for col in range(30)] for row in range(30)]
    grid = ARCGrid(values)
    task = ARCTask("fixture", (ARCExample(grid, grid),), (ARCExample(grid),))
    features = payload_for_condition(task, FEATURES_ONLY)["train_pair_features"][0]["input"]
    graph = payload_for_condition(task, RAW_RELATION_GRAPH)["relation_graph"]["train_pair_relation_graphs"][0]["input_graph"]
    assert len(features["objects"]) <= 12
    assert len(graph["nodes"]) <= 4
    assert features["suppressed_component_count_by_color"]
    assert graph["suppressed_component_count_by_color"]


def test_finite_slots_normalize_without_changing_semantic_ontology() -> None:
    ontology = candidate_ontology()
    assert tuple(ontology) == SLOT_ORDER
    slots = {name: values[0] for name, values in ontology.items()}
    slots.update({"FAMILY": "ITERATION_REPEAT", "OPERATION": "REPEAT", "SOURCE_ROLE": "MOTIF", "TARGET_ROLE": "NONE", "RELATION": "RELATIVE_POSITION", "ITERATION": "TRUE", "DIRECTION": "RIGHT", "STEP_RULE": "FIXED_INTERVAL", "TERMINATION": "GRID_BOUNDARY", "OUTPUT_MODE": "CONSTRUCTED_PATTERN", "REPEAT_OBJECT": "MOTIF"})
    value, status = normalize_slots(slots)
    assert status == "SUCCESS" and value is not None
    assert value["primary_family"] == "ITERATION_REPEAT"
    assert value["iteration"]["direction"] == "RIGHT"


def test_flat_interface_rejects_missing_or_invalid_slots() -> None:
    full = "\n".join(f"{key}={values[0]}" for key, values in candidate_ontology().items())
    assert parse_slot_response(full)[0] is not None
    assert parse_slot_response("FAMILY=ITERATION_REPEAT")[0] is None
    assert parse_slot_response(full.replace("FAMILY=COLOR_RECOLOR", "FAMILY=TASK_SPECIFIC_123"))[0] is None


def test_v2_recognition_runner_is_oracle_and_execution_blind() -> None:
    root = Path(__file__).parents[1]
    text = (root / "scripts/run_parallel_semantic_ablation.py").read_text(encoding="utf-8").lower()
    for forbidden in ("semantic_ir_scorer", "solutions", "macro_compiler", "capabilityprogramexecutor", "hardverifier", "program_search"):
        assert forbidden not in text


def test_tokenization_preflight_is_explicitly_generation_free() -> None:
    root = Path(__file__).parents[1]
    text = (root / "scripts/run_parallel_semantic_ablation.py").read_text(encoding="utf-8")
    start = text.index("def _tokenization_preflight")
    end = text.index("def _validate_conditions", start)
    assert ".generate(" not in text[start:end]
    assert "TOKENIZATION_PREFLIGHT_COMPLETE_NO_GENERATION" in text[start:end]


def test_runtime_fragmentation_fix_is_set_before_transformers_import() -> None:
    root = Path(__file__).parents[1]
    text = (root / "scripts/run_parallel_semantic_ablation.py").read_text(encoding="utf-8")
    worker = text[text.index("def _worker"):text.index("def _buckets")]
    assert worker.index("PYTORCH_ALLOC_CONF") < worker.index("from llm.models")


class _FakeGenerated:
    text = "\n".join(f"{key}={values[0]}" for key, values in candidate_ontology().items())
    prompt_tokens = 1
    completion_tokens = 1
    elapsed_seconds = 0.0


class _FakeProvider:
    def generate_text(self, _prompt: str, _config: object) -> _FakeGenerated:
        return _FakeGenerated()


def test_track_c_runner_resolves_all_prompt_builders() -> None:
    provider = _FakeProvider()
    for condition in ("C1_FLAT_TYPED_SLOTS", "C2_FINITE_CANDIDATE_CLASSIFICATION"):
        result = _infer_one(provider, _task(), condition, object())
        assert result["request_count"] == 1


def test_local_chat_template_fallback_is_explicit_and_not_silent() -> None:
    assert TransformersProvider._resolve_chat_template("template", "fallback") == ("template", "model_tokenizer")
    assert TransformersProvider._resolve_chat_template(None, "fallback") == ("fallback", "local_fallback_tokenizer")
    try:
        TransformersProvider._resolve_chat_template(None, None)
    except RuntimeError as error:
        assert "no chat template" in str(error)
    else:
        raise AssertionError("missing templates must reject")


def test_finalizer_is_the_explicit_local_oracle_boundary() -> None:
    root = Path(__file__).parents[1]
    text = (root / "scripts/finalize_semantic_ablation.py").read_text(encoding="utf-8")
    assert "semantic_ir_scorer" in text
    assert "--gold" in text
    assert "PREDICTIONS_FROZEN_BEFORE_GOLD_SCORING" in text
