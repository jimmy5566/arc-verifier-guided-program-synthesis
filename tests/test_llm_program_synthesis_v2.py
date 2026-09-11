from __future__ import annotations

import numpy as np
import json

from arc.task import ARCExample, ARCGrid, ARCTask
from llm.macro_compiler_v1 import MacroProgramCompilerV1, MacroProgramExecutorV1
from llm.macro_dsl import MACROS, MacroHypothesis, MacroStatus, MacroStep, macro_program_json_schema, parse_macro_hypotheses, validate_macro_hypothesis
from llm.macro_generator_v2 import MacroHypothesisGeneratorV2
from llm.models import GenerationConfig
from llm.parameter_solver_v1 import ParameterSolverV1, ResolutionStatus
from scripts.finalize_llm_program_synthesis_v2 import condition_metrics


def task(inp, out) -> ARCTask:
    return ARCTask("macro_fixture", (ARCExample(ARCGrid(inp), ARCGrid(out)),), (ARCExample(ARCGrid(inp)),))


def hypothesis(*steps: MacroStep) -> MacroHypothesis:
    return MacroHypothesis("h", tuple(steps))


def test_macro_registry_is_small_semantic_and_schema_hides_low_level_ids():
    assert 15 <= len(MACROS) <= 30
    schema = macro_program_json_schema()
    ids = schema["properties"]["hypotheses"]["items"]["properties"]["steps"]["items"]["properties"]["macro_id"]["enum"]
    assert set(ids) == set(MACROS)
    assert "REG_FILL_INTERIOR_V1" not in ids


def test_macro_schema_rejects_low_level_ids_and_raw_parameter_guessing():
    invalid = hypothesis(MacroStep("REG_FILL_INTERIOR_V1", {}))
    assert validate_macro_hypothesis(invalid).status == MacroStatus.MACRO_SCHEMA_INVALID
    raw = hypothesis(MacroStep("FILL_ENCLOSED_REGION", {"color_source": 3}))
    assert validate_macro_hypothesis(raw).status == MacroStatus.MACRO_SCHEMA_INVALID
    with np.testing.assert_raises(ValueError):
        parse_macro_hypotheses({"hypotheses": [{"hypothesis_id": "x", "steps": [{"macro_id": "FILL_ENCLOSED_REGION", "params": {}, "bad": 1}]}]})


def test_legacy_offline_ollama_transport_uses_json_mode_and_preserves_macro_schema(monkeypatch):
    fixture = task([[0]], [[0]])
    observed: dict[str, object] = {}

    class Response:
        def __enter__(self): return self
        def __exit__(self, *_): return None
        def read(self): return b'{"response":"{\\"hypotheses\\":[]}","prompt_eval_count":1,"eval_count":1}'

    def fake_urlopen(request, timeout):
        observed.update(json.loads(request.data.decode()))
        return Response()

    monkeypatch.setattr("llm.macro_generator_v2.urlopen", fake_urlopen)
    generated = MacroHypothesisGeneratorV2(GenerationConfig("qwen3:14b", hypothesis_budget=5)).generate(fixture)
    assert generated.hypotheses == ()
    assert observed["format"] == "json"
    prompt = json.loads(str(observed["prompt"]))
    assert prompt["macro_program_schema"] == macro_program_json_schema()


def test_symbolic_train_color_mapping_compiles_and_hard_verifies_fill_chain():
    inp = np.array([[0, 0, 0, 0, 0], [0, 2, 2, 2, 0], [0, 2, 0, 2, 0], [0, 2, 2, 2, 0], [0, 0, 0, 0, 0]], dtype=np.int16)
    out = inp.copy(); out[2, 2] = 3
    fixture = task(inp, out)
    macro = hypothesis(MacroStep("FILL_ENCLOSED_REGION", {"color_source": {"source": "FROM_TRAIN_COLOR_MAPPING"}}))
    compiled = MacroProgramCompilerV1().compile(macro, fixture)
    assert compiled.status == MacroStatus.COMPILED and compiled.program is not None
    verified = MacroProgramExecutorV1().verify_result(compiled.program, fixture)
    assert verified.status.value == "TRAIN_CONSISTENT"


def test_ambiguous_symbolic_color_is_not_silently_selected():
    inp = np.array([[0, 1], [2, 0]], dtype=np.int16)
    fixture = task(inp, inp)
    macro = hypothesis(MacroStep("FILL_ENCLOSED_REGION", {"color_source": {"source": "FROM_UNIQUE_COLOR"}}))
    solved = ParameterSolverV1().solve(macro, fixture)
    assert solved.status == ResolutionStatus.AMBIGUOUS
    assert MacroProgramCompilerV1().compile(macro, fixture).status == MacroStatus.PARAMETER_AMBIGUOUS


def test_direct_literal_ablation_is_explicit_and_separate_from_symbolic_mode():
    inp = np.array([[0, 0, 0], [0, 2, 0], [0, 0, 0]], dtype=np.int16)
    out = inp.copy()
    fixture = task(inp, out)
    macro = hypothesis(MacroStep("FILL_ENCLOSED_REGION", {"color_source": {"literal": 3}}))
    assert validate_macro_hypothesis(macro).status == MacroStatus.MACRO_SCHEMA_INVALID
    assert validate_macro_hypothesis(macro, allow_direct_literals=True).status == MacroStatus.COMPILED
    assert MacroProgramCompilerV1(allow_direct_literals=True).compile(macro, fixture).status == MacroStatus.COMPILED


def test_count_to_generate_compiles_to_existing_v4_program_and_executes():
    inp = np.array([[2, 0, 2], [0, 0, 0], [2, 0, 0]], dtype=np.int16)
    out = np.array([[2, 2, 2], [0, 0, 0], [0, 0, 0]], dtype=np.int16)
    fixture = task(inp, out)
    macro = hypothesis(
        MacroStep("COUNT_STRUCTURES", {"count_kind": {"literal": "OBJECT_COUNT"}}),
        MacroStep("GENERATE_FROM_COUNT", {"shape_source": {"source": "SAME_AS_INPUT"}, "color_source": {"source": "FROM_UNIQUE_COLOR"}, "orientation": {"source": "HORIZONTAL"}}),
    )
    compiled = MacroProgramCompilerV1().compile(macro, fixture)
    assert compiled.status == MacroStatus.COMPILED and compiled.program is not None
    assert MacroProgramExecutorV1().verify_result(compiled.program, fixture).status.value == "TRAIN_CONSISTENT"


def test_path_serialization_uses_compiler_owned_endpoint_binding():
    inp = np.array([[0, 1, 2, 3]], dtype=np.int16)
    fixture = task(inp, np.array([[1, 2, 3]], dtype=np.int16))
    macro = hypothesis(
        MacroStep("FIND_PATH", {}),
        MacroStep("TRACE_PATH", {"direction": {"source": "PATH_ENDPOINT_ORDER"}}),
        MacroStep("SERIALIZE_PATH", {"orientation": {"source": "HORIZONTAL"}}),
    )
    compiled = MacroProgramCompilerV1().compile(macro, fixture)
    assert compiled.status == MacroStatus.COMPILED and compiled.program is not None
    assert MacroProgramExecutorV1().verify_result(compiled.program, fixture).status.value == "TRAIN_CONSISTENT"


def test_pilot_metrics_separate_provider_failure_from_macro_schema_validity():
    checkpoint = {
        "task_count": 1,
        "runtime_seconds": 0.0,
        "total_prompt_tokens": 0,
        "total_output_tokens": 0,
        "records": {
            "macro_fixture": {
                "prediction": [[[1]]],
                "candidate_results": [
                    {"status": "PROVIDER_FAILED", "parameter_resolutions": [], "compiled_program": None},
                    {"status": "MACRO_SCHEMA_INVALID", "parameter_resolutions": [], "compiled_program": None},
                    {"status": "TRAIN_CONSISTENT", "parameter_resolutions": [], "compiled_program": {"steps": []}},
                ],
            }
        },
    }
    scored_task = ARCTask("macro_fixture", (ARCExample(ARCGrid([[0]]), ARCGrid([[1]])),), (ARCExample(ARCGrid([[0]]), ARCGrid([[1]])),))
    scored = {"macro_fixture": scored_task}
    metrics = condition_metrics(checkpoint, scored, set())
    assert metrics["provider_failure_rate"] == 1 / 3
    assert metrics["macro_response_candidate_count"] == 2
    assert metrics["macro_schema_valid_rate"] == 1 / 2


def test_pilot_metrics_compute_composition_only_against_frozen_baseline():
    checkpoint = {
        "task_count": 1,
        "runtime_seconds": 0.0,
        "total_prompt_tokens": 0,
        "total_output_tokens": 0,
        "records": {"macro_fixture": {"prediction": [[[1]]], "candidate_results": [{"status": "TRAIN_CONSISTENT", "parameter_resolutions": [], "compiled_program": {"steps": []}}]}},
    }
    scored_task = ARCTask("macro_fixture", (ARCExample(ARCGrid([[0]]), ARCGrid([[1]])),), (ARCExample(ARCGrid([[0]]), ARCGrid([[1]])),))
    scored = {"macro_fixture": scored_task}
    metrics = condition_metrics(checkpoint, scored, {"other_task"})
    assert metrics["composition_only_exact_solves"] == 1
    assert metrics["composition_only_task_ids"] == ["macro_fixture"]
