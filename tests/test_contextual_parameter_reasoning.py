from __future__ import annotations

import inspect
import json
from pathlib import Path

from inference.contextual_parameter_reasoning_runner import R1, _r1_records, _score
from llm.compiler_aware_interface import skeleton_by_id
from llm.contextual_parameter_classifier import ContextualParameterClassifierV1, contextual_prompt
from llm.contextual_parameter_ir import ContextualParameterIRV1, contextual_ir_schema
from llm.contextual_selective_parameter_repair import ContextualSelectiveParameterRepairV1
from llm.parameter_grounding import SemanticChoice, choices_from_program, parameter_slots
from llm.parameter_semantic_ontology import ParameterSemanticOntologyV1
from llm.semantic_relation_normalizer import SemanticRelationNormalizerV1, extract_relation_features

ROOT = Path(__file__).resolve().parents[1]


class FixedScorer:
    def score_continuations(self, prompt, continuations):
        return tuple(float(len(item)) for item in continuations)


def frozen(): return json.loads((ROOT / "configs/CONTEXTUAL_PARAMETER_REASONING_V1_FROZEN_CONFIG.json").read_text())


def test_contextual_ir_is_typed_and_excludes_case_labels() -> None:
    f = frozen(); item = f["q1_baseline_programs"]["three_step_path_01"]; skeleton = skeleton_by_id(item["skeleton_id"]); assert skeleton
    slot = next(slot for slot in parameter_slots(skeleton) if slot.parameter == "direction"); ontology = ParameterSemanticOntologyV1()
    ir = ContextualParameterIRV1.build(instruction="trace endpoint order", family=item["family"], skeleton=skeleton, slot=slot, candidates=ontology.candidates_for_slot(slot), semantic_features=extract_relation_features("trace endpoint order"))
    assert ir.public()["target_slot"]["parameter"] == "direction" and "case_id" not in json.dumps(ir.public())
    assert "canonical_program" in contextual_ir_schema()["forbidden"]


def test_relation_extractor_and_normalizer_are_generic_contract_rules() -> None:
    assert set(extract_relation_features("serialize left-to-right from that count")) >= {"RELATION.HORIZONTAL", "RELATION.FROM_PREVIOUS_COUNT"}
    f = frozen(); item = f["q1_baseline_programs"]["three_step_path_01"]; skeleton = skeleton_by_id(item["skeleton_id"]); assert skeleton
    ontology = ParameterSemanticOntologyV1(); current = choices_from_program(skeleton, item["program"]); normalizer = SemanticRelationNormalizerV1(ontology)
    direction = next(slot for slot in parameter_slots(skeleton) if slot.parameter == "direction")
    context = ContextualParameterIRV1.build(instruction="Trace in endpoint order", family=item["family"], skeleton=skeleton, slot=direction, candidates=ontology.candidates_for_slot(direction), semantic_features=extract_relation_features("Trace in endpoint order"))
    decision = normalizer.decide(context, current_slots=current)
    assert decision.state == "UNAMBIGUOUS" and decision.candidate and decision.candidate.choice == SemanticChoice("SOURCE", "PATH_ENDPOINT_ORDER")
    assert "three_step_path_01" not in inspect.getsource(SemanticRelationNormalizerV1)


def test_contextual_classifier_is_deterministic_and_no_generation() -> None:
    f = frozen(); item = f["q1_baseline_programs"]["three_step_path_01"]; skeleton = skeleton_by_id(item["skeleton_id"]); assert skeleton
    slot = next(slot for slot in parameter_slots(skeleton) if slot.parameter == "orientation"); ontology = ParameterSemanticOntologyV1()
    context = ContextualParameterIRV1.build(instruction="serialize vertical", family=item["family"], skeleton=skeleton, slot=slot, candidates=ontology.candidates_for_slot(slot), semantic_features=extract_relation_features("serialize vertical"))
    rank = ContextualParameterClassifierV1(FixedScorer()).rank(context)
    assert rank == ContextualParameterClassifierV1(FixedScorer()).rank(context)
    assert "generate" not in inspect.getsource(ContextualParameterClassifierV1) and "allowed_candidates" in contextual_prompt(context)


def test_r3_default_abstains_and_preserves_other_slots() -> None:
    f = frozen(); item = f["q1_baseline_programs"]["three_step_path_01"]; skeleton = skeleton_by_id(item["skeleton_id"]); assert skeleton
    ontology = ParameterSemanticOntologyV1(); contexts = {}; current = choices_from_program(skeleton, item["program"])
    for slot in parameter_slots(skeleton): contexts[slot.key] = ContextualParameterIRV1.build(instruction="unspecified", family=item["family"], skeleton=skeleton, slot=slot, candidates=ontology.candidates_for_slot(slot), semantic_features=())
    from llm.parameter_semantic_ir import ParameterSemanticIRV1
    r1 = {key: ParameterSemanticIRV1(key, None, "test", "AMBIGUOUS") for key in contexts}; r2 = {key: ParameterSemanticIRV1(key, None, "test", "LOW") for key in contexts}
    repaired = ContextualSelectiveParameterRepairV1(0.05).repair(skeleton, item["program"], contexts, r1, r2)
    assert repaired.changed_slots == () and repaired.program == item["program"] and current == choices_from_program(skeleton, repaired.program)


def test_r1_runs_the_same_pipeline_on_all_60_without_model_or_labels(tmp_path) -> None:
    f = frozen(); benchmark = json.loads((ROOT / "configs/macro_api_comprehension_benchmark_v1.json").read_text()); cases = {item["case_id"]: item for item in benchmark["cases"]}
    records = _r1_records(cases=cases, frozen=f, root=tmp_path)
    assert len(records) == 60 and {item["condition"] for item in records} == {R1}
    scored = _score(records, cases)
    assert sum(item["semantic"]["semantic_success"] for item in scored) == 48
