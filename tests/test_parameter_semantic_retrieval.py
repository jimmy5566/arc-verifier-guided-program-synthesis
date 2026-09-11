from __future__ import annotations

import inspect
import json
from pathlib import Path

from llm.compiler_aware_interface import skeleton_by_id
from llm.parameter_candidate_inventory import build_candidate_inventory
from llm.parameter_candidate_scorer import PairwiseContrastiveScorerV1, ParameterLikelihoodScorerV1
from llm.parameter_grounding import SemanticChoice, choices_from_program, parameter_slots
from llm.parameter_semantic_ir import ParameterSemanticIRV1
from llm.parameter_semantic_ontology import ParameterSemanticOntologyV1, deterministic_match
from llm.selective_parameter_repair_v2 import SelectiveParameterRepairV2
from llm.transformers_provider import TransformersProvider


ROOT = Path(__file__).resolve().parents[1]


def _frozen() -> dict:
    return json.loads((ROOT / "configs" / "PARAMETER_GROUNDING_REPAIR_V1_FROZEN_CONFIG.json").read_text(encoding="utf-8"))


class _FixedScorer:
    def score_continuations(self, prompt: str, continuations: tuple[str, ...]) -> tuple[float, ...]:
        # Candidate-only deterministic fixture: no benchmark instruction or
        # canonical program participates in the scoring rule.
        return tuple(float(len(value)) for value in continuations)


def test_ontology_is_registry_derived_and_candidate_ids_are_stable() -> None:
    ontology = ParameterSemanticOntologyV1()
    payload = ontology.public()
    assert payload["ontology_id"] == "PARAMETER_SEMANTIC_ONTOLOGY_V1"
    assert ontology.sha256 == ParameterSemanticOntologyV1().sha256
    assert all("parameter_path_" not in json.dumps(item) for item in payload["parameters"].values())
    assert all("canonical" not in json.dumps(item).lower() for item in payload["parameters"].values())
    assert any(item["candidate_id"] == "ORIENTATION.SOURCE.HORIZONTAL" for item in payload["parameters"]["orientation"])


def test_deterministic_alias_match_selects_representation_without_case_specific_rule() -> None:
    frozen = _frozen(); ontology = ParameterSemanticOntologyV1()
    skeleton = skeleton_by_id("CV027")
    assert skeleton is not None
    direction = next(slot for slot in parameter_slots(skeleton) if slot.parameter == "direction")
    orientation = next(slot for slot in parameter_slots(skeleton) if slot.parameter == "orientation")
    text = "Find a path, trace it in endpoint order, and serialize it as a vertical grid. Use source wrappers for direction and orientation."
    assert deterministic_match(text, direction, ontology).candidate == next(item for item in ontology.candidates_for_slot(direction) if item.candidate_id == "DIRECTION.SOURCE.PATH_ENDPOINT_ORDER")
    assert deterministic_match(text, orientation, ontology).candidate == next(item for item in ontology.candidates_for_slot(orientation) if item.candidate_id == "ORIENTATION.SOURCE.VERTICAL")
    assert "parameter_path_01" not in inspect.getsource(deterministic_match)
    assert frozen["p0_s2_baseline"]["wrong_parameter"] == 17


def test_inventory_uses_s2_contract_but_no_canonical_answer() -> None:
    frozen = _frozen(); ontology = ParameterSemanticOntologyV1()
    inventory = build_candidate_inventory(s2_baseline_programs=frozen["s2_baseline_programs"], s2_outcomes=frozen["p0_s2_baseline"]["case_outcomes"], ontology=ontology)
    encoded = json.dumps(inventory, sort_keys=True)
    assert inventory["case_count"] == 17
    assert inventory["protocol"]["canonical_answer_used"] is False
    assert "canonical_program" not in encoded and "expected_macro_ids" not in encoded
    assert all(all(item["dsl_valid"] and item["compiler_contract_valid"] for item in slot["candidate_validation"]) for case in inventory["cases"] for slot in case["slots"])


def test_likelihood_and_pairwise_scorers_are_deterministic_and_generation_free() -> None:
    ontology = ParameterSemanticOntologyV1(); skeleton = skeleton_by_id("CV027")
    assert skeleton is not None
    slot = next(item for item in parameter_slots(skeleton) if item.parameter == "orientation")
    likelihood = ParameterLikelihoodScorerV1(_FixedScorer()).rank("serialize vertical", skeleton, slot, ontology)
    contrastive = PairwiseContrastiveScorerV1(_FixedScorer()).rank("serialize vertical", skeleton, slot, ontology)
    assert likelihood == ParameterLikelihoodScorerV1(_FixedScorer()).rank("serialize vertical", skeleton, slot, ontology)
    assert contrastive == PairwiseContrastiveScorerV1(_FixedScorer()).rank("serialize vertical", skeleton, slot, ontology)
    assert "generate_text" not in inspect.getsource(ParameterLikelihoodScorerV1)


def test_forward_likelihood_uses_tensor_tokenizer_path_not_bare_chat_encoding() -> None:
    source = inspect.getsource(TransformersProvider.score_continuations)
    assert "tokenize=False" in source
    assert 'return_tensors="pt"' in source
    assert "Encoding" in source  # documents the pinned-runtime compatibility.
    assert "logits_to_keep" in source and "use_cache=False" in source


def test_selective_repair_v2_preserves_family_skeleton_order_and_unaffected_slots() -> None:
    frozen = _frozen(); baseline = frozen["s2_baseline_programs"]["parameter_path_01"]["program"]
    skeleton = skeleton_by_id("CV027"); ontology = ParameterSemanticOntologyV1()
    assert skeleton is not None
    current = choices_from_program(skeleton, baseline)
    decisions = {key: ParameterSemanticIRV1(key, None, "TEST", "AMBIGUOUS") for key in current}
    orientation = next(slot for slot in parameter_slots(skeleton) if slot.parameter == "orientation")
    target = next(candidate for candidate in ontology.candidates_for_slot(orientation) if candidate.choice == SemanticChoice("SOURCE", "HORIZONTAL"))
    decisions[orientation.key] = ParameterSemanticIRV1(orientation.key, target, "TEST", "HIGH")
    repaired = SelectiveParameterRepairV2().repair(skeleton, baseline, decisions)
    assert repaired.changed_slots == (orientation.key,)
    old_steps = baseline["hypotheses"][0]["steps"]; new_steps = repaired.program["hypotheses"][0]["steps"]
    assert [step["macro_id"] for step in old_steps] == [step["macro_id"] for step in new_steps]
    assert old_steps[0] == new_steps[0] and old_steps[1] == new_steps[1]
    assert new_steps[2]["params"]["orientation"] == {"source": "HORIZONTAL"}
    source = inspect.getsource(SelectiveParameterRepairV2)
    assert "canonical" not in source and "semantic_label" not in source


def test_partial_multislot_repair_is_allowed_without_all_or_nothing_patch() -> None:
    frozen = _frozen(); baseline = frozen["s2_baseline_programs"]["two_step_count_generate_01"]["program"]
    skeleton = skeleton_by_id("CV005"); ontology = ParameterSemanticOntologyV1()
    assert skeleton is not None
    current = choices_from_program(skeleton, baseline)
    decisions = {key: ParameterSemanticIRV1(key, None, "TEST", "AMBIGUOUS") for key in current}
    orientation = next(slot for slot in parameter_slots(skeleton) if slot.parameter == "orientation")
    candidate = next(item for item in ontology.candidates_for_slot(orientation) if item.choice == SemanticChoice("SOURCE", "HORIZONTAL"))
    decisions[orientation.key] = ParameterSemanticIRV1(orientation.key, candidate, "TEST", "HIGH")
    result = SelectiveParameterRepairV2().repair(skeleton, baseline, decisions)
    assert result.changed_slots == (orientation.key,)
    assert result.program["hypotheses"][0]["steps"][1]["params"]["shape_source"] == {"literal": [1, 1]}
