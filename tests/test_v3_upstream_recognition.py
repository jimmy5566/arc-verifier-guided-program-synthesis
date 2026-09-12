from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

from arc.task import ARCExample, ARCGrid, ARCTask
from v3.evidence.cross_pair import derive_cross_pair_evidence
from v3.evidence.extractor import extract_task_evidence
from v3.recognition.recognizer_interface import QwenRuleRecognizer, parse_complete_rulespec_hypotheses, parse_hypotheses
from v3.validation import RuleSpecPreflightValidator


ROOT = Path(__file__).parents[1]


class _Generated:
    def __init__(self, text: str) -> None:
        self.text = text


class _Provider:
    def __init__(self, text: str) -> None:
        self.text = text

    def generate_text(self, prompt: str, config: object) -> _Generated:
        assert "train_grids" in prompt and "[[9,9]]" not in prompt
        return _Generated(self.text)


def _task() -> ARCTask:
    return ARCTask("fixture", (ARCExample(ARCGrid([[0, 1], [0, 0]]), ARCGrid([[0, 2], [0, 0]])),), (ARCExample(ARCGrid([[9, 9]]), None),))


def test_qwen_recognizer_returns_only_unique_complete_rulespecs() -> None:
    raw = json.dumps({"hypotheses": [
        {"family": "RECOLOR", "operations": ["SELECT", "RECOLOR"], "parameters": {"$SELECTOR": "COLOR:1", "$TARGET_COLOR": 2}, "roles": {}, "repeat": None},
        {"family": "RECOLOR", "operations": ["SELECT", "RECOLOR"], "parameters": {"$SELECTOR": "COLOR:2", "$TARGET_COLOR": {"derive": "COLOR_OF", "arguments": {"object": {"role_ref": "reference"}}}}, "roles": {"reference": {"kind": "COLOR", "value": 3}}, "repeat": None},
    ]})
    task = _task(); evidence = extract_task_evidence(task)
    result = QwenRuleRecognizer(_Provider(raw), object()).recognize(task, evidence, derive_cross_pair_evidence(evidence), top_k=2)
    assert [item.to_dict()["skeleton"]["steps"][-1]["operation"] for item in result] == ["RECOLOR", "RECOLOR"]
    assert all(RuleSpecPreflightValidator().validate(item).passed for item in result)
    duplicate = json.dumps({"hypotheses": [
        {"family": "RECOLOR", "operations": ["SELECT", "RECOLOR"], "parameters": {"$SELECTOR": "COLOR:1", "$TARGET_COLOR": 2}, "roles": {}, "repeat": None},
        {"family": "RECOLOR", "operations": ["SELECT", "RECOLOR"], "parameters": {"$SELECTOR": "COLOR:1", "$TARGET_COLOR": 2}, "roles": {}, "repeat": None},
    ]})
    assert parse_complete_rulespec_hypotheses(duplicate, limit=2)[0] == ()
    assert parse_hypotheses(json.dumps({"hypotheses": [{"family": "GLOBAL", "operations": ["ROTATE"], "required_slots": []}]}), limit=1)[1] == "SCHEMA_FAILURE:wrong typed slots"


def test_track_u_source_has_no_downstream_or_gold_import_and_freeze_hashes_match() -> None:
    source = ROOT / "scripts/run_v3_rule_recognition.py"
    source_text = source.read_text(encoding="utf-8")
    imported = {
        node.module for node in ast.walk(ast.parse(source_text))
        if isinstance(node, ast.ImportFrom) and node.module
    }
    forbidden = ("binding", "execution", "verification", "macro", "compiler", "oracle", "solution")
    assert not any(any(term in module.lower() for term in forbidden) for module in imported)
    config = json.loads((ROOT / "configs/ARC2_V3_RULE_RECOGNITION_INDEPENDENT_V1_FROZEN_CONFIG.json").read_text(encoding="utf-8"))
    paths = {
        "evidence_extractor": ROOT / "src/v3/evidence/extractor.py",
        "cross_pair": ROOT / "src/v3/evidence/cross_pair.py",
        "recognizer": ROOT / "src/v3/recognition/recognizer_interface.py",
        "runner": ROOT / "scripts/run_v3_rule_recognition.py",
    }
    assert config["upstream_evidence_version"] == "U16"
    assert all(hashlib.sha256(path.read_bytes()).hexdigest().upper() == config["frozen_source_sha256"][name] for name, path in paths.items())
    assert "PREDICTIONS_FROZEN_BEFORE_GOLD_SCORING" in source_text


def test_prompt_provides_real_operation_slot_contract_without_placeholder_schema_echo() -> None:
    from v3.recognition.recognizer_interface import recognition_prompt
    task = _task(); evidence = extract_task_evidence(task)
    prompt = recognition_prompt(task, evidence, derive_cross_pair_evidence(evidence), top_k=3)
    assert "complete_rulespec_contract" in prompt
    assert '"STRING"' not in prompt and '"CANONICAL_OPERATION"' not in prompt and '"$TYPED_SLOT"' not in prompt
    assert "RECOLOR/FILL:$TARGET_COLOR" in prompt and "COLOR_OF" in prompt


def test_attachment_builder_excludes_gold_and_backend_dependencies() -> None:
    builder = (ROOT / "scripts/prepare_v3_upstream_recognition_source.py").read_text(encoding="utf-8")
    assert "forbidden_terms" in builder
    for forbidden in ("oracle", "solution", "backend_audit", "executor", "verifier", "macro_compiler"):
        assert f'"{forbidden}"' in builder
