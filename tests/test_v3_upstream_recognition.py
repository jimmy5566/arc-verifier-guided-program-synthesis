from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

from arc.task import ARCExample, ARCGrid, ARCTask
from v3.evidence.cross_pair import derive_cross_pair_evidence
from v3.evidence.extractor import extract_task_evidence
from v3.recognition.recognizer_interface import QwenRuleRecognizer, parse_hypotheses


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


def test_qwen_recognizer_returns_only_unique_parameter_free_exposed_skeletons() -> None:
    raw = json.dumps({"hypotheses": [
        {"family": "GLOBAL", "operations": ["ROTATE"], "required_slots": []},
        {"family": "GLOBAL", "operations": ["REFLECT"], "required_slots": []},
    ]})
    task = _task(); evidence = extract_task_evidence(task)
    result = QwenRuleRecognizer(_Provider(raw), object()).recognize(task, evidence, derive_cross_pair_evidence(evidence), top_k=2)
    assert [item.to_dict()["steps"][0]["operation"] for item in result] == ["ROTATE", "REFLECT"]
    duplicate = json.dumps({"hypotheses": [
        {"family": "GLOBAL", "operations": ["ROTATE"], "required_slots": []},
        {"family": "GLOBAL", "operations": ["ROTATE"], "required_slots": []},
    ]})
    assert parse_hypotheses(duplicate, limit=2)[0] == ()


def test_track_u_source_has_no_downstream_or_gold_import_and_freeze_hashes_match() -> None:
    source = ROOT / "scripts/run_v3_rule_recognition.py"
    source_text = source.read_text(encoding="utf-8")
    imported = {
        node.module for node in ast.walk(ast.parse(source_text))
        if isinstance(node, ast.ImportFrom) and node.module
    }
    forbidden = ("parameters", "rule_spec", "execution", "verification", "macro", "compiler", "oracle", "solution")
    assert not any(any(term in module.lower() for term in forbidden) for module in imported)
    config = json.loads((ROOT / "configs/ARC2_V3_RULE_RECOGNITION_INDEPENDENT_V1_FROZEN_CONFIG.json").read_text(encoding="utf-8"))
    paths = {
        "evidence_extractor": ROOT / "src/v3/evidence/extractor.py",
        "cross_pair": ROOT / "src/v3/evidence/cross_pair.py",
        "recognizer": ROOT / "src/v3/recognition/recognizer_interface.py",
        "runner": ROOT / "scripts/run_v3_rule_recognition.py",
    }
    assert config["upstream_evidence_version"] == "U8"
    assert all(hashlib.sha256(path.read_bytes()).hexdigest().upper() == config["frozen_source_sha256"][name] for name, path in paths.items())
    assert "PREDICTIONS_FROZEN_BEFORE_GOLD_SCORING" in source_text


def test_attachment_builder_excludes_gold_and_backend_dependencies() -> None:
    builder = (ROOT / "scripts/prepare_v3_upstream_recognition_source.py").read_text(encoding="utf-8")
    assert "forbidden_terms" in builder
    for forbidden in ("oracle", "solution", "backend_audit", "parameter", "rule_spec", "executor", "verifier", "macro_compiler"):
        assert f'"{forbidden}"' in builder
