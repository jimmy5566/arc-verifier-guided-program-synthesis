from __future__ import annotations

import inspect
from pathlib import Path

from recognition.downstream_v1_capability_audit import candidates
from recognition.grid_feature_extractor import task_features
from recognition.structured_hypothesis_ir import StructuredHypothesisIRV1, schema
from arc.task import ARCExample, ARCGrid, ARCTask


def test_structured_ir_is_finite_and_rejects_free_text_family() -> None:
    ir = StructuredHypothesisIRV1(("SYMMETRY",), "SYMMETRY", (), ("HORIZONTAL",), "complete_symmetry", {"axis": ("AXIS.LITERAL.HORIZONTAL",)}, {"pair_count": 1}, {"primary_family": 1.0})
    assert ir.public()["schema"] == "StructuredHypothesisIRV1"
    assert "task_id" in schema()["forbidden"]


def test_feature_extractor_uses_only_train_pairs() -> None:
    task = ARCTask("fixture", (ARCExample(ARCGrid([[0, 1]]), ARCGrid([[0, 1]])),), (ARCExample(ARCGrid([[1, 0]]), None),))
    features = task_features(task)
    assert features["pair_count"] == len(task.train) and "test" not in repr(features).lower()


def test_oracle_audit_has_no_task_id_branches_or_solution_access() -> None:
    source = inspect.getsource(candidates)
    assert "task_id" not in source and "solutions" not in source and "canonical" not in source


def test_gate_freezes_predictions_before_solution_read() -> None:
    source = (Path(__file__).parents[1] / "scripts/build_grid_recognition_diagnostic_tasks.py").read_text(encoding="utf-8")
    assert source.index("freeze_hash") < source.index("solutions =")
    assert "BLOCKED_CAPABILITY_COVERAGE" in source
