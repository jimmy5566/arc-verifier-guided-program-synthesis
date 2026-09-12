"""Solution-aware, development-only forensics for frozen DOWNSTREAM_V1.

This module is intentionally the *only* new module that reads the development
solution file.  It is an offline diagnostic, never a production solver: every
conclusion is derived from a task-generic finite candidate space and a final
oracle equality check.  Sensitive task identifiers, programs, and grids stay
in an ignored private artifact; callers receive only safe aggregates.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from arc.io import load_dataset
from arc.task import ARCTask
from recognition.downstream_v1_capability_audit import OracleCandidate, candidates
from capabilities.execution import Status
from llm.macro_compiler_v1 import MacroProgramCompilerV1, MacroProgramExecutorV1
from llm.macro_dsl import MacroStatus
from solvers.library_v0 import (
    FixedCropSolver,
    ForegroundBBoxCropSolver,
    GlobalTransformSolver,
    RecolorSolver,
    TranslationSolver,
)
from solvers.library_v1_object import ObjectCopyMoveSolver, ObjectSelectionSolver, ObjectTransformSolver
from solvers.library_v2_relation import ObjectAlignmentSolver, RelationalSelectionSolver, TwoObjectCompositionSolver
from solvers.library_v3_pattern import PatternRepairSolver, PeriodicCompletionSolver, SymmetryCompletionSolver


REPRESENTABILITY = (
    "REPRESENTABLE_EXISTING",
    "REPRESENTABLE_BUT_SEARCH_MISSED",
    "DSL_VALID_BUT_COMPILER_UNSUPPORTED",
    "MISSING_PRIMITIVE",
    "MISSING_RELATIONAL_CAPABILITY",
    "MISSING_COMPOSITION",
    "MISSING_OUTPUT_CONSTRUCTION",
    "MISSING_ITERATION_OR_REPEAT",
    "MISSING_CONDITIONAL_OR_ROLE_LOGIC",
    "MISSING_PATTERN_COMPLETION",
    "OTHER_CAPABILITY_GAP",
)

SEARCH_COVERAGE = (
    "ENUMERATED_AND_FOUND",
    "ENUMERATED_BUT_PRUNED",
    "REPRESENTABLE_BUT_NOT_ENUMERATED",
    "PARAMETER_SPACE_MISSED",
    "COMPOSITION_SPACE_MISSED",
    "NOT_APPLICABLE",
)

DEVELOPMENT_SOLUTION_FILENAME = "arc-agi_training_solutions.json"


@dataclass(frozen=True)
class OracleTaskRecord:
    task_id: str
    transformation_family: str
    classification: str
    search_coverage: str
    frozen_stats: Mapping[str, int]
    compiler_oracle_count: int
    compiler_test_exact_count: int
    witness_family: str | None
    legacy_lookup: str


def stable_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def select_development_cohort(task_ids: Iterable[str], excluded_ids: Iterable[str], *, size: int, salt: str) -> tuple[str, ...]:
    """Freeze a task-outcome-independent deterministic development cohort."""
    excluded = set(excluded_ids)
    eligible = sorted(set(task_ids) - excluded)
    return tuple(sorted(eligible, key=lambda task_id: hashlib.sha256(f"{salt}:{task_id}".encode("utf-8")).hexdigest())[:size])


def cohort_public_record(task_ids: Sequence[str], *, eligible_count: int, excluded_count: int, salt: str) -> dict[str, Any]:
    """A public cohort commitment: no development task identifiers are exposed."""
    return {
        "config_id": "DOWNSTREAM_CAPABILITY_GAP_FORENSICS_V1_TASKS",
        "split": "development",
        "task_count": len(task_ids),
        "selection": "deterministic SHA-256 sample from eligible development tasks before oracle analysis",
        "selection_salt": salt,
        "eligible_development_pool_size": eligible_count,
        "excluded_historical_prompt_study_count": excluded_count,
        "cohort_commitment_hash": stable_hash(list(task_ids)),
        "task_ids_public": False,
        "raw_grids_public": False,
        "solutions_public": False,
        "oracle_programs_public": False,
    }


def exact_predictions(predictions: Sequence[object], expected: Sequence[object]) -> bool:
    """JSON container shape must not affect an ARC grid equality decision."""
    if len(predictions) != len(expected):
        return False
    try:
        return all(np.array_equal(np.asarray(prediction), np.asarray(target)) for prediction, target in zip(predictions, expected))
    except (TypeError, ValueError):
        return False


def _prediction_payload(candidate: OracleCandidate) -> list[list[list[int]]]:
    return [[list(map(int, row)) for row in grid] for grid in candidate.predictions]


def _all_compiler_oracles(task: ARCTask, expected: Sequence[object]) -> tuple[list[OracleCandidate], dict[str, int], list[OracleCandidate]]:
    """Enumerate exactly the frozen finite candidate space, preserving all hits."""
    compiler = MacroProgramCompilerV1(allow_direct_literals=True)
    executor = MacroProgramExecutorV1()
    stats = Counter(generated=0, compiled=0, train_consistent=0)
    train_consistent: list[OracleCandidate] = []
    test_exact: list[OracleCandidate] = []
    for skeleton_id, hypothesis in candidates(task):
        stats["generated"] += 1
        compiled = compiler.compile(hypothesis, task)
        if compiled.status != MacroStatus.COMPILED or compiled.program is None:
            continue
        stats["compiled"] += 1
        first = executor.execute(compiled.program, task.train[0].input.values)
        if first.status != Status.SUCCESS or not np.array_equal(first.value, task.train[0].output.values):
            continue
        if executor.verify_result(compiled.program, task).status != Status.TRAIN_CONSISTENT:
            continue
        stats["train_consistent"] += 1
        outputs: list[list[list[int]]] = []
        valid = True
        for example in task.test:
            result = executor.execute(compiled.program, example.input.values)
            if result.status != Status.SUCCESS:
                valid = False
                break
            outputs.append(np.asarray(result.value).astype(int).tolist())
        if not valid:
            continue
        candidate = OracleCandidate(skeleton_id, hypothesis, tuple(outputs))
        train_consistent.append(candidate)
        if exact_predictions(_prediction_payload(candidate), expected):
            test_exact.append(candidate)
    return train_consistent, dict(stats), test_exact


_WITNESS_SOLVERS: tuple[tuple[str, object, str], ...] = (
    ("GLOBAL_TRANSFORM", GlobalTransformSolver, "MISSING_COMPOSITION"),
    ("RECOLOR", RecolorSolver, "MISSING_COMPOSITION"),
    ("FIXED_CROP", FixedCropSolver, "MISSING_COMPOSITION"),
    ("FOREGROUND_BBOX_CROP", ForegroundBBoxCropSolver, "MISSING_COMPOSITION"),
    ("TRANSLATION", TranslationSolver, "MISSING_COMPOSITION"),
    ("OBJECT_SINGLE", ObjectSelectionSolver, "DSL_VALID_BUT_COMPILER_UNSUPPORTED"),
    ("OBJECT_TRANSFORM", ObjectTransformSolver, "DSL_VALID_BUT_COMPILER_UNSUPPORTED"),
    ("OBJECT_COPY_MOVE", ObjectCopyMoveSolver, "DSL_VALID_BUT_COMPILER_UNSUPPORTED"),
    ("RELATIONAL_SELECTION", RelationalSelectionSolver, "DSL_VALID_BUT_COMPILER_UNSUPPORTED"),
    ("OBJECT_ALIGNMENT", ObjectAlignmentSolver, "DSL_VALID_BUT_COMPILER_UNSUPPORTED"),
    ("TWO_OBJECT_COMPOSITION", TwoObjectCompositionSolver, "DSL_VALID_BUT_COMPILER_UNSUPPORTED"),
    ("PERIODIC_COMPLETION", PeriodicCompletionSolver, "DSL_VALID_BUT_COMPILER_UNSUPPORTED"),
    ("PATTERN_REPAIR", PatternRepairSolver, "DSL_VALID_BUT_COMPILER_UNSUPPORTED"),
    ("SYMMETRY_COMPLETION", SymmetryCompletionSolver, "DSL_VALID_BUT_COMPILER_UNSUPPORTED"),
)


def _witness(task: ARCTask, expected: Sequence[object]) -> tuple[str | None, str | None]:
    """Use existing generic solvers only as evidence, never as an oracle rule."""
    for family, solver_type, classification in _WITNESS_SOLVERS:
        solver = solver_type().fit(task)  # type: ignore[operator]
        predictions = solver.predict(task.test[0].input.values, top_k=1)
        if predictions and exact_predictions([np.asarray(predictions[0]).astype(int).tolist()], expected):
            return family, classification
    return None, None


def _shape_modes(task: ARCTask, expected: Sequence[object]) -> tuple[bool, bool, bool]:
    pairs = [(example.input.values.shape, example.output.values.shape) for example in task.train]
    pairs += [(example.input.values.shape, np.asarray(target).shape) for example, target in zip(task.test, expected)]
    same = all(source == target for source, target in pairs)
    smaller = all(target[0] <= source[0] and target[1] <= source[1] for source, target in pairs) and any(source != target for source, target in pairs)
    larger = any(target[0] > source[0] or target[1] > source[1] for source, target in pairs)
    return same, smaller, larger


def _has_repeat(grid: np.ndarray) -> bool:
    rows, cols = grid.shape
    return (rows >= 4 and any(np.array_equal(grid[:period], grid[period:2 * period]) for period in range(1, rows // 2 + 1))) or (cols >= 4 and any(np.array_equal(grid[:, :period], grid[:, period:2 * period]) for period in range(1, cols // 2 + 1)))


def _gap_from_oracle_shapes(task: ARCTask, expected: Sequence[object]) -> tuple[str, str]:
    """Deterministic taxonomy for tasks with no existing executable witness."""
    same, smaller, larger = _shape_modes(task, expected)
    outputs = [example.output.values for example in task.train] + [np.asarray(target) for target in expected]
    if larger:
        return "OUTPUT_CONSTRUCTION", "MISSING_OUTPUT_CONSTRUCTION"
    if any(_has_repeat(output) for output in outputs):
        return "REPETITION_OR_PATTERN", "MISSING_ITERATION_OR_REPEAT"
    if smaller:
        return "OBJECT_OR_REGION_EXTRACTION", "MISSING_RELATIONAL_CAPABILITY"
    if same:
        return "SAME_CANVAS_CONDITIONAL_TRANSFORMATION", "MISSING_CONDITIONAL_OR_ROLE_LOGIC"
    return "UNCLASSIFIED_GRID_TRANSFORMATION", "OTHER_CAPABILITY_GAP"


def _legacy_lookup(task_id: str, legacy_hashes: set[str]) -> str:
    # The former artifact saved only test-prediction hashes, not canonical
    # program fingerprints.  It can identify an affected record but cannot
    # prove program equivalence; frozen re-enumeration above is authoritative.
    return "LEGACY_TRAIN_CONSISTENT_RECORD" if task_id in legacy_hashes else "NO_LEGACY_RECORD"


def audit_task(task: ARCTask, expected: Sequence[object], *, legacy_ids: set[str]) -> tuple[OracleTaskRecord, dict[str, Any]]:
    train_consistent, stats, test_exact = _all_compiler_oracles(task, expected)
    if test_exact:
        record = OracleTaskRecord(task.task_id, "FROZEN_MACRO_COMPILER_PROGRAM", "REPRESENTABLE_EXISTING", "ENUMERATED_AND_FOUND", stats, len(train_consistent), len(test_exact), None, _legacy_lookup(task.task_id, legacy_ids))
        private = {"task_id": task.task_id, "classification": record.classification, "search_coverage": record.search_coverage, "transformation_family": record.transformation_family, "oracle_program_fingerprints": [stable_hash(candidate.hypothesis.to_dict()) for candidate in test_exact]}
        return record, private
    witness_family, witness_classification = _witness(task, expected)
    if witness_family is not None and witness_classification is not None:
        family, classification = witness_family, witness_classification
    else:
        family, classification = _gap_from_oracle_shapes(task, expected)
    record = OracleTaskRecord(task.task_id, family, classification, "NOT_APPLICABLE", stats, len(train_consistent), 0, witness_family, _legacy_lookup(task.task_id, legacy_ids))
    private = {"task_id": task.task_id, "classification": record.classification, "search_coverage": record.search_coverage, "transformation_family": record.transformation_family, "compiler_train_consistent_count": len(train_consistent), "compiler_oracle_program_fingerprints": []}
    return record, private


def run_solution_aware_audit(challenge_path: Path, solution_path: Path, task_ids: Sequence[str], legacy_private_path: Path) -> tuple[list[OracleTaskRecord], list[dict[str, Any]]]:
    """The guarded oracle boundary: solution access is local to this audit module."""
    tasks = load_dataset(challenge_path)
    expected_by_task = json.loads(solution_path.read_text(encoding="utf-8"))
    legacy = json.loads(legacy_private_path.read_text(encoding="utf-8")) if legacy_private_path.exists() else {}
    legacy_ids = set(legacy.get("train_consistent", {}))
    records: list[OracleTaskRecord] = []
    private: list[dict[str, Any]] = []
    for task_id in task_ids:
        record, detail = audit_task(tasks[task_id], expected_by_task[task_id], legacy_ids=legacy_ids)
        records.append(record); private.append(detail)
    return records, private


def audit_legacy_serialized_scores(solution_path: Path, legacy_private_path: Path) -> dict[str, Any]:
    """Re-score a legacy private artifact with grid equality, not list identity."""
    if not legacy_private_path.exists():
        return {"legacy_records": 0, "genuine_test_wrong": None, "status": "LEGACY_PRIVATE_ARTIFACT_UNAVAILABLE"}
    legacy = json.loads(legacy_private_path.read_text(encoding="utf-8"))
    expected_by_task = json.loads(solution_path.read_text(encoding="utf-8"))
    records = legacy.get("train_consistent", {})
    exact = sum(exact_predictions(item["prediction"], expected_by_task[task_id]) for task_id, item in records.items())
    return {
        "legacy_records": len(records),
        "serialization_false_negative_count": exact,
        "genuine_test_wrong": len(records) - exact,
        "status": "LEGACY_TUPLE_LIST_COMPARISON_FALSE_NEGATIVE" if exact else "NO_SERIALIZATION_FALSE_NEGATIVE",
    }


def legacy_individual_dispositions(solution_path: Path, legacy_private_path: Path) -> list[dict[str, str]]:
    """Private per-record disposition for the four formerly apparent failures."""
    if not legacy_private_path.exists():
        return []
    legacy = json.loads(legacy_private_path.read_text(encoding="utf-8"))
    expected_by_task = json.loads(solution_path.read_text(encoding="utf-8"))
    details: list[dict[str, str]] = []
    for task_id, item in sorted(legacy.get("train_consistent", {}).items()):
        equal = exact_predictions(item["prediction"], expected_by_task[task_id])
        details.append({
            "task_id": task_id,
            "disposition": "SERIALIZATION_COMPARISON_FALSE_NEGATIVE" if equal else "GENUINE_TRAIN_CONSISTENT_TEST_WRONG",
            "generalization_category": "NOT_APPLICABLE_SERIALIZATION_ARTIFACT" if equal else "OTHER_GENERALIZATION_FAILURE",
        })
    return details


def aggregate_records(records: Sequence[OracleTaskRecord]) -> dict[str, Any]:
    categories = Counter(record.classification for record in records)
    search = Counter(record.search_coverage for record in records)
    witnesses = Counter(record.witness_family for record in records if record.witness_family)
    representable = categories["REPRESENTABLE_EXISTING"] + categories["REPRESENTABLE_BUT_SEARCH_MISSED"]
    return {
        "task_count": len(records),
        "representable": representable,
        "representability_rate": 0.0 if not records else representable / len(records),
        "classification_counts": {name: categories[name] for name in REPRESENTABILITY},
        "search_coverage_counts": {name: search[name] for name in SEARCH_COVERAGE},
        "witness_family_counts": dict(sorted(witnesses.items())),
        "compiler_gap": categories["DSL_VALID_BUT_COMPILER_UNSUPPORTED"],
        "capability_gap": sum(categories[name] for name in REPRESENTABILITY if name.startswith("MISSING_") or name == "OTHER_CAPABILITY_GAP"),
        "search_found_oracle": search["ENUMERATED_AND_FOUND"],
        "search_missed": categories["REPRESENTABLE_BUT_SEARCH_MISSED"],
    }
