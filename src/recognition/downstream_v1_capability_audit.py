"""Train-only oracle capability audit for frozen DOWNSTREAM_V1.

The audit has no task-id branches. Candidate domains come only from the frozen
Macro registry/compiler and colors/positions observed in each task's train
pairs. Candidate test predictions are materialized and hashed before the
development solution file is opened by the caller.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from arc.task import ARCTask
from capabilities.execution import Status
from llm.compiler_aware_interface import CompilerValidSkeleton, compiler_valid_skeletons
from llm.macro_compiler_v1 import MacroProgramCompilerV1, MacroProgramExecutorV1
from llm.macro_dsl import MacroHypothesis, MacroStatus, MacroStep


@dataclass(frozen=True)
class OracleCandidate:
    skeleton_id: str
    hypothesis: MacroHypothesis
    predictions: tuple[list[list[int]], ...]


def _colors(task: ARCTask) -> tuple[int, ...]:
    changed: set[int] = set()
    for example in task.train:
        source, target = example.input.values, example.output.values
        if source.shape == target.shape:
            changed.update(map(int, target[source != target]))
        else:
            changed.update(map(int, np.unique(target)))
    return tuple(sorted(changed or {0}))


def _options(macro_id: str, task: ARCTask) -> tuple[dict[str, Any], ...]:
    colors = _colors(task)
    if macro_id == "COMPLETE_SYMMETRY": return tuple({"axis": {"literal": value}} for value in ("HORIZONTAL", "VERTICAL", "MAIN", "ANTI"))
    if macro_id == "FILL_ENCLOSED_REGION":
        return tuple({"color_source": {"literal": value}} for value in colors) + ({"color_source": {"source": "FROM_UNIQUE_COLOR"}}, {"color_source": {"source": "FROM_TRAIN_COLOR_MAPPING"}}, {"color_source": {"source": "FROM_BOUNDARY"}})
    if macro_id == "COUNT_STRUCTURES":
        return tuple({"count_kind": {"literal": value}} for value in ("OBJECT_COUNT", "REGION_COUNT", "RUN_COUNT")) + tuple({"count_kind": {"literal": "COLOR_CELL_COUNT"}, "color_source": {"literal": value}} for value in colors) + ({"count_kind": {"literal": "COLOR_CELL_COUNT"}, "color_source": {"source": "FROM_UNIQUE_COLOR"}},)
    if macro_id == "GENERATE_FROM_COUNT":
        return tuple({"shape_source": shape, "color_source": color, "orientation": orientation} for shape in ({"literal": [1, 1]}, {"source": "SAME_AS_INPUT"}, {"source": "FROM_TRAIN_OUTPUT_RELATION"}) for color in tuple({"literal": value} for value in colors) + ({"source": "FROM_UNIQUE_COLOR"}, {"source": "FROM_TRAIN_COLOR_MAPPING"}) for orientation in ({"literal": "HORIZONTAL"}, {"literal": "VERTICAL"}, {"source": "HORIZONTAL"}, {"source": "VERTICAL"}))
    if macro_id == "EXTRACT_SEQUENCE":
        max_row = min(example.input.values.shape[0] for example in task.train); max_col = min(example.input.values.shape[1] for example in task.train)
        return tuple({"orientation": {"literal": "HORIZONTAL"}, "position": {"literal": value}} for value in range(max_row)) + tuple({"orientation": {"literal": "VERTICAL"}, "position": {"literal": value}} for value in range(max_col))
    if macro_id == "TRANSFORM_SEQUENCE": return ({"operation": {"literal": "reverse"}},)
    if macro_id == "BUILD_GRID_FROM_SEQUENCE": return ({"orientation": {"literal": "HORIZONTAL"}}, {"orientation": {"literal": "VERTICAL"}})
    if macro_id in {"FIND_OBJECTS", "FIND_PATH"}: return ({},)
    if macro_id == "SELECT_OBJECT": return tuple({"selector": {"literal": value}} for value in ("largest", "smallest", "unique_area", "unique_shape", "unique_color", "border", "non_border"))
    if macro_id == "TRANSFORM_OBJECT": return ({"operation": {"literal": "crop"}},) + tuple({"operation": {"literal": "recolor"}, "color_source": {"literal": value}} for value in colors) + ({"operation": {"literal": "recolor"}, "color_source": {"source": "FROM_UNIQUE_COLOR"}}, {"operation": {"literal": "recolor"}, "color_source": {"source": "FROM_TRAIN_COLOR_MAPPING"}})
    if macro_id == "TRACE_PATH": return ({"direction": {"literal": "PATH_ENDPOINT_ORDER"}}, {"direction": {"source": "PATH_ENDPOINT_ORDER"}})
    if macro_id == "SERIALIZE_PATH": return ({"orientation": {"literal": "HORIZONTAL"}}, {"orientation": {"literal": "VERTICAL"}}, {"orientation": {"source": "HORIZONTAL"}}, {"orientation": {"source": "VERTICAL"}})
    raise ValueError(f"no finite frozen candidate domain for {macro_id}")


def _shape_compatible(skeleton: CompilerValidSkeleton, task: ARCTask) -> bool:
    preserve = {"COMPLETE_SYMMETRY", "FILL_ENCLOSED_REGION"}
    changes_shape = any(example.input.values.shape != example.output.values.shape for example in task.train)
    if all(macro in preserve for macro in skeleton.macro_ids): return not changes_shape
    return True


def candidates(task: ARCTask) -> Iterable[tuple[str, MacroHypothesis]]:
    for skeleton in compiler_valid_skeletons():
        if not _shape_compatible(skeleton, task): continue
        for params in itertools.product(*(_options(macro_id, task) for macro_id in skeleton.macro_ids)):
            yield skeleton.skeleton_id, MacroHypothesis("downstream_v1_oracle", tuple(MacroStep(macro_id, value) for macro_id, value in zip(skeleton.macro_ids, params)))


def find_train_consistent_oracle(task: ARCTask) -> tuple[OracleCandidate | None, dict[str, int]]:
    compiler = MacroProgramCompilerV1(allow_direct_literals=True); executor = MacroProgramExecutorV1(); stats = {"generated": 0, "compiled": 0, "train_consistent": 0}
    for skeleton_id, hypothesis in candidates(task):
        stats["generated"] += 1
        compiled = compiler.compile(hypothesis, task)
        if compiled.status != MacroStatus.COMPILED or compiled.program is None: continue
        stats["compiled"] += 1
        # A first-pair mismatch is a deterministic, lossless rejection and
        # avoids repeatedly executing every train pair for most candidates.
        first = executor.execute(compiled.program, task.train[0].input.values)
        if first.status != Status.SUCCESS or not np.array_equal(first.value, task.train[0].output.values): continue
        if executor.verify_result(compiled.program, task).status != Status.TRAIN_CONSISTENT: continue
        stats["train_consistent"] += 1
        outputs = tuple(np.asarray(executor.execute(compiled.program, example.input.values).value).astype(int).tolist() for example in task.test)
        return OracleCandidate(skeleton_id, hypothesis, outputs), stats
    return None, stats


def prediction_hash(predictions: Iterable[list[list[int]]]) -> str:
    return hashlib.sha256(json.dumps(list(predictions), separators=(",", ":")).encode()).hexdigest()
