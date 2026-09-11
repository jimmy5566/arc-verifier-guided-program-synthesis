"""Thin, hard-verified sanity solvers composed only from V4 registry IDs."""
from __future__ import annotations

import numpy as np

from capabilities.composer import GenericProgramComposerV1
from capabilities.execution import Status
from capabilities.pipeline import CapabilityProgram, CapabilityProgramExecutor
from primitives.program import Step


class _RegistryProgramSolver:
    programs: list[CapabilityProgram]

    def predict(self, grid: np.ndarray, top_k: int = 1) -> list[np.ndarray]:
        executor = CapabilityProgramExecutor()
        predictions: list[np.ndarray] = []
        for program in self.programs:
            result = executor.execute(program, grid)
            if result.status == Status.SUCCESS and isinstance(result.value, np.ndarray):
                predictions.append(result.value)
            if len(predictions) >= top_k:
                break
        return predictions


class EnclosureFillSolver(_RegistryProgramSolver):
    def fit(self, task):
        self.programs = GenericProgramComposerV1((), max_depth=4).verified_region_fill(task)
        return self


class CountGenerationSolver(_RegistryProgramSolver):
    def fit(self, task):
        self.programs, _stats = GenericProgramComposerV1((), max_depth=4).compose_task(
            task, families=("counting", "generation")
        )
        return self


class PathSerializationSolver(_RegistryProgramSolver):
    """Serialize a verified simple horizontal/vertical pixel path.

    The program uses graph construction, deterministic shortest-path traversal,
    registered path-colour extraction, and registered row/column serialization.
    No solver-local graph or sequence implementation is used.
    """

    def fit(self, task):
        self.programs, _stats = GenericProgramComposerV1((), max_depth=4).compose_task(task, families=("graph_path",))
        return self
