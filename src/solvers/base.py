from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol
import numpy as np
from arc.task import ARCTask

@dataclass
class Candidate:
    name: str; program: object; training_accuracy: float; exact_match: bool; complexity: int = 1; metadata: dict | None = None
    def apply(self, grid: np.ndarray) -> np.ndarray: return self.program.apply(grid)

class Solver(Protocol):
    def fit(self, task: ARCTask): ...
    def generate_candidates(self, task: ARCTask) -> list[Candidate]: ...
    def predict(self, test_input: np.ndarray, top_k: int = 2) -> list[np.ndarray]: ...
