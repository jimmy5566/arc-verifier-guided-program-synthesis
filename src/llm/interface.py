from __future__ import annotations
from typing import Protocol
from arc.task import ARCTask
class HypothesisGenerator(Protocol):
    """Future implementations return structured rule/program proposals, never final unverified grids alone."""
    def generate(self, task: ARCTask) -> list[dict]: ...
