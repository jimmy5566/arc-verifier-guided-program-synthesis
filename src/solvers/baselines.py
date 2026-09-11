from __future__ import annotations
import numpy as np
from arc.task import ARCTask
from transforms import Identity, Rotate, FlipHorizontal, FlipVertical, Transpose, CropNonBackground
from verification.verifier import verify_pipeline
from .base import Candidate
from .object_rules import candidate_rules

class _BaseSolver:
    candidates: list[Candidate]
    def fit(self, task: ARCTask): self.candidates=self.generate_candidates(task); return self
    def predict(self, test_input: np.ndarray, top_k: int=2) -> list[np.ndarray]: return [candidate.apply(test_input) for candidate in self.candidates[:top_k]]

class IdentitySolver(_BaseSolver):
    def generate_candidates(self, task):
        program=Identity(); reports=verify_pipeline(task,program); return [Candidate("identity",program,float(np.mean([x.pixel_accuracy for x in reports])),all(x.exact_match for x in reports))]

class SimpleTransformSolver(_BaseSolver):
    def generate_candidates(self, task):
        programs=[Identity(),Rotate(90),Rotate(180),Rotate(270),FlipHorizontal(),FlipVertical(),Transpose(),CropNonBackground()]
        candidates=[]
        for program in programs:
            reports=verify_pipeline(task,program); candidates.append(Candidate(program.metadata()["operation"] if "operation" in program.metadata() else str(program),program,float(np.mean([x.pixel_accuracy for x in reports])),all(x.exact_match for x in reports)))
        return sorted(candidates,key=lambda c:(c.exact_match,c.training_accuracy,-c.complexity,c.name),reverse=True)

class ObjectRuleSolver(_BaseSolver):
    """EXP002 bounded, general object-level rule inference over all train pairs."""
    def __init__(self, candidate_budget: int=10): self.candidate_budget=candidate_budget
    def generate_candidates(self, task):
        candidates=[]
        for program in candidate_rules(task):
            reports=verify_pipeline(task,program)
            candidates.append(Candidate(
                name=f"{program.selector}:{program.operation}:{program.transform}", program=program,
                training_accuracy=float(np.mean([x.pixel_accuracy for x in reports])), exact_match=all(x.exact_match for x in reports),
                complexity=1 + (program.operation not in ('crop','keep_only','remove','recolor')) + (program.transform != 'identity'), metadata=program.metadata()))
        # Stable and auditable: exact train fit, partial fit, then simpler program, then lexical description.
        candidates.sort(key=lambda c:(not c.exact_match,-c.training_accuracy,c.complexity,c.name))
        return candidates[:self.candidate_budget]
