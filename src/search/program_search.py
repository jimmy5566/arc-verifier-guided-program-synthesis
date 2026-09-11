"""Bounded compositional search with exact train-pair pruning."""
from __future__ import annotations
from dataclasses import dataclass
from time import perf_counter
import numpy as np
from arc.task import ARCTask
from transforms import TransformPipeline
from verification.verifier import verify_pipeline

@dataclass
class ProgramCandidate:
    operations: tuple; depth: int; training_accuracy: float; exact_match: bool; complexity_score: int; runtime_seconds: float
    def apply(self, grid): return TransformPipeline(self.operations).apply(grid)

def beam_search(task: ARCTask, primitives: list, max_depth: int=2, beam_size: int=20) -> list[ProgramCandidate]:
    started=perf_counter(); beam=[tuple()]; results=[]; seen=set()
    for depth in range(1,max_depth+1):
        generated=[]
        for ops in beam:
            for op in primitives:
                candidate_ops=ops+(op,); pipeline=TransformPipeline(candidate_ops)
                signature=tuple(pipeline.apply(ex.input.values).tobytes() for ex in task.train)
                if signature in seen: continue
                seen.add(signature); reports=verify_pipeline(task,pipeline); score=float(np.mean([r.pixel_accuracy for r in reports])); generated.append(ProgramCandidate(candidate_ops,depth,score,all(r.exact_match for r in reports),depth,perf_counter()-started))
        generated.sort(key=lambda x:(x.exact_match,x.training_accuracy,-x.complexity_score),reverse=True); results.extend(generated); beam=[x.operations for x in generated[:beam_size]]
    return results
