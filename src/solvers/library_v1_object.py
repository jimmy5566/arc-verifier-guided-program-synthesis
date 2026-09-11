"""Solver Library v1 object baselines: bounded, train-consistent single-object hypotheses."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from arc.task import ARCTask
from representations.objects import extract_objects
from representations.object_ops import crop_object, recolor_object, remove_object, render_object
from representations.selectors import ObjectSelector, SELECTORS
from .base import Candidate

def _objects(grid: np.ndarray): return extract_objects(grid,4,background=int(np.bincount(np.asarray(grid).ravel()).argmax()))

@dataclass(frozen=True)
class ObjectProgram:
    selector: ObjectSelector; operation: str; transform: str='identity'; color: int | None=None; shift: tuple[int,int]=(0,0)
    def metadata(self): return {'selection':self.selector.name,'operation':self.operation,'transform':self.transform,'color':self.color,'shift':self.shift}
    def render(self, grid: np.ndarray) -> np.ndarray | None:
        selected=self.selector.select(_objects(grid))
        if len(selected)!=1: return None
        obj=selected[0]
        if self.operation=='crop': return crop_object(obj,grid,self.transform,self.color)
        if self.operation=='recolor': return recolor_object(grid,obj,self.color)  # type: ignore[arg-type]
        if self.operation=='remove': return remove_object(grid,obj)
        if self.operation=='translate': return render_object(grid,obj,*self.shift,copy=False,transform=self.transform,color=self.color)
        if self.operation=='copy': return render_object(grid,obj,*self.shift,copy=True,transform=self.transform,color=self.color)
        raise ValueError(f'unknown operation {self.operation}')
    def apply(self, grid: np.ndarray) -> np.ndarray:
        rendered=self.render(grid)
        # Candidate is rejected before prediction when train rendering fails. This fallback preserves API compatibility.
        return np.asarray(grid).copy() if rendered is None else rendered

def _consistent(task: ARCTask, program: ObjectProgram) -> bool:
    for example in task.train:
        output=program.render(example.input.values)
        if output is None or not np.array_equal(output,example.output.values): return False
    return True

def _translation_parameters(task: ARCTask, selector: ObjectSelector) -> set[tuple[str,tuple[int,int]]]:
    """Derive shape-preserving candidate offsets from train outputs, then exact verification guards them."""
    shared: set[tuple[str,tuple[int,int]]] | None=None
    transforms=('identity','rotate90','rotate180','rotate270','flip_horizontal','flip_vertical')
    for example in task.train:
        chosen=selector.select(_objects(example.input.values))
        if len(chosen)!=1: return set()
        obj=chosen[0]; possibilities=set()
        for transform in transforms:
            source=crop_object(obj,example.input.values,transform)
            for target in _objects(example.output.values):
                patch=crop_object(target,example.output.values)
                if np.array_equal(source,patch): possibilities.add((transform,(target.top-obj.top,target.left-obj.left)))
        shared=possibilities if shared is None else shared & possibilities
    return shared or set()

class _ObjectSolverBase:
    candidates: list[Candidate]
    generated_candidates: int
    rejected_candidates: int
    def fit(self, task: ARCTask):
        programs=self._programs(task); self.generated_candidates=len(programs); consistent=[]
        for program in programs:
            if _consistent(task,program):
                meta=program.metadata(); complexity=1+(meta['operation'] not in ('crop','remove','recolor'))+(meta['transform']!='identity')
                consistent.append(Candidate(f"{meta['selection']}->{meta['operation']}:{meta['transform']}",program,1.0,True,complexity,meta))
        self.rejected_candidates=self.generated_candidates-len(consistent)
        self.candidates=sorted(consistent,key=lambda c:(c.complexity,c.name)); return self
    def predict(self, test_input: np.ndarray, top_k: int=1) -> list[np.ndarray]:
        predictions=[]
        for candidate in self.candidates:
            rendered=candidate.program.render(test_input)
            if rendered is not None: predictions.append(rendered)
            if len(predictions)>=top_k: break
        return predictions

class ObjectSelectionSolver(_ObjectSolverBase):
    """Select one unambiguous object and return its unmodified bounding-box crop."""
    def _programs(self, task): return [ObjectProgram(selector,'crop') for selector in SELECTORS]

class ObjectTransformSolver(_ObjectSolverBase):
    """One selected object followed by exactly one render transform; no composition/search."""
    def _programs(self, task):
        programs=[]
        for selector in SELECTORS:
            programs += [ObjectProgram(selector,'crop',transform) for transform in ('rotate90','rotate180','rotate270','flip_horizontal','flip_vertical')]
            programs += [ObjectProgram(selector,'recolor',color=color) for color in range(10)]
            programs.append(ObjectProgram(selector,'remove'))
            programs += [ObjectProgram(selector,'translate',transform,shift=shift) for transform,shift in _translation_parameters(task,selector) if shift!=(0,0)]
        return programs

class ObjectCopyMoveSolver(_ObjectSolverBase):
    """Optional small extension: selected object -> copy -> fixed translation."""
    def _programs(self, task):
        return [ObjectProgram(selector,'copy',transform,shift=shift) for selector in SELECTORS for transform,shift in _translation_parameters(task,selector) if shift!=(0,0)]
