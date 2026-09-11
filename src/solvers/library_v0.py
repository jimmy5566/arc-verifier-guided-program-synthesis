"""Solver Library v0: five small train-only, deterministic ARC baselines."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from arc.task import ARCTask
from transforms import Identity, Rotate, FlipHorizontal, FlipVertical, Transpose, AntiTranspose
from verification.verifier import verify_pipeline
from .base import Candidate

def _background(grid: np.ndarray) -> int:
    return int(np.bincount(np.asarray(grid).ravel()).argmax())

def _exact(program, task: ARCTask) -> bool:
    return all(report.exact_match for report in verify_pipeline(task,program))

class _SolverBase:
    candidates: list[Candidate]
    def fit(self, task: ARCTask): self.candidates=self.generate_candidates(task); return self
    def predict(self, test_input: np.ndarray, top_k: int=1) -> list[np.ndarray]: return [c.apply(test_input) for c in self.candidates[:top_k]]

class GlobalTransformSolver(_SolverBase):
    """A single whole-grid primitive must explain every train pair exactly."""
    def generate_candidates(self, task: ARCTask) -> list[Candidate]:
        programs=[Identity(),Rotate(90),Rotate(180),Rotate(270),FlipHorizontal(),FlipVertical(),Transpose(),AntiTranspose()]
        result=[]
        for program in programs:
            if _exact(program,task): result.append(Candidate(program.metadata()['operation'],program,1.0,True,1,program.metadata()))
        return result

@dataclass(frozen=True)
class ColorMapProgram:
    mapping: tuple[tuple[int,int], ...]
    def apply(self, grid: np.ndarray) -> np.ndarray:
        out=np.asarray(grid).copy()
        for source,target in self.mapping: out[np.asarray(grid)==source]=target
        return out
    def metadata(self): return {'operation':'recolor','mapping':dict(self.mapping)}

class RecolorSolver(_SolverBase):
    """Infer one consistent colour map across all same-shape train pairs."""
    def generate_candidates(self, task: ARCTask) -> list[Candidate]:
        mapping: dict[int,int]={}
        for example in task.train:
            source,target=example.input.values,example.output.values
            if source.shape != target.shape: return []
            for color in np.unique(source):
                mapped=set(np.unique(target[source==color]).tolist())
                if len(mapped)!=1: return []
                destination=int(next(iter(mapped))); color=int(color)
                if color in mapping and mapping[color]!=destination: return []
                mapping[color]=destination
        # Identity belongs to GlobalTransformSolver; Recolor v0 requires a genuine recolouring rule.
        if not mapping or all(source==target for source,target in mapping.items()): return []
        program=ColorMapProgram(tuple(sorted(mapping.items())))
        return [Candidate('recolor',program,1.0,_exact(program,task),1,program.metadata())] if _exact(program,task) else []

@dataclass(frozen=True)
class FixedCropProgram:
    row: int; col: int; height: int; width: int
    def apply(self, grid: np.ndarray) -> np.ndarray:
        grid=np.asarray(grid)
        if self.row+self.height>grid.shape[0] or self.col+self.width>grid.shape[1]: return grid.copy()
        return grid[self.row:self.row+self.height,self.col:self.col+self.width].copy()
    def metadata(self): return {'operation':'fixed_crop','row':self.row,'col':self.col,'shape':(self.height,self.width)}

def _crop_locations(grid: np.ndarray, target: np.ndarray) -> list[tuple[int,int]]:
    h,w=target.shape
    if h>grid.shape[0] or w>grid.shape[1]: return []
    return [(r,c) for r in range(grid.shape[0]-h+1) for c in range(grid.shape[1]-w+1) if np.array_equal(grid[r:r+h,c:c+w],target)]

class FixedCropSolver(_SolverBase):
    """Accept exactly one shared crop location and shape across every train example."""
    def generate_candidates(self, task: ARCTask) -> list[Candidate]:
        position: tuple[int,int] | None=None; shape: tuple[int,int] | None=None
        for example in task.train:
            target=example.output.values; locations=_crop_locations(example.input.values,target)
            if len(locations)!=1: return []
            if shape is None: shape=target.shape
            if shape != target.shape or (position is not None and position!=locations[0]): return []
            position=locations[0]
        if position is None or shape is None: return []
        program=FixedCropProgram(*position,*shape)
        return [Candidate('fixed_crop',program,1.0,_exact(program,task),1,program.metadata())]

@dataclass(frozen=True)
class ForegroundBBoxProgram:
    def apply(self, grid: np.ndarray) -> np.ndarray:
        grid=np.asarray(grid); bg=_background(grid); points=np.argwhere(grid!=bg)
        if not len(points): return grid.copy()
        return grid[points[:,0].min():points[:,0].max()+1,points[:,1].min():points[:,1].max()+1].copy()
    def metadata(self): return {'operation':'foreground_bbox_crop','background':'most_frequent'}

class ForegroundBBoxCropSolver(_SolverBase):
    def generate_candidates(self, task: ARCTask) -> list[Candidate]:
        program=ForegroundBBoxProgram()
        return [Candidate('foreground_bbox_crop',program,1.0,True,1,program.metadata())] if _exact(program,task) else []

@dataclass(frozen=True)
class TranslationProgram:
    dr: int; dc: int
    def apply(self, grid: np.ndarray) -> np.ndarray:
        grid=np.asarray(grid); bg=_background(grid); points=np.argwhere(grid!=bg)
        if not len(points): return grid.copy()
        shifted=points+np.array([self.dr,self.dc])
        if np.any(shifted[:,0]<0) or np.any(shifted[:,0]>=grid.shape[0]) or np.any(shifted[:,1]<0) or np.any(shifted[:,1]>=grid.shape[1]): return grid.copy()
        out=np.full_like(grid,bg); out[shifted[:,0],shifted[:,1]]=grid[points[:,0],points[:,1]]
        return out
    def metadata(self): return {'operation':'translate','dr':self.dr,'dc':self.dc,'background':'most_frequent'}

def _translation_shift(source: np.ndarray, target: np.ndarray) -> tuple[int,int] | None:
    if source.shape != target.shape: return None
    bg=_background(source)
    # Translation v0 requires a stable background and exact colour-preserving foreground movement.
    if _background(target)!=bg: return None
    sp=np.argwhere(source!=bg); tp=np.argwhere(target!=bg)
    if len(sp)==0 or len(sp)!=len(tp): return None
    shift=tuple((tp.min(axis=0)-sp.min(axis=0)).tolist())
    candidate=TranslationProgram(*map(int,shift))
    return tuple(map(int,shift)) if np.array_equal(candidate.apply(source),target) else None

class TranslationSolver(_SolverBase):
    def generate_candidates(self, task: ARCTask) -> list[Candidate]:
        shifts=[_translation_shift(example.input.values,example.output.values) for example in task.train]
        if not shifts or any(shift is None for shift in shifts) or len(set(shifts))!=1: return []
        shift=shifts[0]
        if shift==(0,0): return []
        program=TranslationProgram(*shift)
        return [Candidate('translation',program,1.0,_exact(program,task),1,program.metadata())] if _exact(program,task) else []
