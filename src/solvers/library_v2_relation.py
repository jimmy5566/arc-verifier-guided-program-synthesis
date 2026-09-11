"""Bounded V2 relational/alignment/two-object solvers using structured Programs only."""
from __future__ import annotations
from dataclasses import replace
import numpy as np
from arc.task import ARCTask
from primitives.program import Program, Step, ProgramExecutor, select_relational
from representations.objects import extract_objects
from representations.object_ops import crop_object
from representations.selectors import SELECTORS
from .base import Candidate

EXECUTOR=ProgramExecutor()
def _objects(grid): return extract_objects(grid,4,background=int(np.bincount(np.asarray(grid).ravel()).argmax()))
def _selector_step(selector,role): return Step('SEL_'+selector.name.upper()+'_V1',{'selector':selector.name,'role':role})
def _consistent(task,program):
    return all((prediction:=EXECUTOR.execute(program,ex.input.values)) is not None and np.array_equal(prediction,ex.output.values) for ex in task.train)
def _rel_configs(task):
    configs=[]
    for kind in ('touching_unique','non_touching_unique','same_shape_unique','different_shape_unique','row_aligned_unique','column_aligned_unique'):
        if all(select_relational(ex.input.values,kind) is not None for ex in task.train): configs.append((kind,None))
    for kind in ('nearest','farthest'):
        for ref in SELECTORS:
            if all(select_relational(ex.input.values,kind,ref.name) is not None for ex in task.train): configs.append((kind,ref.name))
    return configs
def _rel_step(kind,reference=None,role='selected'): return Step('SEL_RELATIONAL_V1',{'kind':kind,'reference_selector':reference,'role':role})
def _translation_params(task,kind,reference):
    shared=None
    for ex in task.train:
        selected=select_relational(ex.input.values,kind,reference)
        if selected is None:return set()
        source=crop_object(selected,ex.input.values); possibilities=set()
        for target in _objects(ex.output.values):
            if np.array_equal(source,crop_object(target,ex.output.values)): possibilities.add((target.top-selected.top,target.left-selected.left))
        shared=possibilities if shared is None else shared&possibilities
    return (shared or set())-{(0,0)}
class _V2Base:
    candidates:list[Candidate]; generated_candidates:int; rejected_candidates:int
    def fit(self,task):
        programs=self._programs(task); self.generated_programs=programs; self.generated_candidates=len(programs); consistent=[]
        for program in programs:
            if _consistent(task,program):
                program=replace(program,train_consistent=True); metadata={'program':program.to_dict(),'primitive_ids':[step.primitive_id for step in program.steps]}
                consistent.append(Candidate(' -> '.join(metadata['primitive_ids']),program,1.0,True,program.complexity_cost,metadata))
        self.rejected_candidates=len(programs)-len(consistent); self.candidates=sorted(consistent,key=lambda c:(c.complexity,c.name)); return self
    def predict(self,grid,top_k=1):
        results=[]
        for candidate in self.candidates:
            output=EXECUTOR.execute(candidate.program,grid)
            if output is not None: results.append(output)
            if len(results)>=top_k:break
        return results
class RelationalSelectionSolver(_V2Base):
    def _programs(self,task):
        programs=[]
        for kind,reference in _rel_configs(task):
            select=_rel_step(kind,reference)
            programs += [Program((select,Step('OBJ_CROP_V1',{'transform':transform})),'RelationalSelectionSolver',complexity_cost=2) for transform in ('identity','rotate90','rotate180','rotate270','flip_horizontal','flip_vertical')]
            programs += [Program((select,Step('OBJ_RECOLOR_V1',{'color':color})),'RelationalSelectionSolver',complexity_cost=2) for color in range(10)]
            programs.append(Program((select,Step('OBJ_REMOVE_V1',{})),'RelationalSelectionSolver',complexity_cost=2))
            programs += [Program((select,Step('OBJ_MOVE_V1',{'dr':dr,'dc':dc,'transform':'identity'})),'RelationalSelectionSolver',complexity_cost=3) for dr,dc in _translation_params(task,kind,reference)]
        return programs
class ObjectAlignmentSolver(_V2Base):
    def _programs(self,task):
        programs=[]
        for selected in SELECTORS:
            for reference in SELECTORS:
                if selected is reference:continue
                if not all(len(selected.select(_objects(ex.input.values)))==1 and len(reference.select(_objects(ex.input.values)))==1 for ex in task.train):continue
                programs += [Program((_selector_step(selected,'selected'),_selector_step(reference,'reference'),Step('ALIGN_EDGE_V1',{'edge':edge,'collision':'overwrite'})),'ObjectAlignmentSolver',complexity_cost=3) for edge in ('top','bottom','left','right','center_row','center_column')]
        return programs
class TwoObjectCompositionSolver(_V2Base):
    def _programs(self,task):
        programs=[]
        for a in SELECTORS:
            for b in SELECTORS:
                if a is b:continue
                if not all(len(a.select(_objects(ex.input.values)))==1 and len(b.select(_objects(ex.input.values)))==1 and a.select(_objects(ex.input.values))[0]!=b.select(_objects(ex.input.values))[0] for ex in task.train):continue
                for operation,primitive in (('union','COMP_MASK_UNION_V1'),('intersection','COMP_MASK_INTERSECTION_V1'),('difference_a_b','COMP_MASK_DIFF_A_B_V1'),('difference_b_a','COMP_MASK_DIFF_B_A_V1')):
                    programs.append(Program((_selector_step(a,'a'),_selector_step(b,'b'),Step(primitive,{'operation':operation,'color_policy':'same_color_or_invalid'})),'TwoObjectCompositionSolver',complexity_cost=3))
        return programs
