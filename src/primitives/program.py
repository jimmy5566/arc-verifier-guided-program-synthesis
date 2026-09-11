"""Structured primitive-ID programs and a deliberately small registry-driven executor."""
from __future__ import annotations
from dataclasses import asdict, dataclass, field
from typing import Any
import numpy as np
from patterns.core import complete_rows,complete_columns,repair_rows,symmetry_complete
from representations.objects import ARCObject, extract_objects
from representations.selectors import ObjectSelector
from representations import relations_v1 as rel
from representations.object_ops import crop_object, recolor_object, remove_object, render_object
from .registry import REGISTRY

@dataclass(frozen=True)
class Step:
    primitive_id:str; params:dict[str,Any]=field(default_factory=dict)
@dataclass(frozen=True)
class Program:
    steps:tuple[Step,...]; provenance_solver:str; train_consistent:bool=False; complexity_cost:int=0
    @property
    def depth(self): return len(self.steps)
    def to_dict(self): return {'steps':[asdict(step) for step in self.steps],'provenance_solver':self.provenance_solver,'train_consistent':self.train_consistent,'complexity_cost':self.complexity_cost}
    @classmethod
    def from_dict(cls,data): return cls(tuple(Step(**item) for item in data['steps']),data['provenance_solver'],data.get('train_consistent',False),data.get('complexity_cost',0))
def _objects(grid): return extract_objects(grid,4,background=int(np.bincount(np.asarray(grid).ravel()).argmax()))
def _one(selector:str, grid):
    candidates=ObjectSelector(selector).select(_objects(grid)); return candidates[0] if len(candidates)==1 else None
def select_relational(grid:np.ndarray,kind:str,reference_selector:str|None=None)->ARCObject|None:
    objects=_objects(grid)
    if kind=='touching_unique': candidates=[a for a in objects if any(rel.touching_4(a,b) for b in objects if b is not a)]
    elif kind=='non_touching_unique': candidates=[a for a in objects if not any(rel.touching_4(a,b) for b in objects if b is not a)]
    elif kind=='same_shape_unique': candidates=[a for a in objects if any(rel.same_shape(a,b) for b in objects if b is not a)]
    elif kind=='different_shape_unique': candidates=[a for a in objects if not any(rel.same_shape(a,b) for b in objects if b is not a)]
    elif kind in ('row_aligned_unique','column_aligned_unique'):
        checker=rel.row_aligned_center if kind.startswith('row') else rel.column_aligned_center; candidates=[a for a in objects if any(checker(a,b) for b in objects if b is not a)]
    elif kind in ('nearest','farthest') and reference_selector:
        reference=_one(reference_selector,grid)
        return (rel.unique_nearest if kind=='nearest' else rel.unique_farthest)(reference,objects) if reference else None
    else: return None
    return candidates[0] if len(candidates)==1 else None
def align_object(grid:np.ndarray,selected:ARCObject,reference:ARCObject,edge:str,collision:str='overwrite')->np.ndarray|None:
    if collision!='overwrite': return None
    if edge=='top': dr=reference.top-selected.top; dc=0
    elif edge=='bottom': dr=reference.bottom-selected.bottom; dc=0
    elif edge=='left': dr=0; dc=reference.left-selected.left
    elif edge=='right': dr=0; dc=reference.right-selected.right
    elif edge=='center_row': dr=(reference.top+reference.bottom-selected.top-selected.bottom)//2; dc=0
    elif edge=='center_column': dr=0; dc=(reference.left+reference.right-selected.left-selected.right)//2
    else: return None
    return render_object(grid,selected,dr,dc,copy=False)
def compose_masks(a:ARCObject,b:ARCObject,grid:np.ndarray,operation:str,color_policy:str='same_color_or_invalid')->np.ndarray|None:
    if a.shape_mask.shape!=b.shape_mask.shape: return None
    if operation in ('union','intersection') and color_policy=='same_color_or_invalid' and a.color!=b.color:return None
    ma,mb=a.shape_mask,b.shape_mask; bg=int(np.bincount(np.asarray(grid).ravel()).argmax()); out=np.full(ma.shape,bg,dtype=np.asarray(grid).dtype)
    if operation=='union': mask=ma|mb; color=a.color
    elif operation=='intersection': mask=ma&mb; color=a.color
    elif operation=='difference_a_b': mask=ma&~mb; color=a.color
    elif operation=='difference_b_a': mask=mb&~ma; color=b.color
    else:return None
    out[mask]=color; return out
def copy_relative(grid:np.ndarray,selected:ARCObject,reference:ARCObject,axis:str,edge:str,gap:int)->np.ndarray|None:
    if axis=='row':
        target=reference.top-gap-selected.height if edge=='top' else reference.bottom+1+gap; return render_object(grid,selected,target-selected.top,0,copy=True)
    if axis=='column':
        target=reference.left-gap-selected.width if edge=='left' else reference.right+1+gap; return render_object(grid,selected,0,target-selected.left,copy=True)
    return None
class ProgramExecutor:
    def execute(self,program:Program,grid:np.ndarray)->np.ndarray|None:
        context:dict[str,Any]={}; result=None
        for step in program.steps:
            if step.primitive_id not in REGISTRY: return None
            p=step.params
            if step.primitive_id=='OBJ_RECOLOR_V1' and (not isinstance(p.get('color'),int) or not 0<=p['color']<=9): return None
            if step.primitive_id in ('OBJ_MOVE_V1','OBJ_COPY_V1') and not isinstance(p.get('dr'),int): return None
            if step.primitive_id in ('OBJ_MOVE_V1','OBJ_COPY_V1') and not isinstance(p.get('dc'),int): return None
            if step.primitive_id=='ALIGN_EDGE_V1' and p.get('edge') not in ('top','bottom','left','right','center_row','center_column'): return None
            if step.primitive_id.startswith('COMP_MASK_') and p.get('operation') not in ('union','intersection','difference_a_b','difference_b_a'): return None
            if step.primitive_id=='SEL_RELATIONAL_V1' and p.get('kind') not in ('touching_unique','non_touching_unique','same_shape_unique','different_shape_unique','row_aligned_unique','column_aligned_unique','nearest','farthest'): return None
            if step.primitive_id.startswith('SEL_') and step.primitive_id!='SEL_RELATIONAL_V1':
                selector=p.get('selector') or step.primitive_id.removeprefix('SEL_').removesuffix('_V1').lower(); selected=_one(selector,grid)
                if selected is None:return None
                context[p.get('role','selected')]=selected
                continue
            elif step.primitive_id=='SEL_RELATIONAL_V1':
                selected=select_relational(grid,p.get('kind'),p.get('reference_selector'))
                if selected is None:return None
                context[p.get('role','selected')]=selected
                continue
            elif step.primitive_id=='OBJ_CROP_V1': result=crop_object(context['selected'],grid,p.get('transform','identity'),p.get('color'))
            elif step.primitive_id=='OBJ_RECOLOR_V1': result=recolor_object(grid,context['selected'],p['color'])
            elif step.primitive_id=='OBJ_REMOVE_V1': result=remove_object(grid,context['selected'])
            elif step.primitive_id in ('OBJ_MOVE_V1','OBJ_COPY_V1'): result=render_object(grid,context['selected'],p['dr'],p['dc'],copy=step.primitive_id=='OBJ_COPY_V1',transform=p.get('transform','identity'))
            elif step.primitive_id=='ALIGN_EDGE_V1': result=align_object(grid,context['selected'],context['reference'],p['edge'],p.get('collision','overwrite'))
            elif step.primitive_id.startswith('COMP_MASK_'): result=compose_masks(context['a'],context['b'],grid,p['operation'],p.get('color_policy','same_color_or_invalid'))
            elif step.primitive_id=='COPY_RELATIVE_REFERENCE_V1': result=copy_relative(grid,context['selected'],context['reference'],p['axis'],p['edge'],p['gap'])
            elif step.primitive_id=='PAT_COMPLETE_2D_TILE_V1':
                rr,cc=p['repeats']; result=np.tile(grid,(rr,cc))[:p['shape'][0],:p['shape'][1]]
            elif step.primitive_id=='PAT_EXTEND_ROW_V1':
                result=complete_rows(grid,p['period'],p['width'])
            elif step.primitive_id=='PAT_EXTEND_COLUMN_V1':
                result=complete_columns(grid,p['period'],p['height'])
            elif step.primitive_id=='PAT_REPAIR_PERIOD_VIOLATION_V1':
                result,changes=repair_rows(grid,p['period']);
                if not changes or len(changes)>=grid.size-len(changes): return None
            elif step.primitive_id.startswith('PAT_COMPLETE_MIRROR_'):
                axis={'PAT_COMPLETE_MIRROR_HORIZONTAL_V1':'horizontal','PAT_COMPLETE_MIRROR_VERTICAL_V1':'vertical','PAT_COMPLETE_MIRROR_DIAGONAL_MAIN_V1':'main','PAT_COMPLETE_MIRROR_DIAGONAL_ANTI_V1':'anti'}[step.primitive_id]; result=symmetry_complete(grid,axis)
            else:return None
            if result is None:return None
        return result
