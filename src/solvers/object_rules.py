"""Small, interpretable object-rule templates for EXP002."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import numpy as np
from representations.objects import ARCObject, extract_objects
from representations.correspondence import correspondences

Selector = Literal['largest','smallest','unique_color','unique_size','border_touching','non_border','singleton']

def background(grid: np.ndarray) -> int: return int(np.bincount(np.asarray(grid).ravel()).argmax())

def select_object(grid: np.ndarray, selector: Selector) -> ARCObject | None:
    objects=extract_objects(grid,4,background=background(grid))
    if not objects: return None
    if selector=='largest':
        best=max(x.area for x in objects); choices=[x for x in objects if x.area==best]
    elif selector=='smallest':
        best=min(x.area for x in objects); choices=[x for x in objects if x.area==best]
    elif selector=='unique_color': choices=[x for x in objects if sum(y.color==x.color for y in objects)==1]
    elif selector=='unique_size': choices=[x for x in objects if sum(y.area==x.area for y in objects)==1]
    elif selector=='border_touching': choices=[x for x in objects if x.touches_border]
    elif selector=='non_border': choices=[x for x in objects if not x.touches_border]
    else: choices=[x for x in objects if x.area==1]
    return choices[0] if len(choices)==1 else None

def _object_patch(obj: ARCObject, grid: np.ndarray, transform: str, color: int | None=None) -> np.ndarray:
    patch=np.full(obj.shape_mask.shape,background(grid),dtype=grid.dtype); patch[obj.shape_mask]=obj.color if color is None else color
    return _transform_patch(patch,transform)

def _transform_patch(patch: np.ndarray, transform: str) -> np.ndarray:
    return {'identity':lambda:patch.copy(),'rotate90':lambda:np.rot90(patch),'rotate180':lambda:np.rot90(patch,2),'rotate270':lambda:np.rot90(patch,3),'flip_horizontal':lambda:np.fliplr(patch),'flip_vertical':lambda:np.flipud(patch)}[transform]()

@dataclass(frozen=True)
class ObjectRule:
    selector: Selector; operation: str; transform: str='identity'; color: int | None=None; offset: tuple[int,int]=(0,0)
    def metadata(self) -> dict:
        return {'family':'object_rule','selector':self.selector,'operation':self.operation,'transform':self.transform,'color':self.color,'offset':self.offset}
    def apply(self, grid: np.ndarray) -> np.ndarray:
        grid=np.asarray(grid); obj=select_object(grid,self.selector)
        if obj is None: return grid.copy()
        bg=background(grid); r0,c0,r1,c1=obj.bbox
        if self.operation=='crop': return _object_patch(obj,grid,self.transform,self.color)
        out=grid.copy()
        if self.operation=='keep_only':
            out[:]=bg
            for r,c in obj.coordinates: out[r,c]=obj.color if self.color is None else self.color
            return out
        if self.operation=='remove':
            for r,c in obj.coordinates: out[r,c]=bg
            return out
        if self.operation=='recolor':
            for r,c in obj.coordinates: out[r,c]=self.color
            return out
        if self.operation=='fill_bbox': out[r0:r1+1,c0:c1+1]=obj.color if self.color is None else self.color; return out
        if self.operation in ('move','copy'):
            if self.operation=='move':
                for r,c in obj.coordinates: out[r,c]=bg
            patch=_object_patch(obj,grid,self.transform,self.color); rr,cc=r0+self.offset[0],c0+self.offset[1]
            for r,c in np.argwhere(patch != bg):
                tr,tc=rr+int(r),cc+int(c)
                if 0<=tr<out.shape[0] and 0<=tc<out.shape[1]: out[tr,tc]=patch[r,c]
            return out
        if self.operation=='transform_inplace':
            out[r0:r1+1,c0:c1+1]=bg; patch=_object_patch(obj,grid,self.transform,self.color)
            # Restrict in-place transformations to identical bounding shape.
            if patch.shape==out[r0:r1+1,c0:c1+1].shape: out[r0:r1+1,c0:c1+1]=patch
            return out
        raise ValueError(f'unknown object operation: {self.operation}')

def candidate_rules(task) -> list[ObjectRule]:
    selectors: tuple[Selector,...]=('largest','smallest','unique_color','unique_size','border_touching','non_border','singleton')
    rules=[]
    for selector in selectors:
        if not all(select_object(ex.input.values,selector) is not None for ex in task.train): continue
        rules += [ObjectRule(selector,'crop',transform) for transform in ('identity','rotate90','rotate180','rotate270','flip_horizontal','flip_vertical')]
        rules += [ObjectRule(selector,operation) for operation in ('keep_only','remove','fill_bbox')]
        rules += [ObjectRule(selector,'recolor',color=color) for color in range(10)]
        rules += [ObjectRule(selector,'fill_bbox',color=color) for color in range(10)]
        rules += [ObjectRule(selector,'transform_inplace',transform) for transform in ('rotate180','flip_horizontal','flip_vertical')]
        # Translation/copy hypotheses come from train-pair correspondence and are intersected across all pairs.
        shared: set[tuple[str,tuple[int,int],int]] | None=None
        for ex in task.train:
            obj=select_object(ex.input.values,selector)
            source_objects=extract_objects(ex.input.values,4,background=background(ex.input.values))
            target_objects=extract_objects(ex.output.values,4,background=background(ex.output.values))
            source_index=source_objects.index(obj)
            matches=correspondences(ex.input.values,ex.output.values)
            candidates={(m.transform or 'identity',m.translation,target_objects[m.output_index].color) for m in matches if m.input_index == source_index and m.same_shape}
            shared=candidates if shared is None else shared & candidates
        for transform,offset,color in shared or set():
            if offset != (0,0): rules += [ObjectRule(selector,'move',transform,color,offset),ObjectRule(selector,'copy',transform,color,offset)]
    return rules
