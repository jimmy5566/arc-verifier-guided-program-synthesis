"""Deterministic object-selection primitives. They select, never render output grids."""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
from .objects import ARCObject

def _shape_key(obj: ARCObject) -> tuple[tuple[int,int],bytes]: return (obj.shape_mask.shape,obj.shape_mask.tobytes())

@dataclass(frozen=True)
class ObjectSelector:
    name: str
    def select(self, objects: list[ARCObject]) -> list[ARCObject]:
        if not objects: return []
        if self.name=='largest':
            area=max(obj.area for obj in objects); return [obj for obj in objects if obj.area==area]
        if self.name=='smallest':
            area=min(obj.area for obj in objects); return [obj for obj in objects if obj.area==area]
        if self.name=='unique_area':
            counts=Counter(obj.area for obj in objects); return [obj for obj in objects if counts[obj.area]==1]
        if self.name=='unique_shape':
            counts=Counter(_shape_key(obj) for obj in objects); return [obj for obj in objects if counts[_shape_key(obj)]==1]
        if self.name=='unique_color':
            counts=Counter(obj.color for obj in objects); return [obj for obj in objects if counts[obj.color]==1]
        if self.name=='border': return [obj for obj in objects if obj.touches_border]
        if self.name=='non_border': return [obj for obj in objects if not obj.touches_border]
        raise ValueError(f'unknown selector: {self.name}')

SelectLargest=ObjectSelector('largest')
SelectSmallest=ObjectSelector('smallest')
SelectUniqueByArea=ObjectSelector('unique_area')
SelectUniqueByShape=ObjectSelector('unique_shape')
SelectUniqueByColor=ObjectSelector('unique_color')
SelectBorderObject=ObjectSelector('border')
SelectNonBorderObject=ObjectSelector('non_border')
SELECTORS=(SelectLargest,SelectSmallest,SelectUniqueByArea,SelectUniqueByShape,SelectUniqueByColor,SelectBorderObject,SelectNonBorderObject)
