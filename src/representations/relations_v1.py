"""Unambiguous object relation primitives for Primitive Registry v1."""
from __future__ import annotations
import math
from .objects import ARCObject

def same_color(a: ARCObject,b: ARCObject)->bool: return a.color==b.color
def same_area(a: ARCObject,b: ARCObject)->bool: return a.area==b.area
def same_bbox_size(a: ARCObject,b: ARCObject)->bool: return (a.height,a.width)==(b.height,b.width)
def same_shape(a: ARCObject,b: ARCObject)->bool: return same_bbox_size(a,b) and (a.shape_mask==b.shape_mask).all()
def rotated_equivalent(a: ARCObject,b: ARCObject)->bool:
    import numpy as np
    return any(np.array_equal(np.rot90(a.shape_mask,k),b.shape_mask) for k in range(4))
def reflected_equivalent(a: ARCObject,b: ARCObject)->bool:
    import numpy as np
    return any(np.array_equal(np.fliplr(np.rot90(a.shape_mask,k)),b.shape_mask) for k in range(4))
def left_of(a: ARCObject,b: ARCObject)->bool: return a.right < b.left
def right_of(a: ARCObject,b: ARCObject)->bool: return a.left > b.right
def above(a: ARCObject,b: ARCObject)->bool: return a.bottom < b.top
def below(a: ARCObject,b: ARCObject)->bool: return a.top > b.bottom
def row_aligned_center(a: ARCObject,b: ARCObject)->bool: return (a.top+a.bottom)==(b.top+b.bottom)
def column_aligned_center(a: ARCObject,b: ARCObject)->bool: return (a.left+a.right)==(b.left+b.right)
def touching_4(a: ARCObject,b: ARCObject)->bool:
    bp=set(b.pixels); return any((r+dr,c+dc) in bp for r,c in a.pixels for dr,dc in ((1,0),(-1,0),(0,1),(0,-1)))
def overlap(a: ARCObject,b: ARCObject)->bool: return bool(set(a.pixels)&set(b.pixels))
def contains_bbox(a: ARCObject,b: ARCObject)->bool: return a.top<=b.top and a.left<=b.left and a.bottom>=b.bottom and a.right>=b.right
def inside_bbox(a: ARCObject,b: ARCObject)->bool: return contains_bbox(b,a)
def min_manhattan(a: ARCObject,b: ARCObject)->int: return min(abs(r-x)+abs(c-y) for r,c in a.pixels for x,y in b.pixels)
def min_chebyshev(a: ARCObject,b: ARCObject)->int: return min(max(abs(r-x),abs(c-y)) for r,c in a.pixels for x,y in b.pixels)
def bbox_center_euclidean(a: ARCObject,b: ARCObject)->float:
    ar,ac=(a.top+a.bottom)/2,(a.left+a.right)/2; br,bc=(b.top+b.bottom)/2,(b.left+b.right)/2; return math.hypot(ar-br,ac-bc)
def _unique_extreme(reference: ARCObject, objects: list[ARCObject], *, farthest: bool)->ARCObject|None:
    options=[obj for obj in objects if obj is not reference]
    if not options:return None
    distances=[min_manhattan(reference,obj) for obj in options]; target=max(distances) if farthest else min(distances)
    chosen=[obj for obj,distance in zip(options,distances) if distance==target]
    return chosen[0] if len(chosen)==1 else None
def unique_nearest(reference: ARCObject,objects:list[ARCObject])->ARCObject|None:return _unique_extreme(reference,objects,farthest=False)
def unique_farthest(reference: ARCObject,objects:list[ARCObject])->ARCObject|None:return _unique_extreme(reference,objects,farthest=True)
