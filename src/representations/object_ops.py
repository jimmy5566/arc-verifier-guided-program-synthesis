"""Explicit, boundary-safe rendering primitives for a single ARC object."""
from __future__ import annotations
import numpy as np
from .objects import ARCObject

def _background(grid: np.ndarray) -> int: return int(np.bincount(np.asarray(grid).ravel()).argmax())
def _transform(array: np.ndarray, name: str) -> np.ndarray:
    return {'identity':lambda:array.copy(),'rotate90':lambda:np.rot90(array),'rotate180':lambda:np.rot90(array,2),'rotate270':lambda:np.rot90(array,3),'flip_horizontal':lambda:np.fliplr(array),'flip_vertical':lambda:np.flipud(array)}[name]()
def object_patch(obj: ARCObject, grid: np.ndarray, transform: str='identity', color: int | None=None) -> np.ndarray:
    patch=np.full((obj.height,obj.width),_background(grid),dtype=np.asarray(grid).dtype); patch[obj.shape_mask]=obj.color if color is None else color
    return _transform(patch,transform)
def crop_object(obj: ARCObject, grid: np.ndarray, transform: str='identity', color: int | None=None) -> np.ndarray:
    return object_patch(obj,grid,transform,color)
def remove_object(grid: np.ndarray, obj: ARCObject) -> np.ndarray:
    out=np.asarray(grid).copy(); out[tuple(np.asarray(obj.coordinates).T)]=_background(grid); return out
def recolor_object(grid: np.ndarray, obj: ARCObject, color: int) -> np.ndarray:
    out=np.asarray(grid).copy(); out[tuple(np.asarray(obj.coordinates).T)]=color; return out
def render_object(grid: np.ndarray, obj: ARCObject, dr: int, dc: int, *, copy: bool=False, transform: str='identity', color: int | None=None) -> np.ndarray | None:
    """Move/copy an object by an explicit offset. Invalid canvas placement returns None, never clips."""
    grid=np.asarray(grid); patch=object_patch(obj,grid,transform,color); top,left=obj.top+dr,obj.left+dc
    if top<0 or left<0 or top+patch.shape[0]>grid.shape[0] or left+patch.shape[1]>grid.shape[1]: return None
    out=grid.copy()
    if not copy: out=remove_object(out,obj)
    mask=patch!=_background(grid); target=out[top:top+patch.shape[0],left:left+patch.shape[1]]; target[mask]=patch[mask]
    return out
