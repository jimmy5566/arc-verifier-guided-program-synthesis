"""Explicitly bounded iterative capabilities; no unconstrained loops."""
from __future__ import annotations
import numpy as np
def repeat_translation(grid, dr:int, dc:int, repeats:int, background:int|None=None):
    if repeats < 0:return None
    out=np.asarray(grid).copy(); bg=int(np.bincount(out.ravel()).argmax()) if background is None else background
    for _ in range(repeats):
        points=np.argwhere(out!=bg); shifted=points+np.array([dr,dc])
        if np.any(shifted<0) or np.any(shifted[:,0]>=out.shape[0]) or np.any(shifted[:,1]>=out.shape[1]):return None
        nxt=np.full_like(out,bg);nxt[shifted[:,0],shifted[:,1]]=out[points[:,0],points[:,1]];out=nxt
    return out
def extend_line_until_boundary(grid, row:int, col:int, dr:int, dc:int, color:int):
    g=np.asarray(grid).copy()
    if (dr,dc)==(0,0) or not (0<=row<g.shape[0] and 0<=col<g.shape[1]):return None
    while 0<=row<g.shape[0] and 0<=col<g.shape[1]:g[row,col]=color;row+=dr;col+=dc
    return g
