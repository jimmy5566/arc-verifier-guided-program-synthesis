from __future__ import annotations
import numpy as np
from .objects import ARCObject

def _touching(a: ARCObject, b: ARCObject) -> bool:
    ap, bp = set(a.coordinates), set(b.coordinates)
    return any((r+dr,c+dc) in bp for r,c in ap for dr,dc in ((1,0),(-1,0),(0,1),(0,-1)))

def object_relations(a: ARCObject, b: ARCObject) -> dict[str, bool | float]:
    ar,ac = a.centroid; br,bc = b.centroid; a0,b0,a1,b1 = a.bbox; c0,d0,c1,d1 = b.bbox
    overlap = not (a1 < c0 or c1 < a0 or b1 < d0 or d1 < b0)
    contains = a0 <= c0 and b0 <= d0 and a1 >= c1 and b1 >= d1
    return {"above": ar < br, "below": ar > br, "left": ac < bc, "right": ac > bc, "touching": _touching(a,b), "overlap": overlap, "containment": contains, "aligned": ar == br or ac == bc, "equal_size": a.area == b.area, "equal_shape": np.array_equal(a.shape_mask,b.shape_mask), "same_color": a.color == b.color, "distance": float(np.hypot(ar-br, ac-bc))}
