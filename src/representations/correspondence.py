"""General object-pair comparison; it never uses task identifiers or test labels."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .objects import ARCObject, extract_objects

def _variants(mask: np.ndarray) -> dict[str,np.ndarray]:
    return {"identity":mask,"rotate90":np.rot90(mask),"rotate180":np.rot90(mask,2),"rotate270":np.rot90(mask,3),"flip_horizontal":np.fliplr(mask),"flip_vertical":np.flipud(mask)}

@dataclass(frozen=True)
class ObjectMatch:
    input_index: int; output_index: int; same_color: bool; same_size: bool; same_shape: bool
    transform: str | None; translation: tuple[int,int]; resized: bool; similarity: float

def compare_objects(source: ARCObject, target: ARCObject, input_index: int=0, output_index: int=0) -> ObjectMatch:
    transform=next((name for name,mask in _variants(source.shape_mask).items() if np.array_equal(mask,target.shape_mask)),None)
    sr,sc=source.bbox[:2]; tr,tc=target.bbox[:2]
    same_size=source.area==target.area; same_shape=transform is not None
    # Shape and area are deliberately stronger evidence than colour, allowing recoloured correspondence.
    similarity=(0.45*same_shape + 0.25*same_size + 0.15*(source.color==target.color) + 0.15*(source.height==target.height and source.width==target.width))
    return ObjectMatch(input_index,output_index,source.color==target.color,same_size,same_shape,transform,(tr-sr,tc-sc),source.shape_mask.size != target.shape_mask.size,float(similarity))

def correspondences(input_grid: np.ndarray, output_grid: np.ndarray, connectivity: int=4) -> list[ObjectMatch]:
    """Return scored all-pairs evidence, including disappeared/new objects as unmatched externally."""
    in_bg=int(np.bincount(input_grid.ravel()).argmax()); out_bg=int(np.bincount(output_grid.ravel()).argmax())
    sources=extract_objects(input_grid,connectivity,background=in_bg); targets=extract_objects(output_grid,connectivity,background=out_bg)
    return sorted((compare_objects(a,b,i,j) for i,a in enumerate(sources) for j,b in enumerate(targets)),key=lambda x:x.similarity,reverse=True)
