"""Small deterministic transform DSL. Operations return new arrays and self-describe metadata."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Protocol
import numpy as np

class Transform(Protocol):
    def apply(self, grid: np.ndarray) -> np.ndarray: ...
    def metadata(self) -> dict: ...

@dataclass(frozen=True)
class Identity:
    def apply(self, grid): return np.asarray(grid).copy()
    def metadata(self): return {"operation": "identity"}

@dataclass(frozen=True)
class Rotate:
    degrees: int
    def __post_init__(self):
        if self.degrees not in (90,180,270): raise ValueError("Rotation must be 90, 180, or 270")
    def apply(self, grid): return np.rot90(grid, self.degrees // 90).copy()
    def metadata(self): return {"operation":"rotate", "degrees":self.degrees}

@dataclass(frozen=True)
class FlipHorizontal:
    def apply(self, grid): return np.fliplr(grid).copy()
    def metadata(self): return {"operation":"flip_horizontal"}
@dataclass(frozen=True)
class FlipVertical:
    def apply(self, grid): return np.flipud(grid).copy()
    def metadata(self): return {"operation":"flip_vertical"}
@dataclass(frozen=True)
class Transpose:
    def apply(self, grid): return np.asarray(grid).T.copy()
    def metadata(self): return {"operation":"transpose"}
@dataclass(frozen=True)
class AntiTranspose:
    """Reflection in the anti-diagonal; supports rectangular grids."""
    def apply(self, grid): return np.asarray(grid)[::-1, ::-1].T.copy()
    def metadata(self): return {"operation":"anti_transpose"}

@dataclass(frozen=True)
class CropNonBackground:
    background: int | None = None
    def apply(self, grid):
        grid=np.asarray(grid); bg=int(np.bincount(grid.ravel()).argmax()) if self.background is None else self.background; coords=np.argwhere(grid != bg)
        return grid.copy() if not len(coords) else grid[coords[:,0].min():coords[:,0].max()+1,coords[:,1].min():coords[:,1].max()+1].copy()
    def metadata(self): return {"operation":"crop_non_background", "background":self.background}

@dataclass(frozen=True)
class ExtractLargestObject:
    """Crop the largest non-background connected component while preserving its source colours."""
    background: int | None = None
    def apply(self, grid):
        from representations.objects import extract_objects
        grid=np.asarray(grid); bg=int(np.bincount(grid.ravel()).argmax()) if self.background is None else self.background; objects=extract_objects(grid,background=bg)
        if not objects: return grid.copy()
        obj=max(objects,key=lambda x:x.area); r0,c0,r1,c1=obj.bbox; out=np.full((obj.height,obj.width),bg,dtype=grid.dtype)
        for r,c in obj.coordinates: out[r-r0,c-c0]=grid[r,c]
        return out
    def metadata(self): return {"operation":"extract_largest_object","background":self.background}

@dataclass(frozen=True)
class RemoveColor:
    color: int; replacement: int = 0
    def apply(self, grid):
        out=np.asarray(grid).copy(); out[out==self.color]=self.replacement; return out
    def metadata(self): return {"operation":"remove_object","color":self.color,"replacement":self.replacement}

@dataclass(frozen=True)
class CopyColorObject:
    """Copy all pixels of one colour to an explicit offset; target coordinates outside the grid are ignored."""
    color: int; row_offset: int; col_offset: int
    def apply(self, grid):
        grid=np.asarray(grid); out=grid.copy(); rows,cols=np.where(grid==self.color); dr=rows+self.row_offset; dc=cols+self.col_offset; valid=(dr>=0)&(dr<grid.shape[0])&(dc>=0)&(dc<grid.shape[1]); out[dr[valid],dc[valid]]=self.color; return out
    def metadata(self): return {"operation":"copy_object","color":self.color,"row_offset":self.row_offset,"col_offset":self.col_offset}

@dataclass(frozen=True)
class Pad:
    amount: int; color: int = 0
    def apply(self, grid): return np.pad(np.asarray(grid), self.amount, constant_values=self.color)
    def metadata(self): return {"operation":"pad", "amount":self.amount,"color":self.color}
@dataclass(frozen=True)
class Recolor:
    source: int; target: int
    def apply(self, grid):
        out=np.asarray(grid).copy(); out[out==self.source]=self.target; return out
    def metadata(self): return {"operation":"recolor","source":self.source,"target":self.target}
@dataclass(frozen=True)
class Translate:
    row_offset: int; col_offset: int; fill: int = 0
    def apply(self, grid):
        grid=np.asarray(grid); out=np.full_like(grid,self.fill); rr,cc=np.indices(grid.shape); dst_r=rr+self.row_offset; dst_c=cc+self.col_offset; valid=(dst_r>=0)&(dst_r<grid.shape[0])&(dst_c>=0)&(dst_c<grid.shape[1]); out[dst_r[valid],dst_c[valid]]=grid[rr[valid],cc[valid]]; return out
    def metadata(self): return {"operation":"translate","row_offset":self.row_offset,"col_offset":self.col_offset}
@dataclass(frozen=True)
class Resize:
    shape: tuple[int,int]
    def apply(self, grid):
        grid=np.asarray(grid); rows=np.floor(np.arange(self.shape[0])*grid.shape[0]/self.shape[0]).astype(int); cols=np.floor(np.arange(self.shape[1])*grid.shape[1]/self.shape[1]).astype(int); return grid[np.ix_(rows,cols)].copy()
    def metadata(self): return {"operation":"resize","shape":self.shape}
@dataclass(frozen=True)
class Tile:
    repeats: tuple[int,int]
    def apply(self, grid): return np.tile(grid,self.repeats)
    def metadata(self): return {"operation":"tile","repeats":self.repeats}
@dataclass(frozen=True)
class Repeat(Tile):
    def metadata(self): return {"operation":"repeat","repeats":self.repeats}
@dataclass(frozen=True)
class FillRegion:
    coordinates: tuple[tuple[int,int],...]; color: int
    def apply(self, grid):
        out=np.asarray(grid).copy()
        for r,c in self.coordinates:
            if 0<=r<out.shape[0] and 0<=c<out.shape[1]: out[r,c]=self.color
        return out
    def metadata(self): return {"operation":"fill_region","pixels":len(self.coordinates),"color":self.color}
@dataclass(frozen=True)
class FillBBox:
    color: int; background: int = 0
    def apply(self, grid):
        out=np.asarray(grid).copy(); pts=np.argwhere(out != self.background)
        if len(pts): out[pts[:,0].min():pts[:,0].max()+1,pts[:,1].min():pts[:,1].max()+1]=self.color
        return out
    def metadata(self): return {"operation":"fill_bbox","color":self.color}
@dataclass(frozen=True)
class DrawLine:
    start: tuple[int,int]; end: tuple[int,int]; color: int
    def apply(self, grid):
        out=np.asarray(grid).copy(); r0,c0=self.start; r1,c1=self.end; steps=max(abs(r1-r0),abs(c1-c0))+1
        for r,c in zip(np.rint(np.linspace(r0,r1,steps)).astype(int),np.rint(np.linspace(c0,c1,steps)).astype(int)):
            if 0<=r<out.shape[0] and 0<=c<out.shape[1]: out[r,c]=self.color
        return out
    def metadata(self): return {"operation":"draw_line","start":self.start,"end":self.end,"color":self.color}
@dataclass(frozen=True)
class ConnectPoints:
    points: tuple[tuple[int,int],...]; color: int
    def apply(self, grid):
        out=np.asarray(grid).copy()
        for first,second in zip(self.points,self.points[1:]): out=DrawLine(first,second,self.color).apply(out)
        return out
    def metadata(self): return {"operation":"connect_points","points":self.points,"color":self.color}
@dataclass(frozen=True)
class ReplacePattern:
    source: np.ndarray; target: np.ndarray
    def apply(self, grid):
        grid=np.asarray(grid); src=np.asarray(self.source); tgt=np.asarray(self.target)
        if src.shape != tgt.shape: raise ValueError("source/target pattern shapes must match")
        out=grid.copy(); h,w=src.shape
        for r in range(grid.shape[0]-h+1):
            for c in range(grid.shape[1]-w+1):
                if np.array_equal(grid[r:r+h,c:c+w],src): out[r:r+h,c:c+w]=tgt
        return out
    def metadata(self): return {"operation":"replace_pattern","shape":self.source.shape}

@dataclass(frozen=True)
class TransformPipeline:
    transforms: tuple[Transform, ...] = field(default_factory=tuple)
    def apply(self, grid):
        out=np.asarray(grid).copy()
        for transform in self.transforms: out=transform.apply(out)
        return out
    def metadata(self): return {"operations":[transform.metadata() for transform in self.transforms]}
