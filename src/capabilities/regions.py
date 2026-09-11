from collections import deque
import numpy as np
def regions(grid,connectivity=4,color=None):
 g=np.asarray(grid);seen=np.zeros(g.shape,bool);out=[];steps=((1,0),(-1,0),(0,1),(0,-1)) if connectivity==4 else tuple((a,b) for a in(-1,0,1) for b in(-1,0,1) if(a,b)!=(0,0))
 for r,c in np.ndindex(g.shape):
  if seen[r,c] or(color is not None and g[r,c]!=color):continue
  v=g[r,c];q=deque([(r,c)]);seen[r,c]=1;pts=[]
  while q:
   x,y=q.popleft();pts.append((x,y))
   for dx,dy in steps:
    a,b=x+dx,y+dy
    if 0<=a<g.shape[0] and 0<=b<g.shape[1] and not seen[a,b] and g[a,b]==v:seen[a,b]=1;q.append((a,b))
  out.append((int(v),tuple(pts)))
 return out
def enclosed_regions(grid,background=None,connectivity=4):
 g=np.asarray(grid);bg=int(np.bincount(g.ravel()).argmax()) if background is None else background;return [p for v,p in regions(g,connectivity,bg) if not any(r in(0,g.shape[0]-1) or c in(0,g.shape[1]-1) for r,c in p)]
def fill_interior(grid,color,background=None):
 out=np.asarray(grid).copy()
 for pts in enclosed_regions(out,background):out[tuple(np.array(pts).T)]=color
 return out
def find_holes(grid,background=None):
 return enclosed_regions(grid,background,4)
def exterior_mask(grid,background=None):
 g=np.asarray(grid);bg=int(np.bincount(g.ravel()).argmax()) if background is None else background;mask=np.zeros(g.shape,bool)
 for _,pts in regions(g,4,bg):
  if any(r in (0,g.shape[0]-1) or c in (0,g.shape[1]-1) for r,c in pts):mask[tuple(np.asarray(pts).T)]=True
 return mask
def interior_mask(grid,background=None):
 g=np.asarray(grid);return (g==(int(np.bincount(g.ravel()).argmax()) if background is None else background)) & ~exterior_mask(g,background)
def region_boundary(points):
 p=set(points);return tuple(sorted((r,c) for r,c in p if any((r+dr,c+dc) not in p for dr,dc in ((1,0),(-1,0),(0,1),(0,-1)))))
def adjacent_regions(grid,background=None):
 regs=regions(grid,4,background);return [(i,j) for i,(_,a) in enumerate(regs) for j,(_,b) in enumerate(regs) if i<j and any(abs(r-x)+abs(c-y)==1 for r,c in a for x,y in b)]
def extract_region_at(grid,row,col,connectivity=4):
 g=np.asarray(grid)
 if not (0<=row<g.shape[0] and 0<=col<g.shape[1]) or connectivity not in (4,8):return None
 color=int(g[row,col])
 for value,points in regions(g,connectivity,color):
  if value==color and (row,col) in points:return points
 return None
