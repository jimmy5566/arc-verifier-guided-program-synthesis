from __future__ import annotations
import numpy as np
def smallest_period(seq,unknown=None):
 s=list(seq)
 for p in range(1,len(s)+1):
  ok=True
  for i in range(p,len(s)):
   if unknown is not None and (s[i]==unknown or s[i-p]==unknown):continue
   if s[i]!=s[i-p]:ok=False;break
  if ok:return p
 return None
def tile_period(grid,unknown=None):
 g=np.asarray(grid)
 for r in range(1,g.shape[0]+1):
  for c in range(1,g.shape[1]+1):
   ok=True
   for i,j in np.ndindex(g.shape):
    a,b=g[i,j],g[i%r,j%c]
    if unknown is not None and (a==unknown or b==unknown):continue
    if a!=b:ok=False;break
   if ok:return r,c,g[:r,:c].copy()
 return None
def complete_rows(grid,period,extent=None):
 g=np.asarray(grid); w=extent or g.shape[1]; out=np.empty((g.shape[0],w),dtype=g.dtype)
 for r in range(g.shape[0]):out[r]=[g[r,i%period] for i in range(w)]
 return out
def complete_columns(grid,period,extent=None):
 g=np.asarray(grid); h=extent or g.shape[0]; out=np.empty((h,g.shape[1]),dtype=g.dtype)
 for c in range(g.shape[1]):out[:,c]=[g[i%period,c] for i in range(h)]
 return out
def repair_rows(grid,period):
 g=np.asarray(grid).copy(); changes=[]
 for r in range(g.shape[0]):
  for c in range(period,g.shape[1]):
   if g[r,c]!=g[r,c%period]:changes.append((r,c,int(g[r,c]),int(g[r,c%period])));g[r,c]=g[r,c%period]
 return g,changes
def symmetry_complete(grid,axis,background=None):
 g=np.asarray(grid).copy(); bg=int(np.bincount(g.ravel()).argmax()) if background is None else background
 if axis=='horizontal':pairs=[((r,c),(g.shape[0]-1-r,c)) for r,c in np.ndindex(g.shape)]
 elif axis=='vertical':pairs=[((r,c),(r,g.shape[1]-1-c)) for r,c in np.ndindex(g.shape)]
 elif axis=='main':
  if g.shape[0]!=g.shape[1]:return None
  pairs=[((r,c),(c,r)) for r,c in np.ndindex(g.shape)]
 elif axis=='anti':
  if g.shape[0]!=g.shape[1]:return None
  n=g.shape[0];pairs=[((r,c),(n-1-c,n-1-r)) for r,c in np.ndindex(g.shape)]
 else:return None
 for (r,c),(x,y) in pairs:
  if g[r,c]==bg and g[x,y]!=bg:g[r,c]=g[x,y]
  elif g[x,y]==bg and g[r,c]!=bg:g[x,y]=g[r,c]
  elif g[x,y]!=g[r,c]:return None
 return g
