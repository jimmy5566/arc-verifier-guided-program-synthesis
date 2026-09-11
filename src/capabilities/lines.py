import numpy as np
def full_horizontal(grid):return [(r,int(row[0])) for r,row in enumerate(np.asarray(grid)) if len(set(row))==1]
def full_vertical(grid):
 g=np.asarray(grid);return [(c,int(g[0,c])) for c in range(g.shape[1]) if len(set(g[:,c]))==1]
def split_separator(grid,axis,index):
 g=np.asarray(grid);return (g[:index].copy(),g[index+1:].copy()) if axis=='row' else (g[:,:index].copy(),g[:,index+1:].copy())
def partial_horizontal(grid,min_length=2):
 g=np.asarray(grid);return [(r,c0,c1,int(g[r,c0])) for r in range(g.shape[0]) for c0 in range(g.shape[1]) for c1 in range(c0+min_length-1,g.shape[1]) if len(set(g[r,c0:c1+1]))==1]
def partial_vertical(grid,min_length=2):
 g=np.asarray(grid);return [(c,r0,r1,int(g[r0,c])) for c in range(g.shape[1]) for r0 in range(g.shape[0]) for r1 in range(r0+min_length-1,g.shape[0]) if len(set(g[r0:r1+1,c]))==1]
def connect_aligned_points(grid,first,second,color=None):
 g=np.asarray(grid).copy();(r0,c0),(r1,c1)=first,second
 if not (0<=r0<g.shape[0] and 0<=r1<g.shape[0] and 0<=c0<g.shape[1] and 0<=c1<g.shape[1]):return None
 if r0!=r1 and c0!=c1:return None
 value=int(g[r0,c0]) if color is None else color
 if r0==r1:g[r0,min(c0,c1):max(c0,c1)+1]=value
 else:g[min(r0,r1):max(r0,r1)+1,c0]=value
 return g
