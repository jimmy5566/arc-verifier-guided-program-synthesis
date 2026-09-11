import numpy as np
from collections import deque
def graph_pixels(grid,color=None):
 g=np.asarray(grid);pts=[tuple(x) for x in np.argwhere(g!=(int(np.bincount(g.ravel()).argmax()) if color is None else color))];S=set(pts);return {p:sorted(q for q in ((p[0]+1,p[1]),(p[0]-1,p[1]),(p[0],p[1]+1),(p[0],p[1]-1)) if q in S) for p in pts}
def endpoints(G):return sorted(p for p,n in G.items() if len(n)==1)
def branches(G):return sorted(p for p,n in G.items() if len(n)>2)
def simple_path(G):return len(G)==1 or (len(endpoints(G))==2 and not branches(G) and all(len(n) in(1,2) for n in G.values()))
def trace(G,start=None):
 if not simple_path(G):return None
 if len(G)==1:return list(G)
 out=[start or endpoints(G)[0]]
 while len(out)<len(G):
  n=[x for x in G[out[-1]] if x not in out]
  if not n:return None
  out.append(n[0])
 return out
def connected(G):
 if not G:return False
 seen={next(iter(G))};q=deque(seen)
 while q:
  node=q.popleft()
  for neighbour in G[node]:
   if neighbour not in seen:seen.add(neighbour);q.append(neighbour)
 return len(seen)==len(G)
def shortest_path(G,start,end):
 if start not in G or end not in G:return None
 q=deque([start]);parent={start:None}
 while q:
  node=q.popleft()
  if node==end:
   out=[]
   while node is not None:out.append(node);node=parent[node]
   return list(reversed(out))
  for neighbour in sorted(G[node]):
   if neighbour not in parent:parent[neighbour]=node;q.append(neighbour)
 return None
def path_color_sequence(grid,path):
 g=np.asarray(grid)
 if not path or any(not (0<=r<g.shape[0] and 0<=c<g.shape[1]) for r,c in path):return None
 return tuple(int(g[r,c]) for r,c in path)
