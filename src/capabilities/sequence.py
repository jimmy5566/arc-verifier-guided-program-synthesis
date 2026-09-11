import numpy as np
def grid_row(grid,row):return tuple(map(int,np.asarray(grid)[row]))
def grid_column(grid,col):return tuple(map(int,np.asarray(grid)[:,col]))
def sequence_row(sequence):return np.asarray([sequence],dtype=int)
def sequence_column(sequence):return np.asarray(sequence,dtype=int).reshape(-1,1)
def reverse(sequence):return tuple(sequence[::-1])
def repeat(sequence,n):return tuple(sequence)*n if n>=0 else None
def run_length_encode(sequence):
 seq=tuple(sequence)
 if not seq:return tuple()
 out=[];current=seq[0];count=1
 for value in seq[1:]:
  if value==current:count+=1
  else:out.append((int(current),count));current=value;count=1
 out.append((int(current),count));return tuple(out)
