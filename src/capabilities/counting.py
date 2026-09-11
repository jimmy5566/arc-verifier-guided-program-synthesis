import numpy as np
from representations.objects import extract_objects
from capabilities.regions import regions
def count_objects(grid):return len(extract_objects(grid,background=int(np.bincount(np.asarray(grid).ravel()).argmax())))
def count_color_cells(grid,color):return int((np.asarray(grid)==color).sum())
def count_runs(seq):return 0 if not len(seq) else 1+sum(a!=b for a,b in zip(seq,seq[1:]))
def count_regions(grid,connectivity=4,color=None):return len(regions(grid,connectivity,color))
def selected_region_area(region):return len(region)
def value_frequency(grid):return tuple((int(color),int(count)) for color,count in zip(*np.unique(np.asarray(grid),return_counts=True)))
def repeated_motif_count(sequence,motif):
 seq=tuple(sequence);motif=tuple(motif)
 if not motif:return None
 return sum(seq[index:index+len(motif)]==motif for index in range(len(seq)-len(motif)+1))
