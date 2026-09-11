"""Deterministic, non-exclusive capability scheduling."""
from dataclasses import dataclass
import numpy as np
from .regions import enclosed_regions
from .lines import full_horizontal,full_vertical
from .paths import graph_pixels,simple_path
@dataclass(frozen=True)
class Schedule:
 priorities:dict[str,float]; primary:tuple[str,...]; fallback:tuple[str,...]; budgets:dict[str,int]
class CapabilitySchedulerV1:
 families=('object','relation','pattern','region','separator','graph_path','counting','generation','sequence','iteration')
 def schedule(self,task,base_budget=24,top_k=3):
  scores={x:0.05 for x in self.families}
  for ex in task.train:
   g=ex.input.values;o=ex.output.values;scores['region']+=0.5*bool(enclosed_regions(g));scores['separator']+=0.25*(bool(full_horizontal(g)) or bool(full_vertical(g)));scores['graph_path']+=0.25*simple_path(graph_pixels(g));scores['pattern']+=0.2*(g.shape!=o.shape);scores['counting']+=0.15*(len(np.unique(g))!=len(np.unique(o)))
  ranked=tuple(sorted(self.families,key=lambda x:(-scores[x],x)));primary=ranked[:top_k];return Schedule(scores,primary,tuple(x for x in ranked if x not in primary),{x:(base_budget if x in primary else max(1,base_budget//6)) for x in ranked})
