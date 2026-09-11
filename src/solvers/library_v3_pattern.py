from dataclasses import replace
import numpy as np
from arc.task import ARCTask
from primitives.program import Program,Step,ProgramExecutor
EXEC=ProgramExecutor()
def exact(task,p):return all((o:=EXEC.execute(p,e.input.values)) is not None and np.array_equal(o,e.output.values) for e in task.train)
class Base:
 def fit(self,t):
  self.generated_programs=self._programs(t);self.generated_candidates=len(self.generated_programs);self.candidates=[]
  for p in self.generated_programs:
   if exact(t,p):self.candidates.append(type('C',(),{'program':replace(p,train_consistent=True),'complexity':p.complexity_cost})())
  self.rejected_candidates=self.generated_candidates-len(self.candidates);return self
 def predict(self,g,top_k=1):
  out=[]
  for c in self.candidates:
   x=EXEC.execute(c.program,g)
   if x is not None:out.append(x)
   if len(out)>=top_k:break
  return out
class PeriodicCompletionSolver(Base):
 def _programs(self,t):
  ps=[]
  # Repeated-tile extent is inferred only from integer train shape ratios.
  for e in t.train[:1]:
   ih,iw=e.input.values.shape;oh,ow=e.output.values.shape
   if oh%ih==0 and ow%iw==0:ps.append(Program((Step('PAT_FIND_2D_TILE_PERIOD_V1',{}),Step('PAT_COMPLETE_2D_TILE_V1',{'repeats':(oh//ih,ow//iw),'shape':(oh,ow)})),'PeriodicCompletionSolver',complexity_cost=2))
  return ps
class PatternRepairSolver(Base):
 def _programs(self,t):
  return [Program((Step('PAT_FIND_ROW_PERIOD_V1',{}),Step('PAT_REPAIR_PERIOD_VIOLATION_V1',{'period':p})),'PatternRepairSolver',complexity_cost=2) for p in range(1,min(e.input.values.shape[1] for e in t.train))]
class SymmetryCompletionSolver(Base):
 def _programs(self,t):
  return [Program((Step(pid,{}),),'SymmetryCompletionSolver',complexity_cost=1) for pid in ('PAT_COMPLETE_MIRROR_HORIZONTAL_V1','PAT_COMPLETE_MIRROR_VERTICAL_V1','PAT_COMPLETE_MIRROR_DIAGONAL_MAIN_V1','PAT_COMPLETE_MIRROR_DIAGONAL_ANTI_V1')]
