import json,csv
from pathlib import Path
from time import perf_counter
from arc.io import discover_dataset_files,load_dataset
from solvers.library_v3_pattern import PeriodicCompletionSolver,PatternRepairSolver,SymmetryCompletionSolver
from verification.verifier import verify_prediction
from primitives.registry import REGISTRY,validate_registry
S={'PeriodicCompletionSolver':PeriodicCompletionSolver,'PatternRepairSolver':PatternRepairSolver,'SymmetryCompletionSolver':SymmetryCompletionSolver};f=discover_dataset_files('data/raw');tasks=load_dataset(f['training_challenges']);start=perf_counter();data={};M={}
for n,C in S.items():
 rows={};g=c=r=a=pa=0
 for tid,t in tasks.items():
  s=C().fit(t);g+=s.generated_candidates;c+=len(s.candidates);r+=s.rejected_candidates;a+=len(s.candidates)>1; pa+=len(s.candidates)>1 and any(len({x.tobytes() for x in s.predict(e.input.values,len(s.candidates))})>1 for e in t.test);rows[tid]=(s,[s.predict(e.input.values,1) for e in t.test])
 data[n]=rows;M[n]={'applicable_tasks':sum(bool(x[0].candidates) for x in rows.values()),'generated_programs':g,'train_consistent_programs':c,'rejected_programs':r,'program_ambiguity_tasks':a,'prediction_ambiguity_tasks':pa}
scored=load_dataset(f['training_challenges'],f['training_solutions']);v3=set();wrong=0
for n,rows in data.items():
 sol=[];w=0
 for tid,t in scored.items():
  s,preds=rows[tid];ok=[bool(p) and verify_prediction(e.output.values,p[0]).exact_match for e,p in zip(t.test,preds)]
  if ok and all(ok):sol.append(tid);v3.add(tid)
  elif s.candidates:w+=1
 M[n].update({'exact_solved_tasks':len(sol),'solved_task_ids':sol,'train_consistent_but_test_wrong_tasks':w});wrong+=w
v0=json.load(open('experiments/results/SOLVER_LIB_V0.json'));v1=json.load(open('experiments/results/SOLVER_LIB_V1_OBJECT.json'));base=set(v0['unique_solved_task_ids'])|set(v1['newly_solved_task_ids']);P={'experiment_id':'SOLVER_LIB_V3_PATTERN','runtime_seconds':perf_counter()-start,'solver_metrics':M,'v0_unique':len(set(v0['unique_solved_task_ids'])),'pre_v3_unique':len(base),'v3_unique':len(v3),'overlap_existing':sorted(v3&base),'newly_solved_task_ids':sorted(v3-base),'cumulative_unique':len(base|v3),'absolute_gain':len(v3-base),'train_consistent_but_test_wrong':wrong,'program_ambiguity_tasks':sum(x['program_ambiguity_tasks'] for x in M.values()),'prediction_ambiguity_tasks':sum(x['prediction_ambiguity_tasks'] for x in M.values()),'registry_validation_errors':validate_registry(),'leakage_audit':['Challenge-only inference before solution loading.','No task-ID logic.'],'regression':{'v0_14':len(set(v0['unique_solved_task_ids']))==14,'pre_v3_18':len(base)==18}}
Path('experiments/results/SOLVER_LIB_V3_PATTERN.json').write_text(json.dumps(P,indent=2));Path('configs/primitive_registry_v1.json').write_text(json.dumps([x.json() for x in REGISTRY.values()],indent=2));Path('reports/pattern_v3_analysis.md').write_text(f'# Pattern V3\n\nV3 unique {len(v3)}; new {len(v3-base)}; cumulative {len(base|v3)}/1000.\n');print(json.dumps(P,indent=2))
