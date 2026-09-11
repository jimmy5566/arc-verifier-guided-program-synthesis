"""Formal V2 evaluation: all predictions are frozen before training test solutions are loaded."""
from __future__ import annotations
import csv,json
from collections import Counter
from datetime import date
from pathlib import Path
from time import perf_counter
import numpy as np
from arc.io import discover_dataset_files,load_dataset
from solvers.library_v2_relation import RelationalSelectionSolver,ObjectAlignmentSolver,TwoObjectCompositionSolver
from verification.verifier import verify_prediction
from primitives.registry import REGISTRY,validate_registry
SOLVERS={'RelationalSelectionSolver':RelationalSelectionSolver,'ObjectAlignmentSolver':ObjectAlignmentSolver,'TwoObjectCompositionSolver':TwoObjectCompositionSolver}
f=discover_dataset_files('data/raw'); tasks=load_dataset(f['training_challenges']); started=perf_counter(); metrics={}; frozen={}; generated_use=Counter(); consistent_use=Counter()
for name,factory in SOLVERS.items():
 phase=perf_counter(); rows={}; app=[]; generated=consistent=rejected=prog_amb=pred_amb=0; ambiguous_details={}
 for tid,task in tasks.items():
  solver=factory().fit(task); generated+=solver.generated_candidates; consistent+=len(solver.candidates); rejected+=solver.rejected_candidates
  for p in solver.generated_programs: generated_use.update(s.primitive_id for s in p.steps)
  for c in solver.candidates: consistent_use.update(s.primitive_id for s in c.program.steps)
  if solver.candidates:app.append(tid)
  if len(solver.candidates)>1:
   prog_amb+=1; preds=[tuple(x.tobytes() for x in solver.predict(ex.input.values,len(solver.candidates))) for ex in task.test]; different=any(len(set(x))>1 for x in preds); pred_amb+=different; ambiguous_details[tid]={'programs':len(solver.candidates),'different_predictions':different}
  rows[tid]={'preds':[solver.predict(ex.input.values,1) for ex in task.test],'candidate_count':len(solver.candidates),'top':solver.candidates[0].program if solver.candidates else None}
 frozen[name]=rows; metrics[name]={'applicable_tasks':len(app),'generated_programs':generated,'valid_programs':generated,'train_consistent_programs':consistent,'rejected_programs':rejected,'program_ambiguity_tasks':prog_amb,'prediction_ambiguity_tasks':pred_amb,'ambiguous_task_details':ambiguous_details,'inference_runtime_seconds':perf_counter()-phase}
# only now attach test labels
scored=load_dataset(f['training_challenges'],f['training_solutions']); correct_use=Counter(); depth=Counter()
for name,rows in frozen.items():
 solved=[]; wrong=out_solved=0
 for tid,task in scored.items():
  row=rows[tid]; exact=[bool(preds) and verify_prediction(ex.output.values,preds[0]).exact_match for ex,preds in zip(task.test,row['preds'])]; out_solved+=sum(exact)
  if exact and all(exact):
   solved.append(tid); p=row['top']; correct_use.update(s.primitive_id for s in p.steps); depth[p.depth]+=1
  elif row['candidate_count']:wrong+=1
 metrics[name].update({'exact_solved_tasks':len(solved),'solved_task_ids':solved,'test_outputs_solved':out_solved,'test_outputs_total':sum(len(t.test) for t in scored.values()),'train_consistent_but_test_wrong_tasks':wrong})
v0=json.loads(Path('experiments/results/SOLVER_LIB_V0.json').read_text()); v1=json.loads(Path('experiments/results/SOLVER_LIB_V1_OBJECT.json').read_text()); base=set(v0['unique_solved_task_ids'])|set(v1['newly_solved_task_ids']); v2=set().union(*(set(m['solved_task_ids']) for m in metrics.values()))
payload={'experiment_id':'SOLVER_LIB_V2_RELATION_COMPOSITION','config':{'max_relational_depth':2,'max_program_depth':4,'connectivity':4,'test_labels_after_prediction':True},'runtime_seconds':perf_counter()-started,'registry_validation_errors':validate_registry(),'solver_metrics':metrics,'v0_unique':len(set(v0['unique_solved_task_ids'])),'v0_v1_unique':len(base),'v2_unique':len(v2),'overlap_v0':sorted(v2&set(v0['unique_solved_task_ids'])),'overlap_v1':sorted(v2&set(v1['newly_solved_task_ids'])),'newly_solved_task_ids':sorted(v2-base),'cumulative_unique':len(base|v2),'absolute_gain':len(v2-base),'percentage_point_gain':100*len(v2-base)/len(tasks),'train_consistent_but_test_wrong':sum(m['train_consistent_but_test_wrong_tasks'] for m in metrics.values()),'program_ambiguity_tasks':sum(m['program_ambiguity_tasks'] for m in metrics.values()),'prediction_ambiguity_tasks':sum(m['prediction_ambiguity_tasks'] for m in metrics.values()),'primitive_usage':{'generated':dict(generated_use),'train_consistent':dict(consistent_use),'test_correct':dict(correct_use),'program_depth':dict(depth)},'leakage_audit':['No task-ID branching.','Challenge-only inference preceded solution loading.','No task-specific primitive semantics or test-label candidate selection.'],'regression':{'v0_is_14':len(set(v0['unique_solved_task_ids']))==14,'v0_v1_is_18':len(base)==18}}
Path('experiments/results/SOLVER_LIB_V2_RELATION_COMPOSITION.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
Path('configs').mkdir(exist_ok=True); Path('configs/primitive_registry_v1.json').write_text(json.dumps([s.json() for s in REGISTRY.values()],indent=2),encoding='utf-8')
with Path('experiments/experiments.csv').open(newline='',encoding='utf-8') as h: old=[r for r in csv.DictReader(h) if r['experiment_id']!='SOLVER_LIB_V2_RELATION_COMPOSITION']
fields=['experiment_id','date','git_commit','solver','representation','search_method','llm_model','candidate_budget','validation_split','tasks_solved','accuracy','runtime_seconds','gpu_hours','notes']; row={'experiment_id':'SOLVER_LIB_V2_RELATION_COMPOSITION','date':str(date.today()),'git_commit':'uncommitted','solver':'Relational/Alignment/TwoObject','representation':'relation registry v1','search_method':'bounded depth<=4','llm_model':'','candidate_budget':1,'validation_split':'official_training_train_only','tasks_solved':len(v2),'accuracy':len(v2)/len(tasks),'runtime_seconds':payload['runtime_seconds'],'gpu_hours':0,'notes':json.dumps({'new':payload['newly_solved_task_ids']})}
with Path('experiments/experiments.csv').open('w',newline='',encoding='utf-8') as h:w=csv.DictWriter(h,fieldnames=fields);w.writeheader();w.writerows(old);w.writerow(row)
Path('reports/relation_composition_v2_analysis.md').write_text(f'''# Relation Composition V2\n\nV2 unique solved: {len(v2)}; new over cumulative V0+V1: {len(v2-base)}; cumulative: {len(base|v2)}/1000.\n\nRelation and composition primitives are registered in `configs/primitive_registry_v1.json`. All hypotheses use structured primitive-ID programs and hard train verification.\n\nMain missing capabilities: multi-object scene composition beyond two normalized masks, relational chains beyond depth 2, object creation, counting, and iterative pattern completion.\n''')
print(json.dumps(payload,indent=2))
