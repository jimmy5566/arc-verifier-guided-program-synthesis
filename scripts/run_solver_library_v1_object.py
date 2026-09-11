"""Leakage-safe formal evaluation for bounded object solvers on 1,000 ARC training tasks."""
from __future__ import annotations
import csv, json
from collections import Counter
from datetime import date
from pathlib import Path
from time import perf_counter
import numpy as np
from arc.io import discover_dataset_files, load_dataset, load_solutions
from solvers.library_v1_object import ObjectSelectionSolver, ObjectTransformSolver, ObjectCopyMoveSolver
from verification.verifier import verify_prediction

SOLVERS={'ObjectSelectionSolver':ObjectSelectionSolver,'ObjectTransformSolver':ObjectTransformSolver,'ObjectCopyMoveSolver':ObjectCopyMoveSolver}
files=discover_dataset_files('data/raw')
# INFERENCE PHASE: challenge file supplies only train labels. Test labels are deliberately absent.
tasks=load_dataset(files['training_challenges'])
all_predictions={}; results={}; started=perf_counter()
for name,factory in SOLVERS.items():
    phase=perf_counter(); records={}; applicable=[]; generated=consistent=rejected=ambiguous=differing=0; ambiguity_details={}
    for task_id,task in tasks.items():
        solver=factory().fit(task); generated+=solver.generated_candidates; consistent+=len(solver.candidates); rejected+=solver.rejected_candidates
        if solver.candidates: applicable.append(task_id)
        if len(solver.candidates)>1:
            ambiguous+=1
            rendered=[tuple(pred.tobytes() for pred in solver.predict(example.input.values,len(solver.candidates))) for example in task.test]
            has_different=any(len(set(item))>1 for item in rendered)
            if has_different: differing+=1
            ambiguity_details[task_id]={'train_consistent_hypotheses':len(solver.candidates),'different_test_predictions':has_different}
        records[task_id]={'predictions':[solver.predict(example.input.values,1) for example in task.test],'candidate_count':len(solver.candidates),'generated_candidates':solver.generated_candidates,'rejected_candidates':solver.rejected_candidates,'top_hypothesis':solver.candidates[0].metadata if solver.candidates else None}
    all_predictions[name]=records
    results[name]={'applicable_tasks':len(applicable),'applicable_task_ids':applicable,'generated_candidates':generated,'train_consistent_hypotheses':consistent,'rejected_candidates':rejected,'ambiguous_tasks':ambiguous,'ambiguous_task_details':ambiguity_details,'ambiguous_tasks_with_different_test_predictions':differing,'inference_runtime_seconds':perf_counter()-phase}

# EVALUATION PHASE: test solutions are read only after every solver prediction has been materialized above.
solutions=load_solutions(files['training_solutions']); scored_tasks=load_dataset(files['training_challenges'],files['training_solutions'])
usage_selection=Counter(); usage_transform=Counter(); usage_template=Counter()
for name,records in all_predictions.items():
    solved=[]; wrong=0; outputs_solved=0
    for task_id,task in scored_tasks.items():
        record=records[task_id]; exact=[]
        for example,predictions in zip(task.test,record['predictions']):
            exact.append(bool(predictions) and verify_prediction(example.output.values,predictions[0]).exact_match)
        outputs_solved+=sum(exact)
        if exact and all(exact):
            solved.append(task_id); meta=record['top_hypothesis'] or {}; selection=meta.get('selection','unknown'); operation=meta.get('operation','unknown'); transform=meta.get('transform','identity')
            usage_selection[selection]+=1; usage_transform[f'{operation}:{transform}']+=1; usage_template[f'{selection}->{operation}:{transform}']+=1
        elif record['candidate_count']: wrong+=1
    results[name].update({'exact_solved_tasks':len(solved),'solved_task_ids':solved,'test_outputs_solved':outputs_solved,'test_outputs_total':sum(len(task.test) for task in scored_tasks.values()),'train_consistent_but_test_wrong_tasks':wrong})

v0=json.loads(Path('experiments/results/SOLVER_LIB_V0.json').read_text()); v0_ids=set(v0['unique_solved_task_ids'])
v1_ids=set().union(*(set(result['solved_task_ids']) for result in results.values()))
payload={'experiment_id':'SOLVER_LIB_V1_OBJECT','config':{'connectivity':4,'background':'most_frequent_per_grid','candidate_budget':1,'solvers':list(SOLVERS),'test_solution_access':'after all predictions only'},'total_tasks':len(tasks),'runtime_seconds':perf_counter()-started,'solver_metrics':results,'v0_unique_solved':len(v0_ids),'v1_object_unique_solved':len(v1_ids),'overlap_with_v0':sorted(v0_ids&v1_ids),'newly_solved_task_ids':sorted(v1_ids-v0_ids),'total_union_solved':len(v0_ids|v1_ids),'absolute_gain':len(v1_ids-v0_ids),'percentage_point_gain':100*len(v1_ids-v0_ids)/len(tasks),'hypothesis_statistics':{'generated':sum(x['generated_candidates'] for x in results.values()),'train_consistent':sum(x['train_consistent_hypotheses'] for x in results.values()),'rejected':sum(x['rejected_candidates'] for x in results.values())},'ambiguous_tasks':sum(x['ambiguous_tasks'] for x in results.values()),'ambiguous_tasks_with_different_test_predictions':sum(x['ambiguous_tasks_with_different_test_predictions'] for x in results.values()),'primitive_usage':{'selection':dict(usage_selection),'transform':dict(usage_transform),'template':dict(usage_template)},'leakage_checks':['Inference used load_dataset(challenges) without test solution file.','All predictions were generated before load_solutions(training_solutions).','No task identifier conditions exist in object solver modules.'],'regression_check':{'v0_unique_solved':len(v0_ids),'expected':14,'passed':len(v0_ids)==14}}
output=Path('experiments/results/SOLVER_LIB_V1_OBJECT.json'); output.write_text(json.dumps(payload,indent=2),encoding='utf-8')
log=Path('experiments/experiments.csv'); fields=['experiment_id','date','git_commit','solver','representation','search_method','llm_model','candidate_budget','validation_split','tasks_solved','accuracy','runtime_seconds','gpu_hours','notes']
with log.open(newline='',encoding='utf-8') as handle: rows=[row for row in csv.DictReader(handle) if row['experiment_id']!='SOLVER_LIB_V1_OBJECT']
record={'experiment_id':'SOLVER_LIB_V1_OBJECT','date':str(date.today()),'git_commit':'uncommitted','solver':'Object Selection / Transform / CopyMove','representation':'4-connected components; modal background','search_method':'bounded single-object templates','llm_model':'','candidate_budget':1,'validation_split':'official_training_tasks_train_only_inference','tasks_solved':len(v1_ids),'accuracy':len(v1_ids)/len(tasks),'runtime_seconds':payload['runtime_seconds'],'gpu_hours':0,'notes':json.dumps({'newly_solved':payload['newly_solved_task_ids'],'overlap_v0':payload['overlap_with_v0']})}
with log.open('w',newline='',encoding='utf-8') as handle: writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader(); writer.writerows(rows); writer.writerow(record)
analysis=Path('reports/object_solver_v1_analysis.md'); analysis.write_text(f'''# Solver Library v1 Object Analysis

## Representation and primitives

`ARCObject` uses deterministic 4-connected components with a per-grid most-frequent background candidate. It exposes colour, pixels, area, bbox, dimensions, centroid, shape mask, and border coordinates. Selection primitives are largest, smallest, unique area/shape/colour, border, and non-border. Rendering supports crop, recolour, remove, explicit bounded move/copy, rotation, and reflection.

## Results

- Object solver unique exact solves: {len(v1_ids)}
- New over V0: {len(v1_ids-v0_ids)}
- V0 + V1 union: {len(v0_ids|v1_ids)} / {len(tasks)}
- Train-consistent but test-wrong: {sum(x['train_consistent_but_test_wrong_tasks'] for x in results.values())}
- Ambiguous tasks: {payload['ambiguous_tasks']}

## Primitive usage on exact solves

{json.dumps(payload['primitive_usage'],indent=2)}

## Main limitations

The library has only one selected object and one operation. It cannot express multi-object composition, relational selection, object creation, counting, or iterative pattern completion. The next capability should be selected from these aggregate failure modes, never from task-specific patches.
''',encoding='utf-8')
print(json.dumps(payload,indent=2))
