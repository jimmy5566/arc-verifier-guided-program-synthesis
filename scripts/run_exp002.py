"""EXP002: bounded train-only object-rule inference, shardable for reproducible local runs."""
from __future__ import annotations
import argparse, csv, json, subprocess
from time import perf_counter
from datetime import date
from pathlib import Path
import numpy as np
from arc.io import discover_dataset_files, load_dataset
from arc.visualization import plot_prediction
from solvers.baselines import ObjectRuleSolver
from representations.objects import extract_objects
from utils.environment import environment_dict
from verification.verifier import verify_prediction

parser=argparse.ArgumentParser(); parser.add_argument('--data-dir',default='data/raw'); parser.add_argument('--shard',type=int); parser.add_argument('--num-shards',type=int,default=1); parser.add_argument('--combine',action='store_true'); args=parser.parse_args()
OUT=Path('experiments/results'); OUT.mkdir(parents=True,exist_ok=True)

def taxonomy(task, candidate_count, predictions):
    if not candidate_count:
        objects=[extract_objects(e.input.values,background=int(np.bincount(e.input.values.ravel()).argmax())) for e in task.train]
        return 'multiple ambiguous objects' if any(len(x)>1 for x in objects) else 'no stable object correspondence'
    expected=task.test[0].output.values
    if all(p.shape != expected.shape for p in predictions): return 'output composition required'
    input_colors=set(np.unique(task.train[0].input.values)); output_colors=set(np.unique(expected))
    if len(output_colors-input_colors)>0: return 'object creation required'
    return 'search space insufficient'

def run_shard(shard: int, num_shards: int):
    files=discover_dataset_files(args.data_dir); tasks=load_dataset(files['evaluation_challenges'],files['evaluation_solutions']); selected=[(task_id,tasks[task_id]) for i,task_id in enumerate(sorted(tasks)) if i % num_shards==shard]
    records=[]
    for task_id,task in selected:
        started=perf_counter()
        solver=ObjectRuleSolver(10).fit(task)  # solver receives only task.train
        attempts_by_test=[solver.predict(example.input.values,2) for example in task.test]
        rank_exact=[]
        for rank in range(2): rank_exact.append(bool(attempts_by_test) and all(len(predictions)>rank and verify_prediction(example.output.values,predictions[rank]).exact_match for example,predictions in zip(task.test,attempts_by_test)))
        predictions=attempts_by_test[0] if attempts_by_test else []
        records.append({'task_id':task_id,'attempt_1':rank_exact[0],'attempt_2':rank_exact[1] if not rank_exact[0] else False,'candidate_count':len(solver.candidates),'top_candidate':solver.candidates[0].metadata if solver.candidates else None,'failure_category':None if any(rank_exact) else taxonomy(task,len(solver.candidates),predictions),'train_examples':len(task.train),'output_shape_change':any(example.input.shape != example.output.shape for example in task.train),'runtime_seconds':perf_counter()-started})
    path=OUT/f'EXP002_shard_{shard:02d}_of_{num_shards:02d}.json'; path.write_text(json.dumps(records,indent=2),encoding='utf-8'); print(f'wrote {len(records)} records to {path}')

def combine():
    records=[]
    for path in sorted(OUT.glob('EXP002_shard_*_of_*.json')): records.extend(json.loads(path.read_text(encoding='utf-8')))
    records=sorted({row['task_id']:row for row in records}.values(),key=lambda x:x['task_id'])
    if len(records)!=120: raise SystemExit(f'Expected 120 shard records, found {len(records)}')
    solved=[x for x in records if x['attempt_1'] or x['attempt_2']]; first=[x for x in records if x['attempt_1']]; second=[x for x in records if x['attempt_2']]
    exp1=json.loads(Path('experiments/results/EXP001.json').read_text())['result']; exp1_ids=set(exp1['solved_task_ids']); solved_ids={x['task_id'] for x in solved}
    result={'experiment_id':'EXP002','protocol':'public evaluation task-level in-context; rules infer from train pairs only','total_tasks':120,'attempt_1_solved':len(first),'attempt_2_additional_solved':len(second),'exact_solved':len(solved),'exact_accuracy':len(solved)/120,'runtime_seconds':float(sum(x['runtime_seconds'] for x in records)),'average_candidates_per_task':float(np.mean([x['candidate_count'] for x in records])),'max_candidates_per_task':max(x['candidate_count'] for x in records),'newly_solved_task_ids':sorted(solved_ids-exp1_ids),'regressions':sorted(exp1_ids-solved_ids),'common_solves':sorted(exp1_ids&solved_ids),'net_gain':len(solved_ids)-len(exp1_ids),'failure_categories':{category:sum(x['failure_category']==category for x in records) for category in sorted({x['failure_category'] for x in records if x['failure_category']})},'records':records,'environment':environment_dict()}
    OUT.joinpath('EXP002.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    table=Path('reports/tables/exp002_solved_tasks.csv'); table.parent.mkdir(parents=True,exist_ok=True)
    fields=['task_id','rule_family','selected_object_criterion','transformation','number_train_examples','output_shape_change','runtime_seconds','attempt_used']
    with table.open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader()
        for row in solved:
            top=row['top_candidate'] or {}; writer.writerow({'task_id':row['task_id'],'rule_family':top.get('family'),'selected_object_criterion':top.get('selector'),'transformation':f"{top.get('operation')}:{top.get('transform')}",'number_train_examples':row['train_examples'],'output_shape_change':row['output_shape_change'],'runtime_seconds':row['runtime_seconds'],'attempt_used':1 if row['attempt_1'] else 2})
    # Representative error visualisations are selected deterministically; no human task-by-task tuning.
    files=discover_dataset_files(args.data_dir); tasks=load_dataset(files['evaluation_challenges'],files['evaluation_solutions']); failures=[row for row in records if not (row['attempt_1'] or row['attempt_2'])][:15]; directory=Path('reports/failure_cases/EXP002'); directory.mkdir(parents=True,exist_ok=True)
    for row in failures:
        task=tasks[row['task_id']]; solver=ObjectRuleSolver(10).fit(task); predicted=solver.predict(task.test[0].input.values,1); plot_prediction(task.test[0].output.values,predicted[0] if predicted else task.test[0].input.values,directory/f"{row['task_id']}.png",row['failure_category'])
    log=Path('experiments/experiments.csv'); base={'experiment_id':'EXP002','date':str(date.today()),'git_commit':'uncommitted','solver':'ObjectRuleSolver','representation':'connected components + correspondence','search_method':'bounded object templates','llm_model':'','candidate_budget':10,'validation_split':'public_evaluation_tasks_in_context','tasks_solved':len(solved),'accuracy':len(solved)/120,'runtime_seconds':result['runtime_seconds'],'gpu_hours':0,'notes':json.dumps({'newly_solved':result['newly_solved_task_ids'],'failure_categories':result['failure_categories']})}
    with log.open(newline='',encoding='utf-8') as handle: previous=[x for x in csv.DictReader(handle) if x['experiment_id']!='EXP002']
    with log.open('w',newline='',encoding='utf-8') as handle: writer=csv.DictWriter(handle,fieldnames=base); writer.writeheader(); writer.writerows(previous); writer.writerow(base)
    print(json.dumps({key:result[key] for key in result if key not in ('records','environment')},indent=2))

if args.combine: combine()
elif args.shard is not None: run_shard(args.shard,args.num_shards)
else: parser.error('provide --shard N --num-shards M or --combine')
