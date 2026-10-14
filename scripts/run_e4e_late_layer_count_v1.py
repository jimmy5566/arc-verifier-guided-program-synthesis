"""No-model production closure for E4E late-layer global-count supervision.

This module intentionally exposes only CPU validation in the prospective
freeze stage.  The future bound worker must use this exact loss contract:
the original Equal-Slot answer CE remains whole and the treatment adds a
layer-31 pooled-query three-way visible-count CE on Count-primary slots.
"""
from __future__ import annotations
import argparse, hashlib, importlib, json, os, shutil, subprocess, sys, tempfile
from pathlib import Path

ROOT=Path(os.environ.get('ARC2_ROOT',Path(__file__).resolve().parents[1]));sys.path[:0]=[str(ROOT),str(ROOT/'src')]
BASE=ROOT/'experiments/capability_repair_baseline_v1/connected_components_object_extraction_capability_pilot_v1'
E3=BASE/'deconfounded_count_curriculum_causal_pilot_v1';PKG=E3/'e4e_late_count_v1';IGNORE=-100
def canon(v):return json.dumps(v,sort_keys=True,separators=(',',':')).encode()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_bytes(canon(v)+b'\n');os.replace(tmp,p)
def fail(x):raise RuntimeError('E4E:'+x)
def schedule():
 rows=load(E3/'TREATMENT_SCHEDULE.json')['episodes']
 if len(rows)!=384 or [r['slot'] for r in rows]!=list(range(384)) or [r['optimizer_step'] for r in rows]!=[i//4 for i in range(384)]:fail('SCHEDULE')
 roles={role:sum(r['role']==role for r in rows) for role in ('PRIMARY_CONNECTED_COMPONENT_COUNT','PRIMARY_OBJECT_EXTRACTION','PROTECTED_REPLAY')}
 if roles!={'PRIMARY_CONNECTED_COMPONENT_COUNT':48,'PRIMARY_OBJECT_EXTRACTION':48,'PROTECTED_REPLAY':288}:fail('ROLES')
 return rows
def serializer_task(row):
 import copy
 return copy.deepcopy(row['task'])
def auxiliary_audit():
 from training_data.pipeline import task_to_sample
 from scripts.e4e_late_layer_global_count import AUXILIARY_COEFFICIENT,LAYER,final_query_grid_layout,loss_contract,multiclass_target_index
 audits=[]
 for row in schedule():
  task=serializer_task(row);sample=task_to_sample(task)
  if row['role']=='PRIMARY_CONNECTED_COMPONENT_COUNT':
   layout=final_query_grid_layout(task)
   if any(sample['labels'][p]!=IGNORE for p in layout.positions):fail('QUERY_NOT_INPUT_MASKED')
   audits.append({'slot':row['slot'],'count':layout.count,'class_index':multiclass_target_index(layout.count),'layer':LAYER,'query_position_count':len(layout.positions),'answer_coefficient':1.0,'auxiliary_coefficient':AUXILIARY_COEFFICIENT,'control':loss_contract(2.0,1.0,False),'treatment':loss_contract(2.0,1.0,True)})
  else:
   # Non-Count paths retain the unmodified Equal-Slot objective exactly.
   if loss_contract(2.0,1.0,False)['step_term'] != .5:fail('NONCOUNT_SCALAR')
 if len(audits)!=48 or {a['count']:sum(x['count']==a['count'] for x in audits) for a in audits}!={1:16,2:16,3:16}:fail('COUNT_GRID')
 if any(a['control']['answer_coefficient']!=a['treatment']['answer_coefficient'] or a['treatment']['slot_mean']<=a['control']['slot_mean'] for a in audits):fail('ANSWER_CE_REPLACED')
 return audits
def validate_panels():
 curriculum=load(PKG/'TRAIN_DERIVED_CURRICULUM.json');ood=load(PKG/'OOD_INPUTS_AND_COMMITMENTS.json')
 if curriculum['schedule_sha256']!=sha(E3/'TREATMENT_SCHEDULE.json') or len(curriculum['count_primary_slots'])!=48:fail('CURRICULUM')
 rows=ood['records'];units={r['canonical_unit_id'] for r in rows}
 if ood['status']!='FROZEN_TARGET_BLIND_INPUTS_AND_COMMITMENTS' or len(rows)!=96 or len(units)!=48 or ood['topology_count_mutual_information_bits']!=0.0:fail('OOD_GRID')
 seen={}
 for r in rows:
  if 'output' in r['prompt']['test'][0] or sha_bytes(canon(r['prompt']))!=r['prompt_sha256']:fail('OOD_TARGET_LEAK')
  seen.setdefault(r['canonical_unit_id'],set()).add(r['demo_order'])
 if any(v!={'A','B'} for v in seen.values()):fail('OOD_ORDERS')
 return curriculum,ood
def sha_bytes(data):return hashlib.sha256(data).hexdigest()
def make_cpu_binding(path):
 commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(); inputs={name:PKG/name for name in ('TRAIN_DERIVED_CURRICULUM.json','OOD_INPUTS_AND_COMMITMENTS.json','PROSPECTIVE_FREEZE_RECEIPT.json','PROTOCOL.json')}
 b={'schema_version':1,'experiment_id':'E4E_LATE_LAYER_GLOBAL_COUNT_BINDING_CAUSAL_PILOT_V1','execution_source_commit':commit,'worker_path':'scripts/run_e4e_late_layer_count_v1.py','worker_sha256':sha(Path(__file__)),'inputs':{k:{'path':v.relative_to(ROOT).as_posix(),'sha256':sha(v)} for k,v in inputs.items()},'training':{'exact_v7_initialization':True,'schedule_sha256':sha(E3/'TREATMENT_SCHEDULE.json'),'seed':20261013,'learning_rate':5e-5,'optimizer':'PagedAdamW8bit','updates':96,'slots_per_update':4,'slot_coefficient':.25,'control':'ORIGINAL_EQUAL_SLOT_ANSWER_AND_REPLAY_CE','treatment':'ORIGINAL_EQUAL_SLOT_ANSWER_AND_REPLAY_CE_PLUS_0.25_LAYER31_POOLED_QUERY_COUNT_CE','auxiliary_layer':31,'auxiliary_classes':[1,2,3]},'evaluation':{'native_greedy':True,'physical_batch_size':1,'raw_freeze_before_sidecar':True,'targets_blind_until_freeze':True}}
 write(path,b);return b
def preclaim(binding):
 b=load(binding)
 if b['worker_sha256']!=sha(Path(__file__)) or b['execution_source_commit']!=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip():fail('BINDING')
 for entry in b['inputs'].values():
  p=ROOT/entry['path']
  if not p.is_file() or sha(p)!=entry['sha256']:fail('INPUT_HASH')
 for name in ('training_data.pipeline','scripts.e04_e_equal_slot_loss','scripts.e4e_late_layer_global_count','scripts.connected_components_object_extraction_semantic_oracle','inference.nvarc_native'):importlib.import_module(name)
 audits=auxiliary_audit();curriculum,ood=validate_panels()
 return {'status':'PASS_E4E_PRODUCTION_PREFLIGHT_NO_MODEL','binding_sha256':sha(binding),'schedule_slots':384,'count_primary_slots':48,'auxiliary_audit':audits,'curriculum_sha256':sha(PKG/'TRAIN_DERIVED_CURRICULUM.json'),'ood_records':len(ood['records']),'model_loaded':False,'gpu_used':False,'sidecar_opened':False,'targets_opened':False}
def cpu_production_path(binding):
 """Exercise launcher-equivalent worker -> raw freeze -> sidecar -> score ordering."""
 pre=preclaim(binding);_,ood=validate_panels();tmp=Path(tempfile.mkdtemp(prefix='e4e-cpu-'))
 try:
  raw={}
  for arm in ('V7','EQUAL_SLOT','LATE_LAYER_COUNT'):
   rows=[{'record_id':r['record_id'],'canonical_unit_id':r['canonical_unit_id'],'demo_order':r['demo_order'],'prompt_sha256':r['prompt_sha256'],'parser_valid':False,'parsed_grid':None,'generated_token_ids':[]} for r in ood['records']]
   p=tmp/(arm+'_RAW.jsonl');p.write_bytes(b''.join(canon(x)+b'\n' for x in rows));raw[arm]=p
  hashes={a:sha(p) for a,p in raw.items()};write(tmp/'RAW_FREEZE_RECEIPT.json',{'status':'COMPLETE_RAW_FROZEN_NO_TARGETS','raw_sha256':hashes,'targets_accessed':False})
  freeze=load(tmp/'RAW_FREEZE_RECEIPT.json')
  if freeze['raw_sha256']!={a:sha(p) for a,p in raw.items()} or freeze['targets_accessed'] is not False:fail('RAW_FREEZE')
  # Only after the freeze can the deterministic oracle derive commitment-checked targets.
  from scripts.connected_components_object_extraction_semantic_oracle import target_for_family
  commitments={r['record_id']:r['target_sha256'] for r in ood['target_commitments']};targets={r['record_id']:target_for_family('CONNECTED_COMPONENT_COUNT',r['prompt']['test'][0]['input']) for r in ood['records']}
  if any(sha_bytes(canon(targets[k]))!=v for k,v in commitments.items()):fail('SIDE_CAR')
  write(tmp/'SYNTHETIC_TARGET_SIDECAR.json',{'status':'DERIVED_AFTER_RAW_FREEZE','targets':targets})
  write(tmp/'SCORE_RECEIPT.json',{'status':'COMPLETE_CPU_MOCK_SCORE_AFTER_RAW_FREEZE','arms':list(raw),'targets_opened_after_raw_freeze':True})
  return {'status':'PASS_E4E_LAUNCHER_WORKER_RAW_FREEZE_SIDECAR_SCORER_CPU_ONLY','preclaim_status':pre['status'],'raw_hashes':hashes,'model_loaded':False,'gpu_used':False,'sidecar_opened_after_freeze':True,'targets_opened_after_freeze':True}
 finally:shutil.rmtree(tmp)
def main():
 p=argparse.ArgumentParser();p.add_argument('--make-cpu-binding',type=Path);p.add_argument('--binding',type=Path);p.add_argument('--preclaim-only',action='store_true');p.add_argument('--cpu-production-path',action='store_true');p.add_argument('--receipt',type=Path);a=p.parse_args()
 if a.make_cpu_binding:r=make_cpu_binding(a.make_cpu_binding)
 elif a.preclaim_only:r=preclaim(a.binding)
 elif a.cpu_production_path:r=cpu_production_path(a.binding)
 else:raise SystemExit('select one mode')
 if a.receipt:write(a.receipt,r)
 print(json.dumps(r,sort_keys=True))
if __name__=='__main__':main()
