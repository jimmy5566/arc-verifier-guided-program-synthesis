"""The sole E3 measurement-only successor: saved adapters, independent OOD, Batch1."""
from __future__ import annotations
import argparse, hashlib, importlib, json, os, random, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
for x in (ROOT,ROOT/'src'):
 if str(x) not in sys.path:sys.path.insert(0,str(x))
BASE=ROOT/'experiments/capability_repair_baseline_v1/connected_components_object_extraction_capability_pilot_v1'
PKG=BASE/'deconfounded_count_curriculum_causal_pilot_v1'
MEASURE=PKG/'batch1_only_independent_measurement_v1'
ID='E3_BATCH1_ONLY_INDEPENDENT_MEASUREMENT_SUCCESSOR_V1'
V7=ROOT/'experiments/capability_repair_baseline_v1/CHECKPOINT_MANIFEST_REMOTE_V1.json'
RUNTIME='/root/code_pair_py311_cu128_v2/bin/python'
OLD_RUN='/workspace/arc2/e3_deconfounded_count_causal_pilot/run_7cad545c'

def canon(x):return json.dumps(x,sort_keys=True,separators=(',',':')).encode()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_bytes(canon(x)+b'\n');os.replace(q,p)
def write_jsonl(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_bytes(b''.join(canon(v)+b'\n' for v in x));os.replace(q,p)
def fail(x):raise RuntimeError('E3_BATCH1:'+x)
def clean_source():
 if subprocess.run(['git','status','--porcelain'],cwd=ROOT,capture_output=True,text=True).stdout:fail('SOURCE_DIRTY')
def task_hash(task):return hashlib.sha256(canon(task)).hexdigest()
def inputs():return {
 'protocol':PKG/'PROTOCOL.json','manifest':PKG/'MANIFEST.json','cpu_validation':PKG/'CPU_VALIDATION.json',
 'control_schedule':PKG/'CONTROL_SCHEDULE.json','treatment_schedule':PKG/'TREATMENT_SCHEDULE.json',
 'train_prompts':PKG/'TRAIN_FIT_INPUT_PROMPTS.json','new_ood':MEASURE/'FRESH_STRUCTURAL_OOD_INPUTS_AND_COMMITMENTS.json',
 'cohort_freeze':MEASURE/'COHORT_FREEZE_RECEIPT.json','measurement_protocol':MEASURE/'MEASUREMENT_PROTOCOL.json','exposed_ood':BASE/'post_v7_generalization_mechanism_audit_v1/FRESH_STRUCTURAL_OOD_INPUTS_AND_COMMITMENTS.json','protected':BASE/'measurement_contract_v1/PROTECTED_CAPABILITY_INPUT_PROMPTS.json',
 'composition':BASE/'scientific_successor_v2/COMPOSITION_SENTINEL_INPUT_PROMPTS.json','v7_manifest':V7,
}
def adapter_entry(tag):
 root=Path(OLD_RUN)/tag/'adapter';files=[]
 for name in ('adapter_model.safetensors','adapter_config.json'):
  p=root/name;files.append({'name':name,'bytes':p.stat().st_size,'sha256':sha(p)})
 return {'path':str(root),'files':files}
def make_binding(path,output_root):
 clean_source();commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
 b={'schema_version':1,'experiment_id':ID,'invalid_predecessor':'run_7cad545c:INVALID_NOT_INTERPRETABLE_BATCH1_GATE','execution_source_commit':commit,
 'inputs':{k:{'path':str(v.relative_to(ROOT)).replace('\\','/'),'sha256':sha(v)} for k,v in inputs().items()},
 'worker_path':'scripts/run_e3_batch1_only_measurement_successor_v1.py','worker_sha256':sha(Path(__file__)),'output_root':str(output_root),'runtime_python':RUNTIME,
 'runtime':{'base_path':load(V7)['base_path'],'context_window':16384,'max_new_tokens':192,'physical_batch_size':1},
 'saved_adapters':{'CONTROL_ORIGINAL_CONFOUNDED':adapter_entry('CONTROL_ORIGINAL_CONFOUNDED'),'TREATMENT_DECONFOUNDED':adapter_entry('TREATMENT_DECONFOUNDED')},
 'hard_cap_seconds':3600,'retry':False,'nonce':hashlib.sha256((commit+str(output_root)+str(time.time_ns())).encode()).hexdigest()[:32],
 'evaluation':{'physical_batch_size':1,'fallback_ladder':[1],'primary_canonical_units':48,'bootstrap_replicates':10000,'bootstrap_seed':20261013},
 'invariants':{'no_retraining':True,'raw_freeze_before_sidecar':True,'new_ood_only':True,'target_blind_inference':True,'duplicate_output_guard':True,'no_batch32_or_fallback':True}}
 write(path,b);return b
def schedules():
 from training_data.pipeline import task_to_sample
 a,b=load(PKG/'CONTROL_SCHEDULE.json'),load(PKG/'TREATMENT_SCHEDULE.json')
 for name,x in [('control',a),('treatment',b)]:
  if len(x['episodes'])!=384 or [r['slot'] for r in x['episodes']]!=list(range(384)):fail('SCHEDULE:'+name)
  if len({r['optimizer_step'] for r in x['episodes']})!=96:fail('SCHEDULE_STEPS:'+name)
  if sum(task_to_sample(r['task'])['assistant_token_count'] for r in x['episodes'])!=27360:fail('SCHEDULE_TOKENS:'+name)
 for x,y in zip(a['episodes'],b['episodes'],strict=True):
  if x['role']!='PRIMARY_CONNECTED_COMPONENT_COUNT' and canon(x)!=canon(y):fail('NON_COUNT_DRIFT')
 return a,b
def validate_cohort():
 c=load(MEASURE/'FRESH_STRUCTURAL_OOD_INPUTS_AND_COMMITMENTS.json');r=c.get('records',[])
 if c.get('status')!='FROZEN_TARGET_BLIND_INPUTS_AND_COMMITMENTS' or len(r)!=96 or len({x['canonical_unit_id'] for x in r})!=48:fail('COHORT_GRID')
 if any('output' in x['prompt']['test'][0] for x in r):fail('TARGET_LEAK')
 if not c.get('zero_overlap',{}).get('training_and_exposed_ood_prompt_sha256') or not c['zero_overlap'].get('training_and_exposed_ood_query_input_sha256'):fail('COHORT_OVERLAP_DECLARATION')
 prior_tasks=[]
 for name in ('CONTROL_SCHEDULE.json','TREATMENT_SCHEDULE.json'):
  prior_tasks += [x['task'] for x in load(PKG/name)['episodes']]
 prior_tasks += [x['prompt'] for x in load(BASE/'post_v7_generalization_mechanism_audit_v1/FRESH_STRUCTURAL_OOD_INPUTS_AND_COMMITMENTS.json')['records']]
 prior_prompt={task_hash(x) for x in prior_tasks};prior_query={hashlib.sha256(canon(x['test'][0]['input'])).hexdigest() for x in prior_tasks}
 for x in r:
  if x.get('prompt_sha256')!=task_hash(x['prompt']) or x.get('query_input_sha256')!=hashlib.sha256(canon(x['prompt']['test'][0]['input'])).hexdigest():fail('COHORT_HASH')
  if x['prompt_sha256'] in prior_prompt or x['query_input_sha256'] in prior_query:fail('COHORT_OVERLAP')
 seen={}
 for x in r:seen.setdefault(x['canonical_unit_id'],set()).add(x['demo_order'])
 if any(v!={'A','B'} for v in seen.values()):fail('COHORT_ORDERS')
 return c
def records(arm=None):
 out=[];ood=validate_cohort()
 for r in ood['records']:out.append({'id':r['record_id'],'split':'OOD_INDEPENDENT','canonical':r['canonical_unit_id'],'order':r['demo_order'],'task':r['prompt']})
 for r in load(inputs()['protected'])['records']:out.append({'id':'PROTECTED:'+r['canonical_base_id']+':'+r['condition']+':'+str(r['control_marker_turn']),'split':'RETENTION_SECONDARY','task':r['prompt'],'protected':r})
 for r in load(inputs()['composition']):out.append({'id':'COMPOSITION:'+r['prompt_id'],'split':'COMPOSITION_SECONDARY','task':r['task']})
 if arm:
  for r in load(PKG/'TRAIN_FIT_INPUT_PROMPTS.json')['records']:
   if r['arm']==arm:out.append({'id':r['record_id'],'split':'TRAIN_FIT_SECONDARY','task':r['prompt']})
 if any('output' in r['task']['test'][0] for r in out):fail('TARGET_LEAK')
 return out
def preclaim(binding_path):
 b=load(binding_path)
 if b.get('experiment_id')!=ID or b.get('retry') is not False or b.get('worker_sha256')!=sha(Path(__file__)):fail('BINDING')
 if Path(b['output_root']).exists():fail('OUTPUT_EXISTS')
 if b.get('hard_cap_seconds',99999)>3600 or b.get('runtime',{}).get('physical_batch_size')!=1 or b.get('evaluation',{}).get('fallback_ladder')!=[1] or not b.get('invariants',{}).get('no_retraining'):fail('BATCH1_BINDING')
 clean_source()
 for k,v in b['inputs'].items():
  p=ROOT/v['path']
  if not p.is_file() or sha(p)!=v['sha256']:fail('INPUT:'+k)
 if load(PKG/'CPU_VALIDATION.json').get('status')!='PASS_CPU_ONLY_SCIENTIFIC_PACKAGE_VALIDATION':fail('CPU_VALIDATION')
 schedules();cohort=validate_cohort()
 for name in ('inference.nvarc_native','training_data.pipeline','scripts.connected_components_object_extraction_semantic_oracle','scripts.e04_orientation_fixed_demonstration_baseline_v3','scripts.run_connected_components_object_extraction_pilot_v1') :importlib.import_module(name)
 if not Path(b['runtime_python']).is_file():fail('RUNTIME_PYTHON')
 m=load(V7)
 for root,files in ((m['adapter_path'],m['adapter_files']),(m['base_path'],m['base_files'])):
  for f in files:
   p=Path(root)/f['name']
   if not p.is_file() or p.stat().st_size!=f['bytes'] or sha(p)!=f['sha256']:fail('V7_ASSET:'+f['name'])
 for tag,e in b['saved_adapters'].items():
  for f in e['files']:
   p=Path(e['path'])/f['name']
   if not p.is_file() or p.stat().st_size!=f['bytes'] or sha(p)!=f['sha256']:fail('SAVED_ADAPTER:'+tag+':'+f['name'])
 gpu=subprocess.run(['nvidia-smi','--query-gpu=memory.total','--format=csv,noheader'],capture_output=True,text=True)
 if gpu.returncode or int(gpu.stdout.split()[0])<24576:fail('GPU_CAPACITY')
 return {'status':'PASS_E3_BATCH1_ONLY_PRODUCTION_PREFLIGHT_NO_MODEL','binding_sha256':sha(binding_path),'execution_source_commit':b['execution_source_commit'],'records':{'V7':len(records()),'CONTROL':len(records('CONTROL_ORIGINAL_CONFOUNDED')),'TREATMENT':len(records('TREATMENT_DECONFOUNDED'))},'new_ood':{'canonical_units':48,'records':96,'zero_overlap':cohort['zero_overlap']},'model_loaded':False,'gpu_used':False,'sidecar_opened':False,'physical_batch_size':1}
def load_model(binding,tag):
 import torch
 from peft import PeftModel
 from transformers import AutoModelForCausalLM
 if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():fail('CUDA_BF16')
 m=load(V7);base=AutoModelForCausalLM.from_pretrained(m['base_path'],local_files_only=True,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True).to('cuda:0')
 path=m['adapter_path'] if tag=='V7' else binding['saved_adapters'][tag]['path']
 model=PeftModel.from_pretrained(base,path,is_trainable=False);model.eval();return model,torch,m
def generate(model,torch,manifest,items,binding,out,tag,deadline):
 from scripts import run_connected_components_object_extraction_pilot_v1 as real
 translated={'runtime':{'context_window':binding['runtime']['context_window']},'evaluation':{'fallback_ladder':[1],'max_new_tokens':binding['runtime']['max_new_tokens']},'training':{'torch_seed':binding['evaluation']['bootstrap_seed']}}
 rows=[{'record_id':r['id'],'split':r['split'],'task':r['task']} for r in items]
 return real._generate(model,torch,manifest,rows,translated,out,deadline,tag,force_batch_size=1)
def make_sidecar(binding_path,out):
 freeze=load(Path(out)/'RAW_FREEZE_RECEIPT.json')
 if freeze.get('status')!='COMPLETE_RAW_FROZEN_NO_TARGETS' or freeze.get('targets_accessed') is not False:fail('RAW_FREEZE_REQUIRED')
 from scripts.connected_components_object_extraction_semantic_oracle import target_for_family
 from scripts import e04_orientation_fixed_demonstration_baseline_v3 as orient
 bases={x['canonical_base_id']:x for x in orient.v2.canonical_bases('VALIDATION')};targets={}
 for row in records('CONTROL_ORIGINAL_CONFOUNDED')+records('TREATMENT_DECONFOUNDED'):
  if row['split'] in {'OOD_INDEPENDENT','TRAIN_FIT_SECONDARY'}:targets[row['id']]=target_for_family('CONNECTED_COMPONENT_COUNT',row['task']['test'][0]['input'])
  elif row['split']=='COMPOSITION_SECONDARY':targets[row['id']]=target_for_family('COUNT_THEN_OBJECT_EXTRACTION',row['task']['test'][0]['input'])
  else:
   q=row['protected'];targets[row['id']]=orient.target(bases[q['canonical_base_id']],q['control_marker_turn'],q['condition'],q['source_row_index'])['target']
 commitments={x['record_id']:x['target_sha256'] for x in validate_cohort()['target_commitments']}
 commitments.update({x['record_id']:x['target_sha256'] for x in load(PKG/'TRAIN_FIT_INPUT_PROMPTS.json')['target_commitments']})
 for k,v in commitments.items():
  if k not in targets or hashlib.sha256(canon(targets[k])).hexdigest()!=v:fail('TARGET_COMMITMENT:'+k)
 p=Path(out)/'SYNTHETIC_TARGET_SIDECAR.json';write(p,{'status':'DERIVED_AFTER_RAW_FREEZE','targets':targets,'commitments_verified':len(commitments),'opened_after_raw_freeze':True});return p
def scored_rows(out,freeze,targets,tag,arm):
 want={x['id']:x for x in records(arm)};p=Path(out)/(tag+'_RAW.jsonl')
 if not p.is_file() or sha(p)!=freeze.get('raw_sha256',{}).get(tag):fail('RAW_IDENTITY:'+tag)
 rows=[json.loads(x) for x in p.read_text().splitlines() if x];ids=[x.get('record_id') for x in rows]
 if len(ids)!=len(set(ids)) or set(ids)!=set(want):fail('RAW_COMPLETENESS:'+tag)
 result={}
 for x in rows:
  if x.get('prompt_sha256')!=task_hash(want[x['record_id']]['task']) or x.get('effective_batch_size')!=1:fail('RAW_BATCH1_IDENTITY:'+tag+':'+x['record_id'])
  result[x['record_id']]={**x,'exact':bool(x.get('parser_valid')) and x.get('parsed_grid')==targets[x['record_id']]}
 return result
def score(binding_path,out,sidecar):
 b=load(binding_path);freeze=load(Path(out)/'RAW_FREEZE_RECEIPT.json');targets=load(sidecar)['targets']
 if freeze.get('status')!='COMPLETE_RAW_FROZEN_NO_TARGETS' or freeze.get('targets_accessed') is not False:fail('FREEZE')
 v=scored_rows(out,freeze,targets,'V7',None);c=scored_rows(out,freeze,targets,'CONTROL_ORIGINAL_CONFOUNDED','CONTROL_ORIGINAL_CONFOUNDED');t=scored_rows(out,freeze,targets,'TREATMENT_DECONFOUNDED','TREATMENT_DECONFOUNDED')
 ood=[x for x in records() if x['split']=='OOD_INDEPENDENT'];units=sorted({x['canonical'] for x in ood})
 def both(rows,u):return int(all(rows[x['id']]['exact'] for x in ood if x['split']=='OOD_INDEPENDENT' and x['canonical']==u))
 diff=[both(t,u)-both(c,u) for u in units];rng=random.Random(b['evaluation']['bootstrap_seed']);draws=sorted(sum(rng.choice(diff) for _ in diff) for _ in range(b['evaluation']['bootstrap_replicates']));delta=sum(diff);lo,hi=draws[249],draws[9749]
 fit_ids={r['id'] for r in records('TREATMENT_DECONFOUNDED') if r['split']=='TRAIN_FIT_SECONDARY'};fit=sum(x['exact'] for x in t.values() if x['record_id'] in fit_ids)
 retention={}
 allr=records()
 for cond in sorted({x['protected']['condition'] for x in allr if x['split']=='RETENTION_SECONDARY'}):
  ids=[x['id'] for x in allr if x['split']=='RETENTION_SECONDARY' and x['protected']['condition']==cond];retention[cond]={'treatment_minus_control':sum(t[i]['exact']-c[i]['exact'] for i in ids),'rows':len(ids)}
 comp=[x['id'] for x in allr if x['split']=='COMPOSITION_SECONDARY'];composition={'control':sum(c[i]['exact'] for i in comp),'treatment':sum(t[i]['exact'] for i in comp),'treatment_minus_control':sum(t[i]['exact']-c[i]['exact'] for i in comp),'rows':len(comp)}
 any_t=[u for u in units if any(t[x['id']]['exact'] for x in ood if x['canonical']==u)];any_c=[u for u in units if any(c[x['id']]['exact'] for x in ood if x['canonical']==u)];both_t=sum(both(t,u) for u in any_t);both_c=sum(both(c,u) for u in any_c)
 aggregate=sum(x['treatment_minus_control'] for x in retention.values());gates={'primary':delta>=6 and lo>0,'train_fit_secondary':fit>=43,'order_invariance':(not any_t or both_t/len(any_t)>=.8) and (not any_c or both_c/len(any_c)>=.8),'retention_secondary':all(x['treatment_minus_control']>=-1 for x in retention.values()) and aggregate>=-2,'composition_secondary':composition['treatment_minus_control']>=-1}
 decision='INCONCLUSIVE' if gates['primary'] else 'REJECT'
 receipt={'status':'COMPLETE_E3_BATCH1_ONLY_SCORE_AFTER_RAW_FREEZE','primary_independent_ood':{'canonical_units':48,'treatment_minus_control':{'delta_correct':delta,'bootstrap_95_ci':[lo,hi],'replicates':10000},'gate':gates['primary']},'secondary_descriptive':{'train_fit_treatment':fit,'retention':{'by_family':retention,'aggregate_treatment_minus_control':aggregate},'composition':composition,'order_invariance':{'treatment_both_orders_correct':both_t,'treatment_any_order_correct':len(any_t),'control_both_orders_correct':both_c,'control_any_order_correct':len(any_c)}},'gates':gates,'decision':decision,'promotion':'FORBIDDEN_REQUIRES_LATER_INDEPENDENT_CONFIRMATION','physical_batch_size':1,'batch32_used':False,'targets_opened_after_raw_freeze':True}
 write(Path(out)/'SCORE_RECEIPT.json',receipt);return receipt
def score_only(binding_path):
 b=load(binding_path);out=Path(b['output_root']);terminal=out/'TERMINAL_RECEIPT.json';sidecar=out/'SYNTHETIC_TARGET_SIDECAR.json'
 if not terminal.is_file() or not sidecar.is_file() or not (out/'RAW_FREEZE_RECEIPT.json').is_file():fail('SCORE_REPAIR_PREREQUISITE')
 prior=load(terminal)
 if prior.get('status')!='FAILED_OR_PARTIAL_NO_RETRY' or prior.get('error')!="KeyError:'canonical'":fail('SCORE_REPAIR_TERMINAL_IDENTITY')
 result=score(binding_path,out,sidecar)
 receipt={'status':'COMPLETE_POST_RAW_SCORER_REPAIR_NO_MODEL','original_terminal_sha256':sha(terminal),'binding_sha256':sha(binding_path),'score_sha256':sha(out/'SCORE_RECEIPT.json'),'raw_freeze_sha256':sha(out/'RAW_FREEZE_RECEIPT.json'),'model_loaded':False,'gpu_used':False,'raw_predictions_reused':True,'original_terminal_preserved':True,'repair_worker_sha256':sha(Path(__file__)),'repair_source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()}
 write(out/'SCORER_REPAIR_RECEIPT.json',receipt);return receipt

def execute(binding_path):
 pre=preclaim(binding_path);b=load(binding_path);out=Path(b['output_root']);out.mkdir(parents=True);started=time.monotonic();deadline=started+b['hard_cap_seconds'];write(out/'PREFLIGHT_RECEIPT.json',pre)
 try:
  write(out/'SCIENTIFIC_BOUNDARY.json',{'status':'FIRST_MODEL_LOAD_NEXT','nonce':b['nonce'],'retry':False,'no_retraining':True})
  raw={}
  for tag,arm in [('V7',None),('CONTROL_ORIGINAL_CONFOUNDED','CONTROL_ORIGINAL_CONFOUNDED'),('TREATMENT_DECONFOUNDED','TREATMENT_DECONFOUNDED')]:
   model,torch,m=load_model(b,tag);raw[tag]=generate(model,torch,m,records(arm),b,out,tag,deadline);del model;torch.cuda.empty_cache()
  write(out/'RAW_FREEZE_RECEIPT.json',{'status':'COMPLETE_RAW_FROZEN_NO_TARGETS','raw_sha256':raw,'targets_accessed':False,'nonce':b['nonce'],'physical_batch_size':1})
  sidecar=make_sidecar(binding_path,out);score(binding_path,out,sidecar);write(out/'TERMINAL_RECEIPT.json',{'status':'COMPLETE','raw_freeze_before_sidecar':True,'score_sha256':sha(out/'SCORE_RECEIPT.json')})
 except Exception as e:
  write(out/'TERMINAL_RECEIPT.json',{'status':'FAILED_OR_PARTIAL_NO_RETRY','error':type(e).__name__+':'+str(e),'raw_predictions_created':(out/'RAW_FREEZE_RECEIPT.json').exists(),'scientific_boundary_entered':(out/'SCIENTIFIC_BOUNDARY.json').exists()});raise
def main():
 p=argparse.ArgumentParser();p.add_argument('--binding',type=Path,required=True);p.add_argument('--output-root');p.add_argument('--make-binding',action='store_true');p.add_argument('--preclaim-only',action='store_true');p.add_argument('--execute',action='store_true');p.add_argument('--score-only',action='store_true');a=p.parse_args()
 if a.make_binding:print(json.dumps(make_binding(a.binding,a.output_root),sort_keys=True))
 elif a.preclaim_only:print(json.dumps(preclaim(a.binding),sort_keys=True))
 elif a.execute:execute(a.binding)
 elif a.score_only:print(json.dumps(score_only(a.binding),sort_keys=True))
 else:raise SystemExit('select mode')
if __name__=='__main__':main()
