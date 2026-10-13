"""Production worker for QUERY_ONLY_COUNT_OBJECTIVE_CAUSAL_PILOT_V1.

The only changed loss path masks the first three assistant turns of the 48 Count
primary slots.  Every other slot is passed byte-for-byte to the established
Equal-Slot implementation.  Targets are derived only after raw predictions
are frozen.
"""
from __future__ import annotations
import argparse,copy,hashlib,importlib,json,os,random,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
for p in (ROOT,ROOT/'src'):
 if str(p) not in sys.path:sys.path.insert(0,str(p))
BASE=ROOT/'experiments/capability_repair_baseline_v1/connected_components_object_extraction_capability_pilot_v1'
E3=BASE/'deconfounded_count_curriculum_causal_pilot_v1'
PKG=E3/'query_only_count_objective_causal_pilot_v1'
ID='QUERY_ONLY_COUNT_OBJECTIVE_CAUSAL_PILOT_V1'
V7=ROOT/'experiments/capability_repair_baseline_v1/CHECKPOINT_MANIFEST_REMOTE_V1.json'
RUNTIME='/root/code_pair_py311_cu128_v2/bin/python'
REFERENCE=Path('/workspace/arc2/e3_deconfounded_count_causal_pilot/run_7cad545c/TREATMENT_DECONFOUNDED/adapter')
IGNORE=-100
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
def fail(x):raise RuntimeError('QUERY_ONLY:'+x)
def task_hash(x):return hashlib.sha256(canon(x)).hexdigest()
def clean_source():
 if subprocess.run(['git','status','--porcelain'],cwd=ROOT,text=True,capture_output=True).stdout:fail('SOURCE_DIRTY')
def serializer_task(item):
 t=copy.deepcopy(item['task']);t.setdefault('source_id',f"SERIALIZER_VIEW:{item['canonical_base_id']}");return t
def inputs():
 return {
  'protocol':PKG/'PROTOCOL.json','e3_protocol':E3/'PROTOCOL.json','e3_treatment_schedule':E3/'TREATMENT_SCHEDULE.json',
  'e3_train_prompts':E3/'TRAIN_FIT_INPUT_PROMPTS.json','new_ood':PKG/'NEW_STRUCTURAL_OOD_INPUTS_AND_COMMITMENTS.json',
  'ood_freeze':PKG/'OOD_COHORT_FREEZE_RECEIPT.json','protected':BASE/'measurement_contract_v1/PROTECTED_CAPABILITY_INPUT_PROMPTS.json',
  'composition':BASE/'scientific_successor_v2/COMPOSITION_SENTINEL_INPUT_PROMPTS.json','v7_manifest':V7,
  'e3_exposed_ood':E3/'batch1_only_independent_measurement_v1/FRESH_STRUCTURAL_OOD_INPUTS_AND_COMMITMENTS.json'
 }
def file_entry(p):
 p=Path(p);return {'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)}
def make_binding(path,output_root):
 clean_source();commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
 b={'schema_version':1,'experiment_id':ID,'execution_source_commit':commit,
 'inputs':{k:{'path':str(v.relative_to(ROOT)).replace('\\\\','/'),'sha256':sha(v)} for k,v in inputs().items()},
 'worker_path':'scripts/run_query_only_count_objective_causal_pilot_v1.py','worker_sha256':sha(Path(__file__)),'output_root':str(output_root),
 'runtime_python':RUNTIME,'hard_cap_seconds':7200,'retry':False,
 'nonce':hashlib.sha256((commit+str(output_root)+str(time.time_ns())).encode()).hexdigest()[:32],
 'runtime':{'context_window':16384,'max_new_tokens':192,'physical_batch_size':1,'cuda_required':True},
 'reference_equal_slot_adapter':{'source':'E3_DECONFOUNDED_TREATMENT_SAVED_ADAPTER','root':str(REFERENCE),'files':[file_entry(REFERENCE/'adapter_model.safetensors'),file_entry(REFERENCE/'adapter_config.json')]},
 'v7':{'manifest_path':str(V7.relative_to(ROOT)).replace('\\\\','/'),'manifest_sha256':sha(V7),'adapter_model_sha256':'1639bf6665ccfbb99cf547f59fcd581663c95b6bad770ae7846fee90f66be1c3'},
 'training':{'schedule_path':str((E3/'TREATMENT_SCHEDULE.json').relative_to(ROOT)).replace('\\\\','/'),'updates':96,'slots_per_update':4,'torch_seed':20261013,'learning_rate':2e-4,'loss':'QUERY_ONLY_COUNT_FINAL_ASSISTANT_MEAN_WITH_EQUAL_SLOT_COEFFICIENT'},
 'evaluation':{'bootstrap_seed':20261013,'bootstrap_replicates':10000,'physical_batch_size':1,'fallback_ladder':[1]},
 'invariants':{'exact_e3_treatment_schedule':True,'reference_not_retrained':True,'only_count_primary_masked':True,'raw_freeze_before_sidecar':True,'target_blind_inference':True,'no_code_pair_locator':True,'duplicate_output_guard':True,'no_batch32_or_fallback':True}}
 write(path,b);return b
def schedule():
 from training_data.pipeline import task_to_sample
 rows=load(E3/'TREATMENT_SCHEDULE.json')['episodes']
 if len(rows)!=384 or [x['slot'] for x in rows]!=list(range(384)):fail('SCHEDULE_SLOTS')
 if [x['optimizer_step'] for x in rows]!=[i//4 for i in range(384)]:fail('SCHEDULE_GROUPING')
 roles={r:sum(x['role']==r for x in rows) for r in ('PRIMARY_CONNECTED_COMPONENT_COUNT','PRIMARY_OBJECT_EXTRACTION','PROTECTED_REPLAY')}
 if roles!={'PRIMARY_CONNECTED_COMPONENT_COUNT':48,'PRIMARY_OBJECT_EXTRACTION':48,'PROTECTED_REPLAY':288}:fail('SCHEDULE_ROLES')
 toks=sum(task_to_sample(serializer_task(x))['assistant_token_count'] for x in rows)
 if toks!=27360:fail('SCHEDULE_TOKEN_BUDGET')
 return rows
def query_only_labels(task):
 """Select precisely the final assistant turn in an exact native sample."""
 from training_data.pipeline import task_to_sample
 sample=task_to_sample(task);labels=sample['labels'];selected=[False]*len(labels);runs=[];start=None
 for i,v in enumerate(labels+[IGNORE]):
  if v!=IGNORE and start is None:start=i
  if v==IGNORE and start is not None:runs.append((start,i));start=None
 if len(runs)!=len(task['train'])+1:fail('ASSISTANT_TURN_COUNT')
 a,b=runs[-1]
 for i in range(a,b):selected[i]=True
 masked=[v if selected[i] else IGNORE for i,v in enumerate(labels)]
 if any(masked[i]!=IGNORE for i in range(a)) or any(masked[i]==IGNORE for i in range(a,b)):fail('MASK_BOUNDARY')
 expected=sample['input_ids'][a:b]
 got=[masked[i] for i in range(a,b)]
 if got!=expected or len(got)!=15:fail('QUERY_TOKEN_IDENTITY')
 return sample,masked,{'assistant_turn_runs':runs,'selected_unshifted_indices':list(range(a,b)),'selected_token_ids':got,'selected_count':len(got),'all_assistant_count':sample['assistant_token_count']}
def mask_audit():
 rows=schedule();audits=[]
 for x in rows:
  s,m,a=query_only_labels(serializer_task(x))
  if x['role']=='PRIMARY_CONNECTED_COMPONENT_COUNT':
   if a['selected_count']!=15 or a['all_assistant_count']!=60:fail('COUNT_MASK_COUNTS')
   audits.append({'slot':x['slot'],'selected_count':a['selected_count'],'all_assistant_count':a['all_assistant_count'],'selected_token_ids_sha256':hashlib.sha256(canon(a['selected_token_ids'])).hexdigest(),'selected_indices_sha256':hashlib.sha256(canon(a['selected_unshifted_indices'])).hexdigest()})
  elif m==s['labels']:fail('NONCOUNT_MASK_SENTINEL')
 if len(audits)!=48:fail('MASK_AUDIT_GRID')
 return audits
def validate_cohort():
 c=load(PKG/'NEW_STRUCTURAL_OOD_INPUTS_AND_COMMITMENTS.json');rows=c.get('records',[])
 if c.get('status')!='FROZEN_TARGET_BLIND_INPUTS_AND_COMMITMENTS' or len(rows)!=96 or len({x['canonical_unit_id'] for x in rows})!=48:fail('OOD_GRID')
 if c.get('topology_count_independence',{}).get('mutual_information_bits')!=0.0:fail('OOD_INDEPENDENCE')
 if not all(c.get('zero_overlap',{}).get(x) for x in ('training_e3_exposed_ood_prompt_sha256','training_e3_exposed_ood_query_input_sha256')):fail('OOD_OVERLAP_DECLARATION')
 prior=[]
 for name in ('CONTROL_SCHEDULE.json','TREATMENT_SCHEDULE.json'):prior += [x['task'] for x in load(E3/name)['episodes']]
 for p in (BASE/'post_v7_generalization_mechanism_audit_v1/FRESH_STRUCTURAL_OOD_INPUTS_AND_COMMITMENTS.json',E3/'batch1_only_independent_measurement_v1/FRESH_STRUCTURAL_OOD_INPUTS_AND_COMMITMENTS.json'):prior += [x['prompt'] for x in load(p)['records']]
 ph={task_hash(x) for x in prior};qh={hashlib.sha256(canon(x['test'][0]['input'])).hexdigest() for x in prior}
 seen={}
 for x in rows:
  if 'output' in x['prompt']['test'][0]:fail('OOD_TARGET_LEAK')
  if x.get('prompt_sha256')!=task_hash(x['prompt']) or x.get('query_input_sha256')!=hashlib.sha256(canon(x['prompt']['test'][0]['input'])).hexdigest():fail('OOD_HASH')
  if x['prompt_sha256'] in ph or x['query_input_sha256'] in qh:fail('OOD_OVERLAP')
  seen.setdefault(x['canonical_unit_id'],set()).add(x['demo_order'])
 if any(x!={'A','B'} for x in seen.values()):fail('OOD_ORDERS')
 return c
def records(arm=None):
 out=[];ood=validate_cohort()
 for x in ood['records']:out.append({'id':x['record_id'],'split':'OOD_INDEPENDENT','canonical':x['canonical_unit_id'],'order':x['demo_order'],'task':x['prompt']})
 for x in load(inputs()['protected'])['records']:out.append({'id':'PROTECTED:'+x['canonical_base_id']+':'+x['condition']+':'+str(x['control_marker_turn']),'split':'RETENTION_SECONDARY','task':x['prompt'],'protected':x})
 for x in load(inputs()['composition']):out.append({'id':'COMPOSITION:'+x['prompt_id'],'split':'COMPOSITION_SECONDARY','task':x['task']})
 if arm:
  for x in load(E3/'TRAIN_FIT_INPUT_PROMPTS.json')['records']:
   if x['arm']=='TREATMENT_DECONFOUNDED':out.append({'id':x['record_id'],'split':'TRAIN_FIT_SECONDARY','task':x['prompt']})
 if any('output' in x['task']['test'][0] for x in out):fail('TARGET_LEAK')
 return out
def preclaim(binding_path):
 b=load(binding_path)
 if b.get('experiment_id')!=ID or b.get('retry') is not False or b.get('worker_sha256')!=sha(Path(__file__)):fail('BINDING')
 if Path(b['output_root']).exists():fail('OUTPUT_EXISTS')
 if b.get('hard_cap_seconds')!=7200 or b['runtime'].get('physical_batch_size')!=1 or b['evaluation'].get('fallback_ladder')!=[1]:fail('EXECUTION_CONTRACT')
 clean_source()
 for k,v in b['inputs'].items():
  p=ROOT/v['path']
  if not p.is_file() or sha(p)!=v['sha256']:fail('INPUT:'+k)
 if b['training']['schedule_path']!='experiments/capability_repair_baseline_v1/connected_components_object_extraction_capability_pilot_v1/deconfounded_count_curriculum_causal_pilot_v1/TREATMENT_SCHEDULE.json':fail('SCHEDULE_IDENTITY_PATH')
 rows=schedule();audit=mask_audit();cohort=validate_cohort()
 for name in ('inference.nvarc_native','training_data.pipeline','scripts.e04_e_equal_slot_loss','scripts.connected_components_object_extraction_semantic_oracle','scripts.e04_orientation_fixed_demonstration_baseline_v3','scripts.run_connected_components_object_extraction_pilot_v1'):importlib.import_module(name)
 if not Path(b['runtime_python']).is_file():fail('RUNTIME_PYTHON')
 m=load(V7)
 if sha(V7)!=b['v7']['manifest_sha256']:fail('V7_MANIFEST')
 for root,files in ((m['adapter_path'],m['adapter_files']),(m['base_path'],m['base_files'])):
  for f in files:
   p=Path(root)/f['name']
   if not p.is_file() or p.stat().st_size!=f['bytes'] or sha(p)!=f['sha256']:fail('V7_ASSET:'+f['name'])
 for f in b['reference_equal_slot_adapter']['files']:
  p=Path(b['reference_equal_slot_adapter']['root'])/Path(f['path']).name if 'path' in f else Path(b['reference_equal_slot_adapter']['root'])/Path(f['name']).name
  # file_entry stores absolute path; support immutable binding record.
  p=Path(f['path'])
  if not p.is_file() or p.stat().st_size!=f['bytes'] or sha(p)!=f['sha256']:fail('REFERENCE_ASSET:'+p.name)
 gpu=subprocess.run(['nvidia-smi','--query-gpu=memory.total','--format=csv,noheader'],capture_output=True,text=True)
 if gpu.returncode or int(gpu.stdout.split()[0])<24576:fail('GPU_CAPACITY')
 return {'status':'PASS_QUERY_ONLY_PRODUCTION_PREFLIGHT_NO_MODEL','binding_sha256':sha(binding_path),'execution_source_commit':b['execution_source_commit'],'schedule':{'sha256':sha(E3/'TREATMENT_SCHEDULE.json'),'slots':len(rows),'updates':96,'raw_assistant_tokens':27360},'query_only_mask':{'count_slots':48,'selected_tokens_per_count_slot':15,'audits':audit},'new_ood':{'canonical_units':48,'records':96,'zero_overlap':cohort['zero_overlap'],'topology_count_independence':cohort['topology_count_independence']},'records':{'V7':len(records()),'REFERENCE':len(records('REFERENCE')),'QUERY_ONLY':len(records('QUERY_ONLY'))},'model_loaded':False,'gpu_used':False,'sidecar_opened':False,'physical_batch_size':1}
def load_model(binding,tag,trainable=False):
 import torch
 from peft import PeftModel
 from transformers import AutoModelForCausalLM
 if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():fail('CUDA_BF16')
 m=load(V7);base=AutoModelForCausalLM.from_pretrained(m['base_path'],local_files_only=True,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True).to('cuda:0')
 path=m['adapter_path'] if tag in {'V7','QUERY_ONLY'} else binding['reference_equal_slot_adapter']['files'][0]['path'].rsplit('/',1)[0]
 model=PeftModel.from_pretrained(base,path,is_trainable=trainable)
 if trainable:model.enable_input_require_grads();model.gradient_checkpointing_enable()
 else:model.eval()
 return model,torch,m
def train_query_only(binding,out,deadline):
 from training_data.pipeline import task_to_sample
 from scripts.e04_e_equal_slot_loss import backward_equal_slot_term
 import bitsandbytes as bnb
 model,torch,m=load_model(binding,'QUERY_ONLY',trainable=True);torch.manual_seed(binding['training']['torch_seed'])
 opt=bnb.optim.PagedAdamW8bit([p for p in model.parameters() if p.requires_grad],lr=binding['training']['learning_rate'])
 counts=[];steps=0
 for offset in range(0,384,4):
  if time.monotonic()>deadline:fail('TRAIN_CAP')
  opt.zero_grad(set_to_none=True);total=0.0
  for item in schedule()[offset:offset+4]:
   sample=task_to_sample(serializer_task(item));labs=sample['labels'];selected=sample['assistant_token_count']
   if item['role']=='PRIMARY_CONNECTED_COMPONENT_COUNT':
    _,labs,a=query_only_labels(serializer_task(item));selected=a['selected_count']
   ids=torch.tensor([sample['input_ids']],device='cuda:0');labels=torch.tensor([labs],device='cuda:0');o=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False)
   term,n=backward_equal_slot_term(o.logits,labels,expected_slots=4);total+=float(term);counts.append({'slot':item['slot'],'role':item['role'],'selected_tokens':n,'slot_coefficient':0.25});del o,term
  if not torch.isfinite(torch.tensor(total)):fail('NONFINITE_LOSS')
  opt.step();steps+=1
 if steps!=96 or len(counts)!=384 or sum(x['role']=='PRIMARY_CONNECTED_COMPONENT_COUNT' and x['selected_tokens']==15 for x in counts)!=48:fail('TRAIN_ACCOUNTING')
 adapter=Path(out)/'QUERY_ONLY_ADAPTER';model.save_pretrained(adapter)
 write(Path(out)/'QUERY_ONLY_TRAINING_RECEIPT.json',{'status':'COMPLETE_QUERY_ONLY_TRAINING','optimizer_steps':steps,'slot_count':len(counts),'selected_token_counts':{'count_primary':720,'object_primary':2880,'protected_replay':21600},'slot_coefficient':0.25,'adapter_sha256':sha(adapter/'adapter_model.safetensors'),'per_slot':counts})
 return model,torch,m
def generate(model,torch,manifest,items,binding,out,tag,deadline):
 from scripts import run_connected_components_object_extraction_pilot_v1 as real
 translated={'runtime':{'context_window':binding['runtime']['context_window']},'evaluation':{'fallback_ladder':[1],'max_new_tokens':binding['runtime']['max_new_tokens']},'training':{'torch_seed':binding['evaluation']['bootstrap_seed']}}
 rows=[{'record_id':x['id'],'split':x['split'],'task':x['task']} for x in items]
 return real._generate(model,torch,manifest,rows,translated,out,deadline,tag,force_batch_size=1)
def target_for_record(row,bases):
 from scripts.connected_components_object_extraction_semantic_oracle import target_for_family
 from scripts import e04_orientation_fixed_demonstration_baseline_v3 as orient
 if row['split'] in {'OOD_INDEPENDENT','TRAIN_FIT_SECONDARY'}:return target_for_family('CONNECTED_COMPONENT_COUNT',row['task']['test'][0]['input'])
 if row['split']=='COMPOSITION_SECONDARY':return target_for_family('COUNT_THEN_OBJECT_EXTRACTION',row['task']['test'][0]['input'])
 q=row['protected'];return orient.target(bases[q['canonical_base_id']],q['control_marker_turn'],q['condition'],q['source_row_index'])['target']
def make_sidecar(binding_path,out):
 fr=load(Path(out)/'RAW_FREEZE_RECEIPT.json')
 if fr.get('status')!='COMPLETE_RAW_FROZEN_NO_TARGETS' or fr.get('targets_accessed') is not False:fail('RAW_FREEZE_REQUIRED')
 from scripts import e04_orientation_fixed_demonstration_baseline_v3 as orient
 bases={x['canonical_base_id']:x for x in orient.v2.canonical_bases('VALIDATION')}
 targets={x['id']:target_for_record(x,bases) for x in records('REFERENCE')+records('QUERY_ONLY')}
 commitments={x['record_id']:x['target_sha256'] for x in validate_cohort()['target_commitments']}
 commitments.update({x['record_id']:x['target_sha256'] for x in load(E3/'TRAIN_FIT_INPUT_PROMPTS.json')['target_commitments'] if any(y['record_id']==x['record_id'] and y['arm']=='TREATMENT_DECONFOUNDED' for y in load(E3/'TRAIN_FIT_INPUT_PROMPTS.json')['records'])})
 for k,v in commitments.items():
  if k not in targets or hashlib.sha256(canon(targets[k])).hexdigest()!=v:fail('TARGET_COMMITMENT:'+k)
 p=Path(out)/'SYNTHETIC_TARGET_SIDECAR.json';write(p,{'status':'DERIVED_AFTER_RAW_FREEZE','targets':targets,'commitments_verified':len(commitments),'opened_after_raw_freeze':True});return p
def scored_rows(out,freeze,targets,tag,arm):
 want={x['id']:x for x in records(arm)};p=Path(out)/(tag+'_RAW.jsonl')
 if not p.is_file() or sha(p)!=freeze['raw_sha256'].get(tag):fail('RAW_IDENTITY:'+tag)
 vals=[json.loads(x) for x in p.read_text().splitlines() if x];ids=[x.get('record_id') for x in vals]
 if len(ids)!=len(set(ids)) or set(ids)!=set(want):fail('RAW_COMPLETENESS:'+tag)
 got={}
 for x in vals:
  w=want[x['record_id']]
  if x.get('prompt_sha256')!=task_hash(w['task']) or x.get('effective_batch_size')!=1:fail('RAW_BATCH1:'+tag+':'+x['record_id'])
  got[x['record_id']]={**x,'exact':bool(x.get('parser_valid')) and x.get('parsed_grid')==targets[x['record_id']]}
 return got
def bootstrap(diff,seed,n):
 rng=random.Random(seed);draw=sorted(sum(rng.choice(diff) for _ in diff) for __ in range(n));return draw[int(.025*n)-1],draw[int(.975*n)-1]
def score(binding_path,out,sidecar):
 b=load(binding_path);fr=load(Path(out)/'RAW_FREEZE_RECEIPT.json');targets=load(sidecar)['targets']
 if fr.get('status')!='COMPLETE_RAW_FROZEN_NO_TARGETS' or fr.get('targets_accessed') is not False:fail('FREEZE')
 v=scored_rows(out,fr,targets,'V7',None);e=scored_rows(out,fr,targets,'EQUAL_SLOT_REFERENCE','REFERENCE');q=scored_rows(out,fr,targets,'QUERY_ONLY','QUERY_ONLY')
 ood=[x for x in records() if x['split']=='OOD_INDEPENDENT'];units=sorted({x['canonical'] for x in ood})
 def both(rows,u):return int(all(rows[x['id']]['exact'] for x in ood if x['canonical']==u))
 diff=[both(q,u)-both(e,u) for u in units];delta=sum(diff);lo,hi=bootstrap(diff,b['evaluation']['bootstrap_seed'],b['evaluation']['bootstrap_replicates'])
 fitids={x['id'] for x in records('QUERY_ONLY') if x['split']=='TRAIN_FIT_SECONDARY'};qfit=sum(q[x]['exact'] for x in fitids);efit=sum(e[x]['exact'] for x in fitids)
 anyq=[u for u in units if any(q[x['id']]['exact'] for x in ood if x['canonical']==u)];bothq=sum(both(q,u) for u in anyq)
 allr=records();ret={}
 for fam in sorted({x['protected']['condition'] for x in allr if x['split']=='RETENTION_SECONDARY'}):
  ids=[x['id'] for x in allr if x['split']=='RETENTION_SECONDARY' and x['protected']['condition']==fam];ret[fam]={'query_only_minus_equal_slot':sum(q[x]['exact']-e[x]['exact'] for x in ids),'rows':len(ids)}
 compids=[x['id'] for x in allr if x['split']=='COMPOSITION_SECONDARY'];comp={'equal_slot':sum(e[x]['exact'] for x in compids),'query_only':sum(q[x]['exact'] for x in compids),'query_only_minus_equal_slot':sum(q[x]['exact']-e[x]['exact'] for x in compids),'rows':len(compids)}
 aggregate=sum(x['query_only_minus_equal_slot'] for x in ret.values())
 gates={'train_fit':qfit>=43 and qfit-efit>=8,'primary_new_ood':delta>=6 and lo>0,'order_invariance':bool(anyq) and bothq/len(anyq)>=.8,'retention':all(x['query_only_minus_equal_slot']>=-1 for x in ret.values()) and aggregate>=-2,'composition':comp['query_only_minus_equal_slot']>=-1}
 decision='PASS_REQUIRES_LATER_INDEPENDENT_CONFIRMATION' if all(gates.values()) else 'REJECT'
 receipt={'status':'COMPLETE_QUERY_ONLY_SCORE_AFTER_RAW_FREEZE','decision':decision,'promotion':'FORBIDDEN_REQUIRES_LATER_INDEPENDENT_CONFIRMATION','primary_new_independent_ood':{'canonical_units':48,'query_only_minus_equal_slot':{'delta_correct':delta,'bootstrap_95_ci':[lo,hi],'replicates':b['evaluation']['bootstrap_replicates']},'gate':gates['primary_new_ood']},'train_fit':{'query_only':qfit,'equal_slot':efit,'delta':qfit-efit,'gate':gates['train_fit']},'order_invariance':{'query_only_both_orders_correct':bothq,'query_only_any_order_correct':len(anyq),'ratio':0 if not anyq else bothq/len(anyq),'gate':gates['order_invariance']},'secondary_descriptive':{'retention':{'by_family':ret,'aggregate_query_only_minus_equal_slot':aggregate},'composition':comp,'v7_records_scored':len(v)},'gates':gates,'physical_batch_size':1,'batch32_used':False,'targets_opened_after_raw_freeze':True}
 write(Path(out)/'SCORE_RECEIPT.json',receipt);return receipt
def execute(binding_path):
 pre=preclaim(binding_path);b=load(binding_path);out=Path(b['output_root']);out.mkdir(parents=True);deadline=time.monotonic()+b['hard_cap_seconds'];write(out/'PREFLIGHT_RECEIPT.json',pre)
 try:
  write(out/'SCIENTIFIC_BOUNDARY.json',{'status':'FIRST_MODEL_LOAD_NEXT','nonce':b['nonce'],'retry':False})
  raw={}
  v,torch,m=load_model(b,'V7');raw['V7']=generate(v,torch,m,records(),b,out,'V7',deadline);del v;torch.cuda.empty_cache()
  e,torch,m=load_model(b,'EQUAL_SLOT_REFERENCE');raw['EQUAL_SLOT_REFERENCE']=generate(e,torch,m,records('REFERENCE'),b,out,'EQUAL_SLOT_REFERENCE',deadline);del e;torch.cuda.empty_cache()
  q,torch,m=train_query_only(b,out,deadline);raw['QUERY_ONLY']=generate(q,torch,m,records('QUERY_ONLY'),b,out,'QUERY_ONLY',deadline);del q;torch.cuda.empty_cache()
  write(out/'RAW_FREEZE_RECEIPT.json',{'status':'COMPLETE_RAW_FROZEN_NO_TARGETS','raw_sha256':raw,'targets_accessed':False,'nonce':b['nonce'],'physical_batch_size':1})
  sidecar=make_sidecar(binding_path,out);score(binding_path,out,sidecar);write(out/'TERMINAL_RECEIPT.json',{'status':'COMPLETE','raw_freeze_before_sidecar':True,'score_sha256':sha(out/'SCORE_RECEIPT.json')})
 except Exception as e:
  write(out/'TERMINAL_RECEIPT.json',{'status':'FAILED_OR_PARTIAL_NO_RETRY','error':type(e).__name__+':'+str(e),'raw_predictions_created':(out/'RAW_FREEZE_RECEIPT.json').exists(),'scientific_boundary_entered':(out/'SCIENTIFIC_BOUNDARY.json').exists()});raise
def main():
 a=argparse.ArgumentParser();a.add_argument('--binding',type=Path,required=True);a.add_argument('--output-root');a.add_argument('--make-binding',action='store_true');a.add_argument('--preclaim-only',action='store_true');a.add_argument('--execute',action='store_true');z=a.parse_args()
 if z.make_binding:print(json.dumps(make_binding(z.binding,z.output_root),sort_keys=True))
 elif z.preclaim_only:print(json.dumps(preclaim(z.binding),sort_keys=True))
 elif z.execute:execute(z.binding)
 else:raise SystemExit('select mode')
if __name__=='__main__':main()
