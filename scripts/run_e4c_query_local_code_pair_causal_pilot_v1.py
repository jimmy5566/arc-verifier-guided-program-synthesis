"""Production worker for E4C_QUERY_LOCAL_CODE_PAIR_CAUSAL_PILOT_V1.

The only changed loss path is mass-preserving corrected Code-Pair weighting on the 48 Count
primary slots at zero-based output cells (1,1)/(1,2). Every other slot is passed
byte-for-byte to the established Equal-Slot implementation.  Targets are derived only after raw predictions
are frozen.
"""
from __future__ import annotations
import argparse,copy,hashlib,importlib,json,os,random,subprocess,sys,time
from pathlib import Path
ROOT=Path(os.environ['ARC2_ROOT']) if 'ARC2_ROOT' in os.environ else Path(__file__).resolve().parents[1]
for p in (ROOT,ROOT/'src'):
 if str(p) not in sys.path:sys.path.insert(0,str(p))
BASE=ROOT/'experiments/capability_repair_baseline_v1/connected_components_object_extraction_capability_pilot_v1'
E3=BASE/'deconfounded_count_curriculum_causal_pilot_v1'
PKG=E3/'e4c_qlcp_v1'
ID='E4C_QUERY_LOCAL_CODE_PAIR_CAUSAL_PILOT_V1'
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
def fail(x):raise RuntimeError('E4C:'+x)
def task_hash(x):return hashlib.sha256(canon(x)).hexdigest()
def clean_source():
 # Exact execution source requires tracked files clean; unrelated preserved evidence may remain untracked.
 for args in (['git','diff','--quiet'],['git','diff','--cached','--quiet']):
  if subprocess.run(args,cwd=ROOT).returncode: fail('TRACKED_SOURCE_DIRTY')
def serializer_task(item):
 t=copy.deepcopy(item['task']);t.setdefault('source_id',f"SERIALIZER_VIEW:{item['canonical_base_id']}");return t
def inputs():
 return {
  'protocol':PKG/'PROTOCOL.json','freeze_receipt':PKG/'PROSPECTIVE_FREEZE_RECEIPT.json',
  'e3_protocol':E3/'PROTOCOL.json','e3_treatment_schedule':E3/'TREATMENT_SCHEDULE.json',
  'e3_train_prompts':E3/'TRAIN_FIT_INPUT_PROMPTS.json',
  'new_ood':PKG/'OOD_INPUTS_AND_COMMITMENTS.json','new_marker':PKG/'MARKER_INPUTS_AND_COMMITMENTS.json',
  'composition':BASE/'scientific_successor_v2/COMPOSITION_SENTINEL_INPUT_PROMPTS.json',
  'v7_manifest':V7,
 }
def file_entry(p):
 p=Path(p);return {'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)}
def make_binding(path,output_root):
 clean_source();commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
 manifest=load(V7)
 b={'schema_version':1,'experiment_id':ID,'execution_source_commit':commit,
 'inputs':{k:{'path':v.relative_to(ROOT).as_posix(),'sha256':sha(v)} for k,v in inputs().items()},
 'worker_path':'scripts/run_e4c_query_local_code_pair_causal_pilot_v1.py','worker_sha256':sha(Path(__file__)),'scorer_path':'scripts/run_e4c_query_local_code_pair_causal_pilot_v1.py','scorer_sha256':sha(Path(__file__)),
 'output_root':str(output_root),'runtime_python':RUNTIME,'hard_cap_seconds':7200,'retry':False,
 'nonce':hashlib.sha256((commit+str(output_root)+str(time.time_ns())).encode()).hexdigest()[:32],
 'command':['bash','scripts/launch_e4c_query_local_code_pair_causal_pilot_v1.sh','--binding',path.resolve().relative_to(ROOT).as_posix()],
 'runtime':{'context_window':16384,'max_new_tokens':192,'physical_batch_size':1,'cuda_required':True},
 'v7':{'manifest_path':V7.relative_to(ROOT).as_posix(),'manifest_sha256':sha(V7),'adapter_path':manifest['adapter_path'],'adapter_files':manifest['adapter_files'],'base_path':manifest['base_path'],'base_files':manifest['base_files']},
 'training':{'schedule_path':(E3/'TREATMENT_SCHEDULE.json').relative_to(ROOT).as_posix(),'schedule_sha256':sha(E3/'TREATMENT_SCHEDULE.json'),'updates':96,'slots_per_update':4,'torch_seed':20261013,'learning_rate':5e-5,'optimizer':'PagedAdamW8bit','precision':'BF16','slot_coefficient':0.25,'loss':'QUERY_LOCAL_CORRECTED_CODE_PAIR_MASS_PRESERVING'},
 'evaluation':{'bootstrap_seed':20261013,'bootstrap_replicates':10000,'physical_batch_size':1,'fallback_ladder':[1]},
 'invariants':{'exact_e3_treatment_schedule':True,'fresh_both_arms_required':True,'only_count_primary_query_local_weighted':True,'corrected_cells_zero_based':[[1,1],[1,2]],'final_query_only':True,'multiplier':2.0,'raw_freeze_before_sidecar':True,'target_blind_inference':True,'duplicate_output_guard':True,'no_batch32_or_fallback':True,'marker_binding_every_endpoint_delta_min':0,'marker_binding_aggregate_delta_min':0}}
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
def code_pair_audit():
 """Use real task tokenization and select only final-query corrected code tokens."""
 from training_data.pipeline import task_to_sample
 from scripts.e4b_corrected_code_pair_supervision import count_code_pair_label_indices
 from scripts.analyze_e4c_cpu_only_mechanism_study_v1 import query_local_code_pair_indices,query_local_code_pair_weights
 out=[]
 for item in schedule():
  if item['role']!='PRIMARY_CONNECTED_COMPONENT_COUNT':continue
  task=serializer_task(item);sample=task_to_sample(task);labels=sample['labels'];allidx,counts=count_code_pair_label_indices(task);idx=query_local_code_pair_indices(task)
  if len(allidx)!=8 or len(idx)!=2 or tuple(idx)!=tuple(allidx[-2:]) or any(labels[i]==IGNORE for i in idx):fail('QUERY_LOCAL_TOKEN_ORACLE')
  query=task['test'][0]['output'];count=counts[-1];color=next((z for row in query for z in row if z),0); expected=(0,0) if count==1 else ((color,0) if count==2 else (color,color))
  if tuple(query[1][1:3])!=expected:fail('QUERY_LOCAL_SEMANTIC_CODE')
  control,weights=query_local_code_pair_weights(labels,task,2.0)
  if abs(sum(control.values())-sum(weights.values()))>1e-12:fail('QUERY_LOCAL_MASS')
  if any(weights[i]!=2.0 for i in idx) or any(weights[i]==2.0 for i in allidx[:-2]):fail('QUERY_LOCAL_DEMO_WEIGHT')
  out.append({'slot':item['slot'],'final_query_indices':list(idx),'demo_indices':list(allidx[:-2]),'query_count':count,'query_semantic_code':list(expected),'selected_label_token_ids':[labels[i] for i in idx],'assistant_token_count':sample['assistant_token_count'],'control_mass':sum(control.values()),'treatment_mass':sum(weights.values())})
 if len(out)!=48 or {x['query_count'] for x in out}!={1,2,3}:fail('QUERY_LOCAL_GRID')
 return out
def validate_cohort():
 c=load(PKG/'OOD_INPUTS_AND_COMMITMENTS.json');rows=c.get('records',[]);m=load(PKG/'MARKER_INPUTS_AND_COMMITMENTS.json');mr=m.get('records',[])
 if c.get('status')!='FROZEN_TARGET_BLIND_INPUTS_AND_COMMITMENTS' or len(rows)!=96 or len({x['canonical_unit_id'] for x in rows})!=48:fail('OOD_GRID')
 if c.get('topology_count_mutual_information_bits')!=0.0:fail('OOD_INDEPENDENCE')
 if m.get('status')!='FROZEN_TARGET_BLIND_INPUTS_AND_COMMITMENTS' or len(mr)!=16 or len({x['canonical_unit_id'] for x in mr})!=8:fail('MARKER_GRID')
 seen={}
 for x in rows:
  if 'output' in x['prompt']['test'][0] or x.get('prompt_sha256')!=task_hash(x['prompt']):fail('OOD_TARGET_OR_HASH')
  seen.setdefault(x['canonical_unit_id'],set()).add(x['demo_order'])
 if any(v!={'A','B'} for v in seen.values()):fail('OOD_ORDERS')
 for x in mr:
  if 'output' in x['prompt']['test'][0] or x.get('prompt_sha256')!=task_hash(x['prompt']):fail('MARKER_TARGET_OR_HASH')
 return {'ood':c,'marker':m}
def records(arm=None):
 out=[];panels=validate_cohort()
 for x in panels['ood']['records']:out.append({'id':x['record_id'],'split':'OOD_INDEPENDENT','canonical':x['canonical_unit_id'],'order':x['demo_order'],'query_count':x['query_count'],'task':x['prompt']})
 for x in panels['marker']['records']:out.append({'id':x['record_id'],'split':'MARKER_BINDING_PRESERVATION','canonical':x['canonical_unit_id'],'endpoint':str(x['marker_turn']),'task':x['prompt']})
 for x in load(inputs()['composition']):out.append({'id':'COMPOSITION:'+x['prompt_id'],'split':'COMPOSITION_SECONDARY','task':x['task']})
 if arm:
  for x in load(E3/'TRAIN_FIT_INPUT_PROMPTS.json')['records']:
   if x['arm']=='TREATMENT_DECONFOUNDED':out.append({'id':x['record_id'],'split':'TRAIN_FIT_SECONDARY','task':x['prompt']})
 if any('output' in x['task']['test'][0] for x in out):fail('TARGET_LEAK')
 return out
def preclaim(binding_path,allow_missing_runtime=False):
 b=load(binding_path)
 if b.get('experiment_id')!=ID or b.get('retry') is not False or b.get('worker_sha256')!=sha(Path(__file__)) or b.get('scorer_sha256')!=sha(Path(__file__)):fail('BINDING')
 if Path(b['output_root']).exists():fail('OUTPUT_EXISTS')
 if b.get('hard_cap_seconds')!=7200 or b['runtime'].get('physical_batch_size')!=1 or b['evaluation'].get('fallback_ladder')!=[1]:fail('EXECUTION_CONTRACT')
 clean_source()
 for k,v in b['inputs'].items():
  p=ROOT/v['path']
  if not p.is_file() or sha(p)!=v['sha256']:fail('INPUT:'+k)
 if b['training'].get('schedule_sha256')!=sha(E3/'TREATMENT_SCHEDULE.json'):fail('SCHEDULE_IDENTITY')
 rows=schedule();audit=code_pair_audit();panels=validate_cohort()
 for name in ('inference.nvarc_native','training_data.pipeline','scripts.e04_e_equal_slot_loss','scripts.connected_components_object_extraction_semantic_oracle','scripts.e4b_corrected_code_pair_supervision','scripts.analyze_e4c_cpu_only_mechanism_study_v1'):importlib.import_module(name)
 manifest=load(V7)
 if sha(V7)!=b['v7']['manifest_sha256']:fail('V7_MANIFEST')
 assets=[]
 for root,files in ((manifest['adapter_path'],manifest['adapter_files']),(manifest['base_path'],manifest['base_files'])):
  for f in files: assets.append({'root':root,'name':f['name'],'bytes':f['bytes'],'sha256':f['sha256']})
 if not allow_missing_runtime:
  if not Path(b['runtime_python']).is_file():fail('RUNTIME_PYTHON')
  for f in assets:
   p=Path(f['root'])/f['name']
   if not p.is_file() or p.stat().st_size!=f['bytes'] or sha(p)!=f['sha256']:fail('V7_ASSET:'+f['name'])
  gpu=subprocess.run(['nvidia-smi','--query-gpu=memory.total','--format=csv,noheader'],capture_output=True,text=True)
  if gpu.returncode or int(gpu.stdout.split()[0])<24576:fail('GPU_CAPACITY')
 return {'status':'PASS_E4C_PRODUCTION_PREFLIGHT_NO_MODEL','binding_sha256':sha(binding_path),'execution_source_commit':b['execution_source_commit'],'schedule':{'sha256':sha(E3/'TREATMENT_SCHEDULE.json'),'slots':len(rows),'updates':96,'raw_assistant_tokens':27360},'query_local_code_pair':{'count_slots':48,'cells_zero_based':[[1,1],[1,2]],'tokens_per_slot':2,'multiplier':2.0,'audits':audit},'new_ood':{'canonical_units':48,'records':96,'topology_count_mi_bits':0.0},'new_marker_binding':{'canonical_units':8,'records':16,'every_endpoint_delta_min':0,'aggregate_delta_min':0},'assets':assets,'records':{'V7':len(records()),'EQUAL_SLOT':len(records('EQUAL_SLOT')),'QUERY_LOCAL_CODE_PAIR':len(records('QUERY_LOCAL_CODE_PAIR'))},'model_loaded':False,'gpu_used':False,'sidecar_opened':False,'physical_batch_size':1}
def load_model(binding,tag,trainable=False):
 import torch
 from peft import PeftModel
 from transformers import AutoModelForCausalLM
 if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():fail('CUDA_BF16')
 m=load(V7);base=AutoModelForCausalLM.from_pretrained(m['base_path'],local_files_only=True,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True).to('cuda:0')
 path=m['adapter_path']
 model=PeftModel.from_pretrained(base,path,is_trainable=trainable)
 if trainable:model.enable_input_require_grads();model.gradient_checkpointing_enable()
 else:model.eval()
 return model,torch,m
def train_code_pair(binding,out,deadline):
 from training_data.pipeline import task_to_sample
 from scripts.e04_e_equal_slot_loss import backward_equal_slot_term,backward_weighted_slot_term
 from scripts.analyze_e4c_cpu_only_mechanism_study_v1 import query_local_code_pair_indices,query_local_code_pair_weights
 import bitsandbytes as bnb
 model,torch,m=load_model(binding,'QUERY_LOCAL_CODE_PAIR',trainable=True);torch.manual_seed(binding['training']['torch_seed']);opt=bnb.optim.PagedAdamW8bit([p for p in model.parameters() if p.requires_grad],lr=binding['training']['learning_rate'])
 counts=[];steps=0
 for offset in range(0,384,4):
  if time.monotonic()>deadline:fail('TRAIN_CAP')
  opt.zero_grad(set_to_none=True);total=0.0
  for item in schedule()[offset:offset+4]:
   sample=task_to_sample(serializer_task(item));labs=sample['labels'];selected=sample['assistant_token_count'];ids=torch.tensor([sample['input_ids']],device='cuda:0');labels=torch.tensor([labs],device='cuda:0');o=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False)
   if item['role']=='PRIMARY_CONNECTED_COMPONENT_COUNT':
    indices=query_local_code_pair_indices(serializer_task(item));_,weights=query_local_code_pair_weights(labs,serializer_task(item),2.0);term,n,_=backward_weighted_slot_term(o.logits,labels,[weights.get(i,0.0) for i in range(len(labs))],expected_slots=4);selected=len(indices)
   else:term,n=backward_equal_slot_term(o.logits,labels,expected_slots=4)
   total+=float(term);counts.append({'slot':item['slot'],'role':item['role'],'selected_tokens':selected,'slot_coefficient':0.25});del o,term
  if not torch.isfinite(torch.tensor(total)):fail('NONFINITE_LOSS')
  opt.step();steps+=1
 if steps!=96 or len(counts)!=384 or sum(x['role']=='PRIMARY_CONNECTED_COMPONENT_COUNT' and x['selected_tokens']==2 for x in counts)!=48:fail('TRAIN_ACCOUNTING')
 adapter=Path(out)/'QUERY_LOCAL_CODE_PAIR_ADAPTER';model.save_pretrained(adapter);write(Path(out)/'QUERY_LOCAL_CODE_PAIR_TRAINING_RECEIPT.json',{'status':'COMPLETE_E4C_QUERY_LOCAL_CODE_PAIR_TRAINING','optimizer':'PagedAdamW8bit','learning_rate':binding['training']['learning_rate'],'optimizer_steps':steps,'slot_count':len(counts),'selected_token_counts':{'count_primary':96,'object_primary':'ORIGINAL_EQUAL_SLOT','protected_replay':'ORIGINAL_EQUAL_SLOT'},'slot_coefficient':0.25,'adapter_sha256':sha(adapter/'adapter_model.safetensors'),'per_slot':counts});return model,torch,m
def messages(task):
 from inference.nvarc_native import serialize_grid
 out=[]
 for pair in task['train']:
  out += [{'role':'user','content':serialize_grid(pair['input'])},{'role':'assistant','content':serialize_grid(pair['output'])}]
 out.append({'role':'user','content':serialize_grid(task['test'][0]['input'])})
 return out
def train_equal_slot(binding,out,deadline):
 from training_data.pipeline import task_to_sample
 from scripts.e04_e_equal_slot_loss import backward_equal_slot_term
 import bitsandbytes as bnb
 model,torch,m=load_model(binding,'QUERY_LOCAL_CODE_PAIR',trainable=True);torch.manual_seed(binding['training']['torch_seed'])
 opt=bnb.optim.PagedAdamW8bit([p for p in model.parameters() if p.requires_grad],lr=binding['training']['learning_rate'])
 counts=[]
 for offset in range(0,384,4):
  if time.monotonic()>deadline:fail('TRAIN_CAP')
  opt.zero_grad(set_to_none=True)
  for item in schedule()[offset:offset+4]:
   sample=task_to_sample(serializer_task(item));ids=torch.tensor([sample['input_ids']],device='cuda:0');labels=torch.tensor([sample['labels']],device='cuda:0')
   o=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False);term,n=backward_equal_slot_term(o.logits,labels,expected_slots=4);counts.append({'slot':item['slot'],'role':item['role'],'selected_tokens':n,'slot_coefficient':0.25});del o,term
  opt.step()
 if len(counts)!=384 or any(x['selected_tokens']<=0 for x in counts):fail('EQUAL_SLOT_TRAIN_ACCOUNTING')
 adapter=Path(out)/'EQUAL_SLOT_5E5_ADAPTER';model.save_pretrained(adapter)
 write(Path(out)/'EQUAL_SLOT_5E5_TRAINING_RECEIPT.json',{'status':'COMPLETE_E4C_EQUAL_SLOT_TRAINING','optimizer':'PagedAdamW8bit','learning_rate':binding['training']['learning_rate'],'optimizer_steps':96,'slot_count':384,'slot_coefficient':0.25,'adapter_sha256':sha(adapter/'adapter_model.safetensors'),'per_slot':counts})
 return model,torch,m

def generate(model,torch,manifest,items,binding,out,tag,deadline):
 from inference.nvarc_native import NVARCNativeProvider,checkpoint_native_tokenizer,parse_native_grid
 config=ROOT/'experiments/capability_repair_baseline_v1/e04_orientation_fixed_demonstration_baseline_v3/E04_V3_RUNTIME_NATIVE_CONFIG_IDENTITY_V1.json'
 c=load(config); config_dir=ROOT/'configs'/c['runtime_config_dir_name']
 tok,_=checkpoint_native_tokenizer(Path(manifest['base_path']),config_dir)
 provider=NVARCNativeProvider(model_path=Path(manifest['base_path']),tokenizer_config_dir=config_dir,device='cuda:0');provider.model,provider.tokenizer=model,tok
 ordered=sorted(items,key=lambda x:(len(canon(x['task'])),x['id']));raw=[]
 for i,item in enumerate(ordered):
  if time.monotonic()>deadline:fail('EVALUATION_CAP')
  got=provider.generate_many([messages(item['task'])],max_new_tokens=int(binding['runtime']['max_new_tokens']),context_window=int(binding['runtime']['context_window']),seeds=[int(binding['training']['torch_seed'])+i])
  x=got[0];grid=parse_native_grid(x.text)
  raw.append({'record_id':item['id'],'split':item['split'],'prompt_sha256':task_hash(item['task']),'generated_token_ids':[int(v) for v in x.token_ids],'parsed_grid':None if grid is None else [list(r) for r in grid],'parser_valid':grid is not None,'effective_batch_size':1})
 path=Path(out)/(tag+'_RAW.jsonl');write_jsonl(path,raw)
 write(Path(out)/(tag+'_GENERATION_RECEIPT.json'),{'requested_batch_size':1,'effective_batch_sizes':[1],'fallback_events':[],'raw_sha256':sha(path),'rows':len(raw),'target_accessed':False})
 return sha(path)

def target_for_record(row,bases=None):
 from scripts.connected_components_object_extraction_semantic_oracle import target_for_family
 if row['split'] in {'OOD_INDEPENDENT','TRAIN_FIT_SECONDARY'}:return target_for_family('CONNECTED_COMPONENT_COUNT',row['task']['test'][0]['input'])
 if row['split']=='COMPOSITION_SECONDARY':return target_for_family('COUNT_THEN_OBJECT_EXTRACTION',row['task']['test'][0]['input'])
 from scripts import freeze_e4c_query_local_code_pair_prospective_v1 as freezer
 # Deterministic post-freeze oracle; the target is absent from inference inputs.
 for i,base0 in enumerate(__import__('scripts.e04_orientation_marker_counterfactual_v2',fromlist=['canonical_bases']).canonical_bases('VALIDATION')[:8]):
  base=freezer.marker_base(base0,i)
  for turn in (0,2):
   if row['id']==f'E4C_NEW_MARKER_BINDING:{base["canonical_base_id"]}:T{turn}':return freezer.marker_oracle(base,turn)
 fail('MARKER_TARGET_RECORD')
def make_sidecar(binding_path,out):
 fr=load(Path(out)/'RAW_FREEZE_RECEIPT.json')
 if fr.get('status')!='COMPLETE_RAW_FROZEN_NO_TARGETS' or fr.get('targets_accessed') is not False:fail('RAW_FREEZE_REQUIRED')
 targets={x['id']:target_for_record(x) for x in records('EQUAL_SLOT')+records('QUERY_LOCAL_CODE_PAIR')}
 panels=validate_cohort();commitments={x['record_id']:x['target_sha256'] for x in panels['ood']['target_commitments']+panels['marker']['target_commitments']}
 train=load(E3/'TRAIN_FIT_INPUT_PROMPTS.json');commitments.update({x['record_id']:x['target_sha256'] for x in train['target_commitments'] if any(y['record_id']==x['record_id'] and y['arm']=='TREATMENT_DECONFOUNDED' for y in train['records'])})
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
 arms={'V7':scored_rows(out,fr,targets,'V7','V7'),'EQUAL_SLOT_5E5':scored_rows(out,fr,targets,'EQUAL_SLOT_5E5','EQUAL_SLOT'),'QUERY_LOCAL_CODE_PAIR_5E5':scored_rows(out,fr,targets,'QUERY_LOCAL_CODE_PAIR_5E5','QUERY_LOCAL_CODE_PAIR')}
 ood=[x for x in records() if x['split']=='OOD_INDEPENDENT'];units=sorted({x['canonical'] for x in ood});unit_count={u:next(x['query_count'] for x in ood if x['canonical']==u) for u in units}
 def both(rows,u):return int(all(rows[x['id']]['exact'] for x in ood if x['canonical']==u))
 def arm_summary(rows,arm):
  train=[x['id'] for x in records(arm) if x['split']=='TRAIN_FIT_SECONDARY'];vals={u:both(rows,u) for u in units};anyok={u:int(any(rows[x['id']]['exact'] for x in ood if x['canonical']==u)) for u in units};marker=[x['id'] for x in records(arm) if x['split']=='MARKER_BINDING_PRESERVATION'];comp=[x['id'] for x in records(arm) if x['split']=='COMPOSITION_SECONDARY']
  return {'train_fit_correct':sum(rows[x]['exact'] for x in train),'train_fit_total':len(train),'ood_both_orders_correct':sum(vals.values()),'ood_canonical_units':len(units),'ood_any_order_correct':sum(anyok.values()),'order_invariance_both_over_any':0 if not sum(anyok.values()) else sum(vals.values())/sum(anyok.values()),'ood_by_count':{str(c):{'canonical_units':sum(unit_count[u]==c for u in units),'both_orders_correct':sum(vals[u] for u in units if unit_count[u]==c)} for c in (1,2,3)},'marker_binding_correct':sum(rows[x]['exact'] for x in marker),'marker_binding_total':len(marker),'composition_correct':sum(rows[x]['exact'] for x in comp),'composition_total':len(comp)}
 sm={k:arm_summary(v,k) for k,v in arms.items()};diff=[both(arms['QUERY_LOCAL_CODE_PAIR_5E5'],u)-both(arms['EQUAL_SLOT_5E5'],u) for u in units];delta=sum(diff);lo,hi=bootstrap(diff,b['evaluation']['bootstrap_seed'],b['evaluation']['bootstrap_replicates']);allr=records();marker={}
 for endpoint in sorted({x['endpoint'] for x in allr if x['split']=='MARKER_BINDING_PRESERVATION'}):
  ids=[x['id'] for x in allr if x['split']=='MARKER_BINDING_PRESERVATION' and x['endpoint']==endpoint];marker[endpoint]={'equal_slot':sum(arms['EQUAL_SLOT_5E5'][x]['exact'] for x in ids),'query_local':sum(arms['QUERY_LOCAL_CODE_PAIR_5E5'][x]['exact'] for x in ids),'query_local_minus_equal_slot':sum(arms['QUERY_LOCAL_CODE_PAIR_5E5'][x]['exact']-arms['EQUAL_SLOT_5E5'][x]['exact'] for x in ids),'rows':len(ids)}
 compids=[x['id'] for x in allr if x['split']=='COMPOSITION_SECONDARY'];comp={'equal_slot':sum(arms['EQUAL_SLOT_5E5'][x]['exact'] for x in compids),'query_local':sum(arms['QUERY_LOCAL_CODE_PAIR_5E5'][x]['exact'] for x in compids),'query_local_minus_equal_slot':sum(arms['QUERY_LOCAL_CODE_PAIR_5E5'][x]['exact']-arms['EQUAL_SLOT_5E5'][x]['exact'] for x in compids),'rows':len(compids)};aggregate=sum(x['query_local_minus_equal_slot'] for x in marker.values());q=sm['QUERY_LOCAL_CODE_PAIR_5E5'];e=sm['EQUAL_SLOT_5E5']
 gates={'train_fit':q['train_fit_correct']>=43 and q['train_fit_correct']-e['train_fit_correct']>=8,'primary_new_ood':delta>=6 and lo>0,'order_invariance':q['order_invariance_both_over_any']>=.8,'marker_binding_preservation':all(x['query_local_minus_equal_slot']>=0 for x in marker.values()) and aggregate>=0,'composition':comp['query_local_minus_equal_slot']>=-1};decision='PASS_REQUIRES_LATER_INDEPENDENT_CONFIRMATION' if all(gates.values()) else 'REJECT'
 receipt={'status':'COMPLETE_E4C_SCORE_AFTER_RAW_FREEZE','decision':decision,'promotion':'FORBIDDEN_REQUIRES_LATER_INDEPENDENT_CONFIRMATION','arms':sm,'primary_new_independent_ood':{'canonical_units':48,'query_local_minus_equal_slot':{'delta_correct':delta,'bootstrap_95_ci':[lo,hi],'replicates':b['evaluation']['bootstrap_replicates']},'count_stratified_delta':{str(c):sum(both(arms['QUERY_LOCAL_CODE_PAIR_5E5'],u)-both(arms['EQUAL_SLOT_5E5'],u) for u in units if unit_count[u]==c) for c in (1,2,3)},'gate':gates['primary_new_ood']},'marker_binding_preservation':{'by_endpoint':marker,'aggregate_query_local_minus_equal_slot':aggregate,'gate':gates['marker_binding_preservation']},'composition_secondary':comp,'gates':gates,'physical_batch_size':1,'batch32_used':False,'targets_opened_after_raw_freeze':True};write(Path(out)/'SCORE_RECEIPT.json',receipt);return receipt
def execute(binding_path):
 pre=preclaim(binding_path);b=load(binding_path);out=Path(b['output_root']);out.mkdir(parents=True);deadline=time.monotonic()+b['hard_cap_seconds'];write(out/'PREFLIGHT_RECEIPT.json',pre)
 try:
  write(out/'SCIENTIFIC_BOUNDARY.json',{'status':'FIRST_MODEL_LOAD_NEXT','nonce':b['nonce'],'retry':False,'fresh_both_arms':True})
  raw={}
  v,torch,m=load_model(b,'V7');raw['V7']=generate(v,torch,m,records('V7'),b,out,'V7',deadline);del v;torch.cuda.empty_cache()
  e,torch,m=train_equal_slot(b,out,deadline);raw['EQUAL_SLOT_5E5']=generate(e,torch,m,records('EQUAL_SLOT'),b,out,'EQUAL_SLOT_5E5',deadline);del e;torch.cuda.empty_cache()
  q,torch,m=train_code_pair(b,out,deadline);raw['QUERY_LOCAL_CODE_PAIR_5E5']=generate(q,torch,m,records('QUERY_LOCAL_CODE_PAIR'),b,out,'QUERY_LOCAL_CODE_PAIR_5E5',deadline);del q;torch.cuda.empty_cache()
  write(out/'RAW_FREEZE_RECEIPT.json',{'status':'COMPLETE_RAW_FROZEN_NO_TARGETS','raw_sha256':raw,'targets_accessed':False,'nonce':b['nonce'],'physical_batch_size':1})
  sidecar=make_sidecar(binding_path,out);score(binding_path,out,sidecar);write(out/'TERMINAL_RECEIPT.json',{'status':'COMPLETE','raw_freeze_before_sidecar':True,'score_sha256':sha(out/'SCORE_RECEIPT.json')})
 except Exception as e:
  write(out/'TERMINAL_RECEIPT.json',{'status':'FAILED_OR_PARTIAL_NO_RETRY','error':type(e).__name__+':'+str(e),'raw_predictions_created':(out/'RAW_FREEZE_RECEIPT.json').exists(),'scientific_boundary_entered':(out/'SCIENTIFIC_BOUNDARY.json').exists()});raise

def main():
 a=argparse.ArgumentParser();a.add_argument('--binding',type=Path,required=True);a.add_argument('--output-root');a.add_argument('--make-binding',action='store_true');a.add_argument('--preclaim-only',action='store_true');a.add_argument('--execute',action='store_true');a.add_argument('--cpu-preflight',action='store_true');z=a.parse_args()
 if z.make_binding:print(json.dumps(make_binding(z.binding,z.output_root),sort_keys=True))
 elif z.preclaim_only:print(json.dumps(preclaim(z.binding),sort_keys=True))
 elif z.cpu_preflight:print(json.dumps(preclaim(z.binding,allow_missing_runtime=True),sort_keys=True))
 elif z.execute:execute(z.binding)
 else:raise SystemExit('select mode')
if __name__=='__main__':main()
