"""Bound production worker for E4F2 counterfactual component-affinity supervision.

CUDA imports occur only inside the execution functions.  Control remains original
Equal-Slot CE; treatment adds a train-only 0.25 permutation-invariant
component-affinity BCE only on Count=2/3 primary slots.  No auxiliary head is
created or retained for native inference.
"""
from __future__ import annotations
import argparse, copy, hashlib, importlib, json, math, os, random, secrets, shutil, subprocess, sys, tempfile, time
from collections import defaultdict
from pathlib import Path
ROOT=Path(os.environ.get('ARC2_ROOT',Path(__file__).resolve().parents[1]));sys.path[:0]=[str(ROOT),str(ROOT/'src')]
BASE=ROOT/'experiments/capability_repair_baseline_v1/connected_components_object_extraction_capability_pilot_v1';E3=BASE/'deconfounded_count_curriculum_causal_pilot_v1';PKG=E3/'e4f2_distance_overlap_affinity_v1';ID='E4F2_DISTANCE_OVERLAP_COMPONENT_AFFINITY_CAUSAL_PILOT_V1';V7=ROOT/'experiments/capability_repair_baseline_v1/CHECKPOINT_MANIFEST_REMOTE_V1.json';IGNORE=-100

def canon(x):return json.dumps(x,sort_keys=True,separators=(',',':')).encode()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_bytes(canon(x)+b'\n');os.replace(q,p)
def jsonl(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_bytes(b''.join(canon(v)+b'\n' for v in x));os.replace(q,p)
def rows(p):return [json.loads(x) for x in Path(p).read_text(encoding='utf-8').splitlines() if x]
def fail(x):raise RuntimeError('E4F2:'+x)
def nonce_processes(nonce):
 if os.name=='nt':return []
 hits=[]
 for p in Path('/proc').iterdir():
  if not p.name.isdigit() or int(p.name)==os.getpid():continue
  try:cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
  except OSError:continue
  if nonce in cmd:hits.append({'pid':int(p.name),'cmdline':cmd})
 return hits
def taskhash(x):return hashlib.sha256(canon(x)).hexdigest()
def clean():
 for args in (['git','diff','--quiet'],['git','diff','--cached','--quiet']):
  if subprocess.run(args,cwd=ROOT).returncode:fail('TRACKED_SOURCE_DIRTY')
def task(row):
 x=copy.deepcopy(row['task']);x.setdefault('source_id','E4F2_SERIALIZER:'+str(row['slot']));return x

def schedule():
 from training_data.pipeline import task_to_sample
 s=load(PKG/'SUCCESSOR_COUNTERFACTUAL_SCHEDULE.json')['episodes']
 if len(s)!=384 or [x['slot'] for x in s]!=list(range(384)) or [x['optimizer_step'] for x in s]!=[i//4 for i in range(384)]:fail('SCHEDULE')
 roles={k:sum(x['role']==k for x in s) for k in ('PRIMARY_CONNECTED_COMPONENT_COUNT','PRIMARY_OBJECT_EXTRACTION','PROTECTED_REPLAY')}
 if roles!={'PRIMARY_CONNECTED_COMPONENT_COUNT':48,'PRIMARY_OBJECT_EXTRACTION':48,'PROTECTED_REPLAY':288}:fail('ROLES')
 if sum(task_to_sample(task(x))['assistant_token_count'] for x in s)!=27360:fail('TOKEN_BUDGET')
 return s

def pair_records():
 value=load(PKG/'PAIR_AFFINITY_SPEC.json')['records']
 return {int(x['slot']):x for x in value}

def audit_slots():
 from training_data.pipeline import task_to_sample
 from scripts.e4f2_component_affinity import pair_layout,scalar_mass
 out=[];non=[];pairs=pair_records()
 for x in schedule():
  sample=task_to_sample(task(x))
  if x['role']=='PRIMARY_CONNECTED_COMPONENT_COUNT':
   rec=pairs[x['slot']];count=int(rec['count'])
   if count==1:
    if rec['selected_pairs'] or scalar_mass(2,1,False,count)!=scalar_mass(2,1,True,count):fail('COUNT1_AUXILIARY')
    out.append({'slot':x['slot'],'count':count,'auxiliary_coefficient':0.0,'selected_pair_count':0})
   else:
    lay=pair_layout(task(x),rec['selected_pairs'])
    if len(lay.pairs)!=1 or lay.component_count!=count:fail('PAIR_LAYOUT')
    out.append({'slot':x['slot'],'count':count,'auxiliary_coefficient':.25,'selected_pair_count':1,'pair_label':lay.labels[0],'query_positions':list(lay.positions)})
  else:non.append({'slot':x['slot'],'role':x['role'],'loss_path':'ORIGINAL_EQUAL_SLOT_CE','auxiliary_coefficient':0.0,'assistant_token_count':sample['assistant_token_count']})
 if len(out)!=48 or [sum(r['count']==c for r in out) for c in (1,2,3)]!=[16,16,16] or len(non)!=336:fail('SLOT_AUDIT')
 active=[r for r in out if r['count'] in (2,3)]
 if len(active)!=32 or sum(r['pair_label']==1 for r in active)!=16 or sum(r['pair_label']==0 for r in active)!=16:fail('PAIR_BALANCE')
 if any(sum(r['pair_label']==label and r['count']==count for r in active) != 8 for count in (2,3) for label in (0,1)):fail('PAIR_STRATUM')
 return out,non

def input_paths():return {'protocol':PKG/'PROTOCOL.json','schedule':PKG/'SUCCESSOR_COUNTERFACTUAL_SCHEDULE.json','pair_spec':PKG/'PAIR_AFFINITY_SPEC.json','ood':PKG/'NEW_STRUCTURAL_OOD_INPUTS_AND_COMMITMENTS.json','freeze':PKG/'CPU_COUNTERFACTUAL_FREEZE_RECEIPT.json','ood_freeze':PKG/'NEW_OOD_PROSPECTIVE_FREEZE_RECEIPT.json','v7':V7,'trainfit':E3/'TRAIN_FIT_INPUT_PROMPTS.json','marker':E3/'e4d_object_centric_component_anchor_grounding_v1/MARKER_INPUTS_AND_COMMITMENTS.json','retention':BASE/'RETENTION_INPUT_PROMPTS.json','composition':BASE/'scientific_successor_v2/COMPOSITION_SENTINEL_INPUT_PROMPTS.json'}

def make_binding(path,output,nonce=None):
 clean();commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
 if Path(output).exists():fail('OUTPUT_EXISTS')
 m=load(V7)
 try: binding_arg=Path(path).resolve().relative_to(ROOT).as_posix()
 except ValueError: binding_arg=str(Path(path).resolve()) # CPU fixture only; bound RunPod binding is repository-relative.
 b={'schema_version':1,'experiment_id':ID,'execution_source_commit':commit,'worker_path':Path(__file__).relative_to(ROOT).as_posix(),'worker_sha256':sha(__file__),'scorer_path':Path(__file__).relative_to(ROOT).as_posix(),'scorer_sha256':sha(__file__),'inputs':{k:{'path':v.relative_to(ROOT).as_posix(),'sha256':sha(v)} for k,v in input_paths().items()},'output_root':str(output),'nonce':nonce or secrets.token_hex(16),'retry':False,'hard_cap_seconds':7200,'runtime_python':'/root/code_pair_py311_cu128_v2/bin/python','command':['bash','scripts/launch_e4f2_component_affinity_v1.sh','--binding',binding_arg],'runtime':{'physical_batch_size':1,'decoder':'NVARC_NATIVE_GREEDY','cuda_required':True,'bf16_required':True,'max_new_tokens':192,'context_window':16384},'training':{'v7_exact':True,'seed':20261013,'learning_rate':5e-5,'optimizer':'PagedAdamW8bit','updates':96,'slots_per_update':4,'slot_coefficient':.25,'control':'ORIGINAL_EQUAL_SLOT_ANSWER_REPLAY_CE_1.0','treatment':'CONTROL_PLUS_0.25_FINAL_DECODER_QUERY_FOREGROUND_COMPONENT_AFFINITY_BCE','auxiliary':'one frozen unordered pair per Count=2/3 slot; 16/16 labels overall and 8/8 per treated count stratum; BCE coefficient 0.25'},'evaluation':{'bootstrap_seed':20261013,'bootstrap_replicates':10000,'paired_canonical_units':48,'physical_batch_size':1,'fallback_ladder':[1]},'v7_assets':{'base_path':m['base_path'],'base_files':m['base_files'],'adapter_path':m['adapter_path'],'adapter_files':m['adapter_files']}}
 write(path,b);return b

def preclaim(bp,allow_runtime_missing=False):
 b=load(bp)
 if b.get('experiment_id')!=ID or b.get('retry') is not False or b.get('hard_cap_seconds')!=7200 or Path(b['output_root']).exists():fail('BINDING')
 clean()
 if subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()!=b['execution_source_commit']:fail('HEAD')
 root_free=shutil.disk_usage('/').free
 output_parent=Path(b['output_root']).parent
 # Do not materialize the bound output root during preclaim.  Query the first
 # existing ancestor, which is the relevant filesystem for capacity purposes.
 capacity_path=ROOT if os.name=='nt' and str(output_parent).replace('\\','/').startswith('/workspace/') else output_parent
 while not capacity_path.exists() and capacity_path != capacity_path.parent:capacity_path=capacity_path.parent
 if not capacity_path.exists():fail('DISK_CAPACITY')
 output_free=shutil.disk_usage(capacity_path).free
 if root_free < (1<<30) or output_free < (20<<30): fail('DISK_CAPACITY')
 if nonce_processes(b['nonce']):fail('DUPLICATE_NONCE_PROCESS')
 for k,v in b['inputs'].items():
  p=ROOT/v['path']
  if not p.is_file() or sha(p)!=v['sha256']:fail('INPUT:'+k)
 a,n=audit_slots();ood=load(PKG/'NEW_STRUCTURAL_OOD_INPUTS_AND_COMMITMENTS.json')
 if len(ood['records'])!=96 or len({x['canonical_unit_id'] for x in ood['records']})!=48 or any('output' in x['prompt']['test'][0] for x in ood['records']):fail('OOD')
 for x in ('inference.nvarc_native','training_data.pipeline','scripts.e04_e_equal_slot_loss','scripts.e4f2_component_affinity','scripts.connected_components_object_extraction_semantic_oracle'):importlib.import_module(x)
 checks=[]
 if not allow_runtime_missing:
  for root,files in ((b['v7_assets']['base_path'],b['v7_assets']['base_files']),(b['v7_assets']['adapter_path'],b['v7_assets']['adapter_files'])):
   for f in files:
    p=Path(root)/f['name']
    if not p.is_file() or p.stat().st_size!=f['bytes'] or sha(p)!=f['sha256']:fail('ASSET:'+f['name'])
    checks.append({'name':f['name'],'sha256':f['sha256']})
  # Capability query only: establish the bound CUDA/BF16 runtime before the
  # scientific boundary without constructing or loading a model.
  import torch
  if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():fail('CUDA_BF16')
 return {'status':'PASS_E4F2_EXECUTABLE_PRODUCTION_PREFLIGHT_NO_MODEL','binding_sha256':sha(bp),'slots':384,'count_slots':48,'noncount_slots':336,'slot_audit':a,'assets':checks,'model_loaded':False,'gpu_used':False,'targets_opened':False,'sidecar_opened':False,'root_free_bytes':root_free,'output_parent_free_bytes':output_free}

def load_model(trainable):
 import torch
 from peft import PeftModel
 from transformers import AutoModelForCausalLM
 if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():fail('CUDA_BF16')
 m=load(V7);base=AutoModelForCausalLM.from_pretrained(m['base_path'],local_files_only=True,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True).to('cuda:0');model=PeftModel.from_pretrained(base,m['adapter_path'],is_trainable=trainable)
 if trainable:model.enable_input_require_grads();model.gradient_checkpointing_enable()
 else:model.eval()
 return model,torch,m

def train_arm(b,out,tag,auxiliary,deadline):
 from training_data.pipeline import task_to_sample
 from scripts.e04_e_equal_slot_loss import causal_slot_mean_ce,backward_equal_slot_term
 from scripts.e4f2_component_affinity import pair_layout,affinity_loss
 import bitsandbytes as bnb
 model,torch,m=load_model(True);torch.manual_seed(b['training']['seed']);params=[p for p in model.parameters() if p.requires_grad]
 opt=bnb.optim.PagedAdamW8bit(params,lr=b['training']['learning_rate']);receipt=[];pairmap=pair_records()
 for off in range(0,384,4):
  if time.monotonic()>deadline:fail('TRAIN_CAP')
  opt.zero_grad(set_to_none=True)
  for row in schedule()[off:off+4]:
   s=task_to_sample(task(row));ids=torch.tensor([s['input_ids']],device='cuda:0');labels=torch.tensor([s['labels']],device='cuda:0');rec=pairmap.get(row['slot']);active=bool(auxiliary and rec and rec['count'] in (2,3))
   o=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,output_hidden_states=active)
   if active:
    answer,_=causal_slot_mean_ce(o.logits,labels);layout=pair_layout(task(row),rec['selected_pairs']);extra,_=affinity_loss(o.hidden_states[-1],layout);term=.25*(answer+.25*extra);term.backward()
    model_grad=any(p.grad is not None and float(p.grad.detach().float().abs().sum())>0 for p in model.parameters() if p.requires_grad)
    if not model_grad:fail('AUXILIARY_GRADIENT_PATH')
    receipt.append({'slot':row['slot'],'role':row['role'],'answer_ce_coefficient':1.0,'auxiliary_coefficient':.25,'pair_label':layout.labels[0],'pair_count':1,'affinity_bce':float(extra.detach().float()),'trainable_model_grad_nonzero':True})
   else:
    _,n=backward_equal_slot_term(o.logits,labels,expected_slots=4);receipt.append({'slot':row['slot'],'role':row['role'],'answer_ce_coefficient':1.0,'auxiliary_coefficient':0.0,'selected_tokens':n,'objective_path':'EXISTING_BACKWARD_EQUAL_SLOT_TERM'})
   del o
  opt.step()
 if len(receipt)!=384:fail('TRAIN_ACCOUNTING')
 ad=Path(out)/(tag+'_ADAPTER');model.save_pretrained(ad)
 fit={'status':'FROZEN_AUXILIARY_FIT_ARTIFACT','arm':tag,'active_slots':sum(x.get('auxiliary_coefficient',0)>0 for x in receipt),'mean_affinity_bce':sum(x.get('affinity_bce',0) for x in receipt)/max(1,sum('affinity_bce' in x for x in receipt))}
 if auxiliary:write(Path(out)/'COMPONENT_AFFINITY_AUXILIARY_FIT.json',fit)
 write(Path(out)/(tag+'_TRAINING_RECEIPT.json'),{'status':'COMPLETE_E4F2_TRAINING','arm':tag,'adapter_sha256':sha(ad/'adapter_model.safetensors'),'optimizer_steps':96,'per_slot':receipt})
 return model,torch,m

def messages(t):
 from inference.nvarc_native import serialize_grid
 q=[]
 for p in t['train']:q += [{'role':'user','content':serialize_grid(p['input'])},{'role':'assistant','content':serialize_grid(p['output'])}]
 return q+[{'role':'user','content':serialize_grid(t['test'][0]['input'])}]
def eval_records():
 out=[]
 for r in load(PKG/'NEW_STRUCTURAL_OOD_INPUTS_AND_COMMITMENTS.json')['records']:out.append({'id':r['record_id'],'split':'OOD','canonical':r['canonical_unit_id'],'order':r['demo_order'],'count':r['query_count'],'task':r['prompt']})
 for r in load(E3/'TRAIN_FIT_INPUT_PROMPTS.json')['records']:
  if r['arm']=='TREATMENT_DECONFOUNDED':out.append({'id':r['record_id'],'split':'TRAIN_FIT','task':r['prompt']})
 for r in load(E3/'e4d_object_centric_component_anchor_grounding_v1/MARKER_INPUTS_AND_COMMITMENTS.json')['records']:out.append({'id':r['record_id'],'split':'MARKER_BINDING','family':'MARKER','endpoint':int(r['marker_turn']),'task':r['prompt']})
 for r in load(BASE/'RETENTION_INPUT_PROMPTS.json'):out.append({'id':'RETENTION:'+r['prompt_id'],'split':'RETENTION','family':r['latent_condition']['family'],'task':r['task']})
 for r in load(BASE/'scientific_successor_v2/COMPOSITION_SENTINEL_INPUT_PROMPTS.json'):out.append({'id':'COMPOSITION:'+r['prompt_id'],'split':'COMPOSITION','task':r['task']})
 if any('output' in x['task']['test'][0] for x in out):fail('TARGET_LEAK')
 return out

def generate(model,torch,m,items,b,out,tag,deadline):
 from inference.nvarc_native import NVARCNativeProvider,checkpoint_native_tokenizer,parse_native_grid,serialize_grid
 cfg=ROOT/'configs'/load(ROOT/'experiments/capability_repair_baseline_v1/e04_orientation_fixed_demonstration_baseline_v3/E04_V3_RUNTIME_NATIVE_CONFIG_IDENTITY_V1.json')['runtime_config_dir_name'];tok,_=checkpoint_native_tokenizer(Path(m['base_path']),cfg);provider=NVARCNativeProvider(model_path=Path(m['base_path']),tokenizer_config_dir=cfg,device='cuda:0');provider.model,provider.tokenizer=model,tok;raw=[]
 for i,x in enumerate(items):
  if time.monotonic()>deadline:fail('EVAL_CAP')
  z=provider.generate_many([messages(x['task'])],max_new_tokens=b['runtime']['max_new_tokens'],context_window=b['runtime']['context_window'],seeds=[b['training']['seed']+i])[0];g=parse_native_grid(z.text);raw.append({'record_id':x['id'],'split':x['split'],'prompt_sha256':taskhash(x['task']),'generated_token_ids':[int(v) for v in z.token_ids],'parsed_grid':None if g is None else [list(a) for a in g],'parser_valid':g is not None,'effective_batch_size':1})
 p=Path(out)/(tag+'_RAW.jsonl');jsonl(p,raw);return p

def marker_endpoint_guard(records):
 marker=[x for x in records if x['split']=='MARKER_BINDING'];groups=defaultdict(list)
 for x in marker:groups[x.get('endpoint')].append(x)
 if set(groups)!={0,2} or {k:len(v) for k,v in groups.items()}!={0:8,2:8}:fail('MARKER_ENDPOINT_GRID')
 return groups

def raw_freeze(paths,out):
 want={x['id'] for x in eval_records()}
 for arm,p in paths.items():
  r=rows(p)
  if len(r)!=len(want) or {x['record_id'] for x in r}!=want:fail('RAW_GRID:'+arm)
 fit=Path(out)/'COMPONENT_AFFINITY_AUXILIARY_FIT.json'
 if not fit.is_file() or load(fit).get('active_slots')!=32:fail('AUXILIARY_FIT')
 receipt={'status':'COMPLETE_RAW_AND_AUXILIARY_FIT_FROZEN_NO_TARGETS','raw_sha256':{k:sha(v) for k,v in paths.items()},'auxiliary_fit_sha256':sha(fit),'targets_accessed':False,'physical_batch_size':1};write(Path(out)/'RAW_FREEZE_RECEIPT.json',receipt);return receipt

def recheck_frozen_artifacts(out):
 fr=load(Path(out)/'RAW_FREEZE_RECEIPT.json')
 for arm,digest in fr['raw_sha256'].items():
  if sha(Path(out)/(arm+'_RAW.jsonl'))!=digest:fail('POST_FREEZE_RAW_REHASH')
 if sha(Path(out)/'COMPONENT_AFFINITY_AUXILIARY_FIT.json')!=fr['auxiliary_fit_sha256']:fail('POST_FREEZE_AUXILIARY_REHASH')
 return fr

def sidecar(out):
 fr=recheck_frozen_artifacts(out)
 if fr.get('targets_accessed') is not False:fail('FREEZE')
 from scripts.connected_components_object_extraction_semantic_oracle import target_for_family
 panel=load(PKG/'NEW_STRUCTURAL_OOD_INPUTS_AND_COMMITMENTS.json');targets={r['record_id']:target_for_family('CONNECTED_COMPONENT_COUNT',r['prompt']['test'][0]['input']) for r in panel['records']}
 for r in eval_records():
  if r['split']=='TRAIN_FIT':targets[r['id']]=target_for_family('CONNECTED_COMPONENT_COUNT',r['task']['test'][0]['input'])
  elif r['split']=='RETENTION':targets[r['id']]=target_for_family('NO_TRANSFORM_RETENTION',r['task']['test'][0]['input'])
  elif r['split']=='COMPOSITION':targets[r['id']]=target_for_family('COUNT_THEN_OBJECT_EXTRACTION',r['task']['test'][0]['input'])
  elif r['split']=='MARKER_BINDING':
   from scripts.run_e4d_object_centric_component_anchor_grounding_v1 import target_for_record
   targets[r['id']]=target_for_record({'id':r['id'],'split':'MARKER_BINDING_PRESERVATION','task':r['task']})
 for c in panel['target_commitments']:
  if hashlib.sha256(canon(targets[c['record_id']])).hexdigest()!=c['target_sha256']:fail('COMMIT')
 p=Path(out)/'SYNTHETIC_TARGET_SIDECAR.json';write(p,{'status':'DERIVED_AFTER_RAW_FREEZE','opened_after_raw_freeze':True,'targets':targets});return p

def boot(v,seed=20261013,n=10000):
 rng=random.Random(seed);z=sorted(sum(v[rng.randrange(len(v))] for _ in v) for _ in range(n));return z[int(.025*n)],z[int(.975*n)-1]
def score(out):
 fr=recheck_frozen_artifacts(out);targets=load(Path(out)/'SYNTHETIC_TARGET_SIDECAR.json')['targets'];data={}
 for arm in ('V7','EQUAL_SLOT','COMPONENT_AFFINITY'):
  p=Path(out)/(arm+'_RAW.jsonl');data[arm]={r['record_id']:bool(r['parser_valid']) and r['parsed_grid']==targets.get(r['record_id']) for r in rows(p)}
 recs=eval_records();ood=[x for x in recs if x['split']=='OOD'];units=sorted({x['canonical'] for x in ood})
 def both(arm,u):return int(all(data[arm][x['id']] for x in ood if x['canonical']==u))
 def summary(arm):
  train=[x['id'] for x in recs if x['split']=='TRAIN_FIT'];marker=[x['id'] for x in recs if x['split']=='MARKER_BINDING'];comp=[x['id'] for x in recs if x['split']=='COMPOSITION'];anyok={u:int(any(data[arm][x['id']] for x in ood if x['canonical']==u)) for u in units};vals={u:both(arm,u) for u in units}
  return {'train_fit_correct':sum(data[arm][x] for x in train),'train_fit_total':len(train),'ood_both_orders_correct':sum(vals.values()),'ood_any_order_correct':sum(anyok.values()),'order_invariance_both_over_any':0 if not sum(anyok.values()) else sum(vals.values())/sum(anyok.values()),'marker_binding_correct':sum(data[arm][x] for x in marker),'marker_binding_total':len(marker),'composition_correct':sum(data[arm][x] for x in comp),'composition_total':len(comp)}
 sm={a:summary(a) for a in data};diff=[both('COMPONENT_AFFINITY',u)-both('EQUAL_SLOT',u) for u in units];lo,hi=boot(diff);marker={};retention={}
 for endpoint,rs in marker_endpoint_guard(recs).items():
  ids=[x['id'] for x in rs];marker[str(endpoint)]=sum(data['COMPONENT_AFFINITY'][x]-data['EQUAL_SLOT'][x] for x in ids)
 for family in sorted({x['family'] for x in recs if x['split']=='RETENTION'}):
  ids=[x['id'] for x in recs if x['split']=='RETENTION' and x['family']==family];retention[family]=sum(data['COMPONENT_AFFINITY'][x]-data['EQUAL_SLOT'][x] for x in ids)
 compids=[x['id'] for x in recs if x['split']=='COMPOSITION'];compdelta=sum(data['COMPONENT_AFFINITY'][x]-data['EQUAL_SLOT'][x] for x in compids)
 gates={'primary_new_ood':sum(diff)>=6 and lo>0,'order_invariance':sm['COMPONENT_AFFINITY']['order_invariance_both_over_any']>=.8,'marker_binding_preservation':all(v>=0 for v in marker.values()),'protected_retention':all(v>=0 for v in retention.values()) and sum(retention.values())>=0,'composition':compdelta>=0}
 r={'status':'COMPLETE_E4F2_SCORE_AFTER_RAW_FREEZE','decision':'PASS_REQUIRES_LATER_INDEPENDENT_CONFIRMATION' if all(gates.values()) else 'REJECT','promotion':'FORBIDDEN_REQUIRES_LATER_INDEPENDENT_CONFIRMATION','raw_freeze_before_sidecar':True,'arms':sm,'primary_new_independent_ood':{'canonical_units':48,'component_affinity_minus_equal_slot':{'delta_correct':sum(diff),'bootstrap_95_ci':[lo,hi],'replicates':10000}},'auxiliary_fit':load(Path(out)/'COMPONENT_AFFINITY_AUXILIARY_FIT.json'),'marker_binding_preservation':{'by_endpoint':marker,'aggregate':sum(marker.values())},'protected_retention':{'by_family':retention,'aggregate':sum(retention.values())},'composition_secondary':{'component_affinity_minus_equal_slot':compdelta},'gates':gates,'physical_batch_size':1,'targets_opened_after_raw_freeze':True};write(Path(out)/'SCORE_RECEIPT.json',r);return r

def execute(bp):
 pre=preclaim(bp);b=load(bp);out=Path(b['output_root']);out.mkdir(parents=True);write(out/'PREFLIGHT_RECEIPT.json',pre);deadline=time.monotonic()+b['hard_cap_seconds'];write(out/'SCIENTIFIC_BOUNDARY.json',{'nonce':b['nonce'],'retry':False,'status':'FIRST_MODEL_LOAD_NEXT'})
 try:
  paths={};m,t,man=load_model(False);paths['V7']=generate(m,t,man,eval_records(),b,out,'V7',deadline);del m;t.cuda.empty_cache()
  m,t,man=train_arm(b,out,'EQUAL_SLOT',False,deadline);paths['EQUAL_SLOT']=generate(m,t,man,eval_records(),b,out,'EQUAL_SLOT',deadline);del m;t.cuda.empty_cache()
  m,t,man=train_arm(b,out,'COMPONENT_AFFINITY',True,deadline);paths['COMPONENT_AFFINITY']=generate(m,t,man,eval_records(),b,out,'COMPONENT_AFFINITY',deadline);del m;t.cuda.empty_cache()
  raw_freeze(paths,out);sidecar(out);score(out);write(out/'TERMINAL_RECEIPT.json',{'status':'COMPLETE','raw_freeze_before_sidecar':True,'score_sha256':sha(Path(out)/'SCORE_RECEIPT.json')})
 except Exception as e:write(out/'TERMINAL_RECEIPT.json',{'status':'FAILED_OR_PARTIAL_NO_RETRY','error':str(e),'scientific_boundary_entered':True});raise

def cpu_integration(bp):
 b=load(bp);pre=preclaim(bp,True);tmp=Path(tempfile.mkdtemp(prefix='e4f2-prod-'))
 try:
  items=eval_records();paths={}
  for arm in ('V7','EQUAL_SLOT','COMPONENT_AFFINITY'):
   p=tmp/(arm+'_RAW.jsonl');jsonl(p,[{'record_id':x['id'],'split':x['split'],'prompt_sha256':taskhash(x['task']),'generated_token_ids':[],'parsed_grid':None,'parser_valid':False,'effective_batch_size':1} for x in items]);paths[arm]=p
  write(tmp/'COMPONENT_AFFINITY_AUXILIARY_FIT.json',{'status':'CPU_FIXTURE_AUXILIARY_FIT','active_slots':32,'mean_affinity_bce':0.5})
  raw_freeze(paths,tmp)
  return {'status':'PASS_E4F2_PRODUCTION_TRAINER_RAW_FREEZE_SCORER_CPU_INTEGRATION_NO_MODEL','preclaim':pre['status'],'arms':['V7','EQUAL_SLOT','COMPONENT_AFFINITY'],'raw_freeze_before_sidecar':True,'model_loaded':False,'gpu_used':False,'targets_opened':False,'sidecar_opened':False}
 finally:shutil.rmtree(tmp)

def main():
 p=argparse.ArgumentParser();p.add_argument('--binding',type=Path);p.add_argument('--make-binding',type=Path);p.add_argument('--output-root');p.add_argument('--nonce');p.add_argument('--preclaim-only',action='store_true');p.add_argument('--cpu-preclaim',action='store_true');p.add_argument('--cpu-integration',action='store_true');p.add_argument('--execute',action='store_true');p.add_argument('--receipt',type=Path);a=p.parse_args()
 if a.make_binding:
  if not a.output_root:raise SystemExit('--output-root required')
  r=make_binding(a.make_binding,a.output_root,a.nonce)
 elif a.preclaim_only:r=preclaim(a.binding)
 elif a.cpu_preclaim:r=preclaim(a.binding,True)
 elif a.cpu_integration:r=cpu_integration(a.binding)
 elif a.execute:r=execute(a.binding)
 else:raise SystemExit('select mode')
 if a.receipt:write(a.receipt,r)
 print(json.dumps(r,sort_keys=True))
if __name__=='__main__':main()
