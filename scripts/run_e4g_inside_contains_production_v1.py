"""Complete E4G worker: target-blind two-arm runtime with post-freeze scoring."""
from __future__ import annotations
import argparse, copy, hashlib, importlib, json, os, random, subprocess, sys, time
from pathlib import Path, PurePosixPath, PureWindowsPath
ROOT=Path(os.environ.get('ARC2_ROOT',Path(__file__).resolve().parents[1]))
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
PKG=ROOT/'experiments/capability_repair_baseline_v1/connected_components_object_extraction_capability_pilot_v1/deconfounded_count_curriculum_causal_pilot_v1/e4g_inside_contains_atomic_plasticity_v1'
V7=ROOT/'experiments/capability_repair_baseline_v1/CHECKPOINT_MANIFEST_REMOTE_V1.json'
ID='E4G_INSIDE_CONTAINS_ATOMIC_PLASTICITY_POSITIVE_CONTROL_V1'; IGNORE=-100

def canon(x): return json.dumps(x,sort_keys=True,separators=(',',':')).encode()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def digest(x): return hashlib.sha256(canon(x)).hexdigest()
def load(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_bytes(canon(x)+b'\n');os.replace(q,p)
def write_jsonl(p,rows):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_bytes(b''.join(canon(x)+b'\n' for x in rows));os.replace(q,p)
def read_jsonl(p): return [json.loads(x) for x in Path(p).read_text(encoding='utf-8').splitlines() if x]
def fail(s): raise RuntimeError('E4G:'+s)
def clean_source():
 if subprocess.run(['git','status','--porcelain'],cwd=ROOT,text=True,capture_output=True).stdout: fail('SOURCE_DIRTY')
def git_head(): return subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()

def e4g_target(grid):
 direction=grid[0][0]; colors=sorted({v for row in grid[1:] for v in row if v})
 if direction not in (8,9) or len(colors)!=2: fail('E4G_ORACLE_INPUT')
 counts={c:sum(v==c for row in grid for v in row) for c in colors}; frame=max(colors,key=lambda c:counts[c]); inner=min(colors,key=lambda c:counts[c])
 return [[0,0,0],[0,inner,0],[0,0,0]] if direction==8 else [[frame,frame,frame],[frame,0,frame],[frame,frame,frame]]

def binding_inputs():
 return {'freeze':PKG/'CPU_FREEZE_RECEIPT.json','ood':PKG/'STRUCTURAL_OOD_INPUTS_AND_COMMITMENTS.json','preflight':PKG/'CPU_PRODUCTION_PATH_RECEIPT.json','schedule':PKG/'TRAINING_SCHEDULE.json','protection':PKG/'PROTECTION_PANEL_REFERENCES.json','trainfit':PKG/'TRAIN_FIT_INPUTS_AND_COMMITMENTS.json','scoring':PKG/'SCORING_CONTRACT.json','protocol':PKG/'PROTOCOL.json'}

def resolve_frozen_repo_reference(refs, field):
 """Resolve only a declared repository-relative frozen reference on any host.

 The protection-reference file is frozen as bytes and historically contains
 Windows separators.  Treat those separators as serialization, never as a
 POSIX filename.  The caller may select only the two declared frozen fields;
 absolute paths, traversal, and paths outside this exact checkout fail closed.
 """
 allowed_fields=('retention_path','composition_path')
 if field not in allowed_fields or set(refs) != {'retention_path','retention_sha256','composition_path','composition_sha256','status'}: fail('REFERENCE_SCHEMA')
 raw=refs.get(field)
 if not isinstance(raw,str) or not raw: fail('REFERENCE_TYPE')
 if PureWindowsPath(raw).is_absolute() or PurePosixPath(raw.replace('\\','/')).is_absolute(): fail('REFERENCE_ABSOLUTE')
 normalized=raw.replace('\\','/')
 parts=PurePosixPath(normalized).parts
 if not parts or any(part in ('.','..') for part in parts): fail('REFERENCE_TRAVERSAL')
 candidate=(ROOT.joinpath(*parts)).resolve()
 checkout=ROOT.resolve()
 if candidate != checkout and checkout not in candidate.parents: fail('REFERENCE_OUTSIDE_CHECKOUT')
 # This check is deliberately against the frozen references, rather than any
 # dynamic caller-provided path: no third reference can be resolved here.
 covered={
  (ROOT.joinpath(*PurePosixPath(str(refs[name]).replace('\\','/')).parts)).resolve()
  for name in allowed_fields
 }
 if candidate not in covered: fail('REFERENCE_NOT_FROZEN')
 return candidate

def schedule_task_for_sample(row):
 """Attach an already-frozen schedule identity to a transient task copy only."""
 if not isinstance(row,dict) or not isinstance(row.get('task'),dict): fail('SCHEDULE_TASK_MISSING')
 task=copy.deepcopy(row['task'])
 nested=task.get('source_id'); enclosing=row.get('source_id')
 if nested is not None and (not isinstance(nested,str) or not nested): fail('SCHEDULE_NESTED_SOURCE_ID_INVALID')
 if enclosing is not None and (not isinstance(enclosing,str) or not enclosing): fail('SCHEDULE_ENCLOSING_SOURCE_ID_INVALID')
 if nested is not None and enclosing is not None and nested!=enclosing: fail('SCHEDULE_SOURCE_ID_CONFLICT')
 resolved=nested if nested is not None else enclosing
 if resolved is None: fail('SCHEDULE_SOURCE_ID_MISSING')
 task['source_id']=resolved
 return task

def resolve(binding_path, *, check_output=True, verify_assets=True, require_clean_source=True):
 b=load(binding_path)
 if require_clean_source: clean_source()
 if b.get('experiment_id')!=ID or b.get('retry') is not False or b.get('hard_cap_seconds')!=7200: fail('BINDING_CONTRACT')
 if b.get('execution_source_commit')!=git_head(): fail('HEAD')
 for name,p in binding_inputs().items():
  x=b.get('inputs',{}).get(name,{})
  if x.get('path')!=str(p.relative_to(ROOT)).replace('\\','/') or x.get('sha256')!=sha(p): fail('INPUT_'+name)
 if sha(Path(b['worker_path']) if Path(b.get('worker_path','')).is_absolute() else ROOT/b.get('worker_path','')) != b.get('worker_sha256'): fail('WORKER_HASH')
 if check_output and Path(b['output_root']).exists(): fail('OUTPUT_EXISTS')
 guard=b.get('duplicate_guard_path')
 if guard and Path(guard).exists(): fail('DUPLICATE_ACTIVE_PROCESS')
 s=load(PKG/'TRAINING_SCHEDULE.json')['episodes']; roles={k:sum(r['role']==k for r in s) for k in ('PRIMARY_CONNECTED_COMPONENT_COUNT','PRIMARY_OBJECT_EXTRACTION','PROTECTED_REPLAY')}
 if len(s)!=384 or roles!={'PRIMARY_CONNECTED_COMPONENT_COUNT':48,'PRIMARY_OBJECT_EXTRACTION':48,'PROTECTED_REPLAY':288} or [r['optimizer_step'] for r in s]!=[i//4 for i in range(384)]: fail('SCHEDULE')
 for m in ('training_data.pipeline','inference.nvarc_native','scripts.e04_e_equal_slot_loss','scripts.connected_components_object_extraction_semantic_oracle','scripts.score_e4g_inside_contains_v1'): importlib.import_module(m)
 man=load(V7); v=b.get('v7',{})
 if v.get('manifest_sha256')!=sha(V7) or v.get('adapter_model_sha256')!='1639bf6665ccfbb99cf547f59fcd581663c95b6bad770ae7846fee90f66be1c3': fail('V7_BINDING')
 if verify_assets:
  for root,files in ((man['adapter_path'],man['adapter_files']),(man['base_path'],man['base_files'])):
   for f in files:
    p=Path(root)/f['name']
    if not p.is_file() or p.stat().st_size!=f['bytes'] or sha(p)!=f['sha256']: fail('V7_ASSET_'+f['name'])
 if not Path(b['runtime_python']).is_file(): fail('RUNTIME_PYTHON')
 return b,s,roles,man

def cpu_preclaim(binding_path):
 b,s,roles,_=resolve(binding_path,check_output=False,require_clean_source=True)
 return {'status':'PASS_E4G_PRODUCTION_PREFLIGHT_NO_MODEL','binding_sha256':sha(binding_path),'source_commit':git_head(),'schedule_slots':len(s),'roles':roles,'target_blind':True,'deferred_imports':'PASS','model_loaded':False,'gpu_used':False,'targets_opened':False,'sidecar_opened':False}

def task_records():
 ood=load(PKG/'STRUCTURAL_OOD_INPUTS_AND_COMMITMENTS.json')['records']; fit=load(PKG/'TRAIN_FIT_INPUTS_AND_COMMITMENTS.json')['records']; refs=load(PKG/'PROTECTION_PANEL_REFERENCES.json')
 out=[]
 for split,rows in (('TRAIN_FIT',fit),('OOD',ood)):
  for r in rows:
   task=copy.deepcopy(r['input_task'])
   if 'output' in task['test'][0]: fail('TARGET_LEAK')
   out.append({'record_id':split+':'+r['canonical_unit_id']+':'+r['demo_order'],'split':split,'canonical_unit_id':r['canonical_unit_id'],'demo_order':r['demo_order'],'task':task,'target_sha256':r['target_sha256'],'semantic_family':'E4G'})
 for r in load(resolve_frozen_repo_reference(refs,'retention_path')):
  task=copy.deepcopy(r['task']); out.append({'record_id':'RETENTION:'+r['prompt_id'],'split':'RETENTION','task':task,'semantic_family':r['latent_condition']['family']})
 for r in load(resolve_frozen_repo_reference(refs,'composition_path')):
  task=copy.deepcopy(r['task']); out.append({'record_id':'COMPOSITION:'+r['prompt_id'],'split':'COMPOSITION','task':task,'semantic_family':r['latent_condition']['family']})
 if any('output' in x['task']['test'][0] for x in out): fail('TARGET_LEAK')
 return out

def messages(task):
 from inference.nvarc_native import serialize_grid
 out=[]
 for pair in task['train']: out += [{'role':'user','content':serialize_grid(pair['input'])},{'role':'assistant','content':serialize_grid(pair['output'])}]
 out.append({'role':'user','content':serialize_grid(task['test'][0]['input'])});return out

def load_model(trainable=False):
 import torch
 from peft import PeftModel
 from transformers import AutoModelForCausalLM
 if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(): fail('CUDA_BF16')
 m=load(V7); base=AutoModelForCausalLM.from_pretrained(m['base_path'],local_files_only=True,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True).to('cuda:0')
 model=PeftModel.from_pretrained(base,m['adapter_path'],is_trainable=trainable)
 if trainable: model.enable_input_require_grads();model.gradient_checkpointing_enable()
 else:model.eval()
 return model,torch,m

def train(binding,out,deadline):
 from training_data.pipeline import task_to_sample
 from scripts.e04_e_equal_slot_loss import backward_equal_slot_term
 import bitsandbytes as bnb
 model,torch,_=load_model(True); torch.manual_seed(int(binding['training']['torch_seed'])); opt=bnb.optim.PagedAdamW8bit([p for p in model.parameters() if p.requires_grad],lr=float(binding['training']['learning_rate'])); accounting=[]
 for off in range(0,384,4):
  if time.monotonic()>deadline:fail('TRAIN_CAP')
  opt.zero_grad(set_to_none=True)
  for row in load(PKG/'TRAINING_SCHEDULE.json')['episodes'][off:off+4]:
   sample=task_to_sample(schedule_task_for_sample(row)); ids=torch.tensor([sample['input_ids']],device='cuda:0'); labels=torch.tensor([sample['labels']],device='cuda:0'); o=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False); term,n=backward_equal_slot_term(o.logits,labels,expected_slots=4); accounting.append({'slot':row['slot'],'role':row['role'],'selected_tokens':n,'slot_coefficient':0.25});del o,term
  opt.step()
 if len(accounting)!=384:fail('TRAIN_ACCOUNTING')
 adapter=Path(out)/'INSIDE_CONTAINS_ADAPTER';model.save_pretrained(adapter); write(Path(out)/'TRAINING_RECEIPT.json',{'status':'COMPLETE_E4G_TRAINING','optimizer':'PagedAdamW8bit','optimizer_steps':96,'slot_count':384,'slot_coefficient':0.25,'adapter_sha256':sha(adapter/'adapter_model.safetensors'),'accounting':accounting})
 return model,torch

def generate(model,torch,manifest,records,binding,out,arm,deadline):
 from inference.nvarc_native import NVARCNativeProvider,checkpoint_native_tokenizer,parse_native_grid
 config=ROOT/'experiments/capability_repair_baseline_v1/e04_orientation_fixed_demonstration_baseline_v3/E04_V3_RUNTIME_NATIVE_CONFIG_IDENTITY_V1.json'; config_dir=ROOT/'configs'/load(config)['runtime_config_dir_name']; tok,_=checkpoint_native_tokenizer(Path(manifest['base_path']),config_dir); p=NVARCNativeProvider(model_path=Path(manifest['base_path']),tokenizer_config_dir=config_dir,device='cuda:0');p.model,p.tokenizer=model,tok; raw=[]
 for i,row in enumerate(sorted(records,key=lambda x:(len(canon(x['task'])),x['record_id']))):
  if time.monotonic()>deadline:fail('GENERATION_CAP')
  x=p.generate_many([messages(row['task'])],max_new_tokens=int(binding['runtime']['max_new_tokens']),context_window=int(binding['runtime']['context_window']),seeds=[int(binding['training']['torch_seed'])+i])[0]; grid=parse_native_grid(x.text); raw.append({**{k:row[k] for k in ('record_id','split','semantic_family')},'canonical_unit_id':row.get('canonical_unit_id'),'demo_order':row.get('demo_order'),'parsed_grid':None if grid is None else [list(z) for z in grid],'parser_valid':grid is not None,'generated_token_ids':[int(v) for v in x.token_ids],'prompt_sha256':digest(row['task']),'effective_batch_size':1})
 path=Path(out)/(arm+'_RAW.jsonl');write_jsonl(path,raw);write(Path(out)/(arm+'_GENERATION_RECEIPT.json'),{'status':'COMPLETE_TARGET_BLIND_NATIVE_GREEDY_BATCH1','rows':len(raw),'raw_sha256':sha(path),'target_accessed':False,'effective_batch_size':1});return path

def target(row):
 from scripts.connected_components_object_extraction_semantic_oracle import target_for_family
 g=row['task']['test'][0]['input']; return e4g_target(g) if row['semantic_family']=='E4G' else target_for_family(row['semantic_family'],g)
def make_sidecar(records,raw_paths,out):
 from scripts.score_e4g_inside_contains_v1 import require_post_freeze
 require_post_freeze(Path(out)/'RAW_FREEZE_RECEIPT.json',raw_paths)
 targets={}
 for r in records:
  t=target(r)
  if r['semantic_family']=='E4G' and digest(t)!=next(x['target_sha256'] for x in records if x['record_id']==r['record_id']):fail('COMMITMENT')
  targets[r['record_id']]=t
 path=Path(out)/'SYNTHETIC_TARGET_SIDECAR.json';write(path,{'status':'DERIVED_AFTER_RAW_FREEZE','opened_after_raw_freeze':True,'targets':targets,'records':len(targets)});return path

def score(binding,out,records,raw_paths,sidecar):
 from scripts.score_e4g_inside_contains_v1 import require_post_freeze
 require_post_freeze(Path(out)/'RAW_FREEZE_RECEIPT.json',raw_paths)
 expected={r['record_id'] for r in records}
 if set(raw_paths)!={'V7','TRAINED'}: fail('RAW_ARMS')
 arms={a:{r['record_id']:r for r in read_jsonl(p)} for a,p in raw_paths.items()}
 if any(set(rows)!=expected or len(rows)!=len(expected) for rows in arms.values()): fail('RAW_COMPLETENESS')
 sidecar_data=load(sidecar)
 if sidecar_data.get('status')!='DERIVED_AFTER_RAW_FREEZE' or sidecar_data.get('opened_after_raw_freeze') is not True: fail('SIDECAR_ORDER')
 targets=sidecar_data.get('targets',{})
 if set(targets)!=expected: fail('SIDECAR_COMPLETENESS')
 def good(a,rid): return arms[a][rid]['parsed_grid']==targets[rid]
 units=sorted({r['canonical_unit_id'] for r in records if r['split']=='OOD'}); fitunits=sorted({r['canonical_unit_id'] for r in records if r['split']=='TRAIN_FIT'})
 def both(a,u,split): return all(good(a,r['record_id']) for r in records if r['split']==split and r['canonical_unit_id']==u)
 v=[both('V7',u,'OOD') for u in units];t=[both('TRAINED',u,'OOD') for u in units];rng=random.Random(20261013);d=[int(y)-int(x) for x,y in zip(v,t)];draws=sorted(sum(d[rng.randrange(48)] for _ in range(48)) for _ in range(10000)); retention={}
 for fam in sorted({r['semantic_family'] for r in records if r['split']=='RETENTION'}): retention[fam]=sum(good('TRAINED',r['record_id'])-good('V7',r['record_id']) for r in records if r['split']=='RETENTION' and r['semantic_family']==fam)
 comp=sum(good('TRAINED',r['record_id'])-good('V7',r['record_id']) for r in records if r['split']=='COMPOSITION'); trainfit=sum(both('TRAINED',u,'TRAIN_FIT') for u in fitunits); anyo=sum(any(good('TRAINED',r['record_id']) for r in records if r['split']=='OOD' and r['canonical_unit_id']==u) for u in units); botho=sum(t); metrics={'train_fit':trainfit,'ood_delta':sum(d),'bootstrap_lower':draws[250],'order_invariance':botho/anyo if anyo else 0.0,'retention_aggregate':sum(retention.values()),'retention_families':retention,'composition_delta':comp}; from scripts.e4g_inside_contains_atomic import decision
 receipt={'status':'COMPLETE_E4G_SCORE_AFTER_RAW_FREEZE','metrics':metrics,'bootstrap_95_ci':[draws[250],draws[9749]],'decision':decision(metrics),'targets_opened_after_raw_freeze':True,'raw_sha256':{a:sha(p) for a,p in raw_paths.items()}};write(Path(out)/'SCORE_RECEIPT.json',receipt);return receipt

def acquire_duplicate_guard(binding):
 guard=binding.get('duplicate_guard_path')
 if not guard: return None
 p=Path(guard); p.parent.mkdir(parents=True,exist_ok=True)
 try:
  with p.open('x',encoding='utf-8') as f: json.dump({'experiment_id':ID,'nonce':binding['nonce'],'pid':os.getpid()},f,sort_keys=True)
 except FileExistsError: fail('DUPLICATE_ACTIVE_PROCESS')
 return p

def execute(binding_path):
 pre=cpu_preclaim(binding_path); b=load(binding_path);out=Path(b['output_root']);out.mkdir(parents=True); lock=acquire_duplicate_guard(b); deadline=time.monotonic()+7200;write(out/'PREFLIGHT_RECEIPT.json',pre)
 try:
  write(out/'SCIENTIFIC_BOUNDARY_RECEIPT.json',{'status':'FIRST_MODEL_LOAD_NEXT','nonce':b['nonce'],'retry':False}); records=task_records(); raw={};v,torch,m=load_model(False);raw['V7']=generate(v,torch,m,records,b,out,'V7',deadline);del v;torch.cuda.empty_cache();trained,torch=train(b,out,deadline);raw['TRAINED']=generate(trained,torch,m,records,b,out,'TRAINED',deadline);del trained;torch.cuda.empty_cache(); hashes={a:sha(p) for a,p in raw.items()};write(out/'RAW_FREEZE_RECEIPT.json',{'status':'COMPLETE_RAW_FROZEN_NO_TARGETS','raw_sha256':hashes,'targets_accessed':False,'nonce':b['nonce'],'complete_arms':sorted(raw)});side=make_sidecar(records,raw,out);score(b,out,records,raw,side);write(out/'TERMINAL_RECEIPT.json',{'status':'COMPLETE','raw_freeze_before_sidecar':True,'score_sha256':sha(out/'SCORE_RECEIPT.json')})
 except Exception as e:
  write(out/'TERMINAL_RECEIPT.json',{'status':'FAILED_OR_PARTIAL_NO_RETRY','error':type(e).__name__+':'+str(e),'scientific_boundary_entered':(out/'SCIENTIFIC_BOUNDARY_RECEIPT.json').exists(),'raw_frozen':(out/'RAW_FREEZE_RECEIPT.json').exists()});raise
 finally:
  if lock is not None and lock.exists(): lock.unlink()

def cpu_fixture(binding_path):
 # This uses newly constructed zero-target fixtures only.  It never reads a real
 # sidecar and therefore proves the launcher/worker/freeze/scorer ordering without
 # opening an experiment target or importing a model backend.
 b,s,roles,_=resolve(binding_path,check_output=False,verify_assets=False,require_clean_source=False)
 records=[r for r in task_records() if r['split'] in {'TRAIN_FIT','OOD'}]
 if any('output' in r['task']['test'][0] for r in records) or len(records)!=192: fail('FIXTURE_TARGET_BLIND')
 out=Path(binding_path).parent/'e4g_cpu_fixture_output'; out.mkdir(exist_ok=True)
 raw={}
 for arm in ('V7','TRAINED'):
  rows=[]
  for r in records:
   rows.append({'record_id':r['record_id'],'split':r['split'],'semantic_family':r['semantic_family'],'canonical_unit_id':r['canonical_unit_id'],'demo_order':r['demo_order'],'parsed_grid':None,'parser_valid':False,'generated_token_ids':[],'prompt_sha256':digest(r['task']),'effective_batch_size':1})
  path=out/(arm+'_RAW.jsonl'); write_jsonl(path,rows); raw[arm]=path
 raw_hashes={a:sha(p) for a,p in raw.items()}
 write(out/'RAW_FREEZE_RECEIPT.json',{'status':'COMPLETE_RAW_FROZEN_NO_TARGETS','raw_sha256':raw_hashes,'targets_accessed':False,'complete_arms':['TRAINED','V7'],'fixture_only':True})
 # The CPU fixture's targets are independent constant grids, not E4G targets.
 targets={r['record_id']:[[0]] for r in records}
 side=out/'SYNTHETIC_TARGET_SIDECAR.json'; write(side,{'status':'DERIVED_AFTER_RAW_FREEZE','opened_after_raw_freeze':True,'fixture_only':True,'targets':targets})
 scored=score(b,out,records,raw,side)
 write(out/'TERMINAL_RECEIPT.json',{'status':'COMPLETE_CPU_FIXTURE','raw_freeze_before_sidecar':True,'score_sha256':sha(out/'SCORE_RECEIPT.json')})
 receipt={'status':'PASS_E4G_COMPLETE_PRODUCTION_PATH_CPU_FIXTURE','roles':roles,'records_per_arm':len(records),'arms':sorted(raw),'stage_order':['preclaim','target_blind_generation','complete_raw_freeze','sidecar','score','terminal'],'score_status':scored['status'],'model_loaded':False,'gpu_used':False,'targets_opened':False,'sidecar_opened':False,'fixture_only':True}
 write(Path(binding_path).parent/'E4G_CPU_COMPLETE_PATH_RECEIPT.json',receipt)
 return receipt

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--binding',required=True);ap.add_argument('--preclaim-only',action='store_true');ap.add_argument('--cpu-self-test',action='store_true');ap.add_argument('--execute',action='store_true');a=ap.parse_args()
 if a.preclaim_only: print(json.dumps(cpu_preclaim(a.binding),sort_keys=True))
 elif a.cpu_self_test: print(json.dumps(cpu_fixture(a.binding),sort_keys=True))
 elif a.execute: execute(a.binding)
 else: ap.error('select mode')
