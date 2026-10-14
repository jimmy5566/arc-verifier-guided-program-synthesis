"""Hash-bound worker for the E4D representation-versus-generation diagnostic.

Importing this module is CPU-only.  Model and CUDA imports are deliberately
deferred to ``execute`` after binding preclaim succeeds.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, math, os, subprocess, sys
from pathlib import Path

ROOT=Path(os.environ.get("ARC2_ROOT",Path(__file__).resolve().parents[1]))
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
BASE=ROOT/'experiments/capability_repair_baseline_v1/connected_components_object_extraction_capability_pilot_v1'
E3=BASE/'deconfounded_count_curriculum_causal_pilot_v1'
PKG=E3/'e4d_representation_vs_generation_diagnostic_v1'
E4D=E3/'e4d_object_centric_component_anchor_grounding_v1'
ID='E4D_REPRESENTATION_VS_GENERATION_DIAGNOSTIC_V1'; LAYERS=(8,16,24,31)

def canon(x): return json.dumps(x,sort_keys=True,separators=(',',':')).encode()
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_bytes(canon(x)+b'\n');os.replace(q,p)
def write_jsonl(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_bytes(b''.join(canon(r)+b'\n' for r in x));os.replace(q,p)
def fail(x): raise RuntimeError('E4D_DIAGNOSTIC:'+x)

def task_for_positions(task):
 """Provide only a fixed dummy assistant grid after the query user turn.

 The dummy exists solely because the frozen training serializer represents a
 supervised pair; it is never a target and lies after every extracted query
 position in the causal sequence.
 """
 item=copy.deepcopy(task)
 item['test'][0].setdefault('output',[[0,0,0],[0,0,0],[0,0,0]])
 return item

def final_query_digit_positions(task):
 from scripts.e4d_object_centric_component_anchor_grounding import query_grid_anchor_layout
 layout=query_grid_anchor_layout(task_for_positions(task))
 h,w=layout.query_shape
 if len(layout.positions)!=h*w or tuple(layout.positions)!=tuple(sorted(layout.positions)): fail('FINAL_QUERY_POSITION_LAYOUT')
 return list(layout.positions),[h,w]

def records():
 train=load(PKG/'TRAIN_DERIVED_PROBE_SPLIT.json')['records'];ood=load(PKG/'OOD_INPUTS_AND_COMMITMENTS.json')['records']
 out=[]
 for row in train: out.append({'record_id':'TRAIN:'+row['canonical_unit_id'],'split':'TRAIN_PROBE','count':row['count'],'task':row['task'],'prompt_sha256':row['prompt_sha256']})
 for row in ood: out.append({'record_id':row['record_id'],'split':'OOD','canonical_unit_id':row['canonical_unit_id'],'demo_order':row['demo_order'],'task':row['prompt'],'prompt_sha256':row['prompt_sha256']})
 if len(out)!=144 or len({x['record_id'] for x in out})!=144 or sum(x['split']=='TRAIN_PROBE' for x in out)!=48: fail('RECORD_GRID')
 return out

def input_paths():
 return {name:PKG/name for name in ('OOD_INPUTS_AND_COMMITMENTS.json','TRAIN_DERIVED_PROBE_SPLIT.json','CHECKPOINT_IDENTITIES.json','PROBE_CONTRACT.json','CPU_VALIDATION_RECEIPT.json','CHECKPOINT_BYTE_VERIFICATION_RECEIPT.json')}

def make_binding(path,output_root,nonce,allow_dirty=False):
 if not allow_dirty: clean_source()
 commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
 launcher=ROOT/'scripts/launch_e4d_representation_binding_diagnostic_v1.sh'; scorer=ROOT/'scripts/score_e4d_representation_binding_diagnostic_v1.py'
 binding_path=Path(path).resolve()
 binding_ref=binding_path.relative_to(ROOT).as_posix() if binding_path.is_relative_to(ROOT) else str(binding_path)
 b={'schema_version':1,'experiment_id':ID,'execution_source_commit':commit,'nonce':nonce,'output_root':str(output_root),'retry':False,'hard_cap_seconds':7200,
    'command':['bash','scripts/launch_e4d_representation_binding_diagnostic_v1.sh','--binding',binding_ref],
    'worker':{'path':'scripts/run_e4d_representation_binding_diagnostic_v1.py','sha256':sha(Path(__file__))},'launcher':{'path':'scripts/launch_e4d_representation_binding_diagnostic_v1.sh','sha256':sha(launcher)},'scorer':{'path':'scripts/score_e4d_representation_binding_diagnostic_v1.py','sha256':sha(scorer)},
    'inputs':{name:{'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p)} for name,p in input_paths().items()},
    'checkpoint_identities':load(PKG/'CHECKPOINT_IDENTITIES.json'),'runtime':{'physical_batch_size':1,'decoder':'NVARC_NATIVE_GREEDY','max_new_tokens':192,'context_window':16384,'cuda_required':True,'bf16_required':True},
    'probe':{'layers':list(LAYERS),'primary_layer':31,'position':'mean_pool_all_final_query_digit_tokens_row_major','readout':'class_centroid_affine_linear','model_weights_frozen':True,'readout_fit_split':'TRAIN_DERIVED_PROBE_SPLIT.json'},
    'invariants':{'three_arm_completeness':True,'raw_freeze_before_sidecar':True,'target_blind_inference':True,'no_training':True,'no_checkpoint_promotion':True,'duplicate_output_guard':True}}
 write(path,b);return b

def clean_source():
 for args in (['git','diff','--quiet'],['git','diff','--cached','--quiet']):
  if subprocess.run(args,cwd=ROOT).returncode: fail('TRACKED_SOURCE_DIRTY')

def position_audit():
 audit=[]
 for row in records():
  positions,shape=final_query_digit_positions(row['task']);audit.append({'record_id':row['record_id'],'split':row['split'],'query_shape':shape,'position_count':len(positions),'first':positions[0],'last':positions[-1]})
 if len(audit)!=144 or any(x['position_count']!=x['query_shape'][0]*x['query_shape'][1] for x in audit):fail('POSITION_AUDIT')
 return audit

def preclaim(binding_path,allow_missing_runtime=False,allow_dirty=False):
 b=load(binding_path)
 if b.get('experiment_id')!=ID or b.get('retry') is not False or b.get('hard_cap_seconds')!=7200:fail('BINDING')
 if Path(b['output_root']).exists():fail('OUTPUT_EXISTS')
 if subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()!=b['execution_source_commit']:fail('SOURCE_HEAD')
 if not allow_dirty: clean_source()
 for key in ('worker','launcher','scorer'):
  p=ROOT/b[key]['path']
  if not p.is_file() or sha(p)!=b[key]['sha256']:fail('CODE_HASH:'+key)
 for _,entry in b['inputs'].items():
  p=ROOT/entry['path']
  if not p.is_file() or sha(p)!=entry['sha256']:fail('INPUT_HASH')
 audit=position_audit()
 if not allow_missing_runtime:
  if subprocess.run(['nvidia-smi'],capture_output=True).returncode:fail('CUDA')
 return {'status':'PASS_E4D_REPRESENTATION_BINDING_PRODUCTION_PRECLAIM_NO_MODEL','records_per_arm':144,'arms':['V7','EQUAL_SLOT','COMPONENT_ANCHOR'],'position_audit':audit,'model_loaded':False,'gpu_used':False,'sidecar_opened':False}

def raw_freeze(raw_paths,out):
 required={'V7','EQUAL_SLOT','COMPONENT_ANCHOR'}
 if set(raw_paths)!=required or any(not Path(p).is_file() for p in raw_paths.values()):fail('RAW_MISSING')
 validation=validate_raw_paths(raw_paths)
 values={arm:sha(path) for arm,path in raw_paths.items()};write(Path(out)/'RAW_FREEZE_RECEIPT.json',{'status':'COMPLETE_RAW_FROZEN_NO_TARGETS','raw_sha256':values,'record_validation':validation,'targets_accessed':False});return values

def raw_rows(path):
 try:return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line]
 except Exception as error:fail('RAW_JSON:'+type(error).__name__)

def validate_raw_paths(raw_paths):
 """Validate every target-blind raw record before it can establish a freeze."""
 expected={row['record_id']:row for row in records()};summary={}
 for arm,path in raw_paths.items():
  rows=raw_rows(path);ids=[row.get('record_id') for row in rows]
  if len(rows)!=144 or len(ids)!=len(set(ids)) or set(ids)!=set(expected):fail('RAW_IDENTITY:'+arm)
  train=ood=0;widths=set()
  for row in rows:
   source=expected[row['record_id']]
   if row.get('split')!=source['split'] or row.get('prompt_sha256')!=source['prompt_sha256']:fail('RAW_PROMPT:'+arm+':'+row['record_id'])
   if source['split']=='TRAIN_PROBE':
    train+=1
    if row.get('count')!=source['count']:fail('RAW_TRAIN_LABEL:'+arm+':'+row['record_id'])
   else:
    ood+=1
    if row.get('count') is not None:fail('RAW_OOD_LABEL_PRESENT:'+arm+':'+row['record_id'])
   features=row.get('layer_features')
   if not isinstance(features,dict) or set(features)!={str(layer) for layer in LAYERS}:fail('RAW_LAYERS:'+arm+':'+row['record_id'])
   for vector in features.values():
    if not isinstance(vector,list) or not vector or any(not isinstance(value,(int,float)) or not math.isfinite(float(value)) for value in vector):fail('RAW_FEATURE_VALUES:'+arm+':'+row['record_id'])
    widths.add(len(vector))
   if not isinstance(row.get('parser_valid'),bool) or not isinstance(row.get('generated_token_ids'),list) or any(not isinstance(value,int) for value in row['generated_token_ids']):fail('RAW_NATIVE_FIELDS:'+arm+':'+row['record_id'])
   native=row.get('native_count')
   if native is not None and (not isinstance(native,int) or native<0 or native>3):fail('RAW_NATIVE_COUNT:'+arm+':'+row['record_id'])
  if train!=48 or ood!=96 or len(widths)!=1:fail('RAW_GRID_OR_WIDTH:'+arm)
  summary[arm]={'records':144,'train_probe':train,'ood':ood,'feature_width':next(iter(widths)),'sha256':sha(path)}
 return summary

def reverify_raw_freeze(raw_paths,out):
 receipt=load(Path(out)/'RAW_FREEZE_RECEIPT.json')
 if receipt.get('status')!='COMPLETE_RAW_FROZEN_NO_TARGETS' or receipt.get('targets_accessed') is not False:fail('RAW_FREEZE_STATUS')
 if any(not Path(path).is_file() for path in raw_paths.values()):fail('RAW_POST_FREEZE_MUTATION')
 if receipt.get('raw_sha256')!={arm:sha(path) for arm,path in raw_paths.items()}:fail('RAW_POST_FREEZE_MUTATION')
 validation=validate_raw_paths(raw_paths)
 if receipt.get('raw_sha256')!={arm:summary['sha256'] for arm,summary in validation.items()}:fail('RAW_POST_FREEZE_MUTATION')
 return receipt

def sidecar_allowed(out):
 out=Path(out);paths={arm:out/(arm+'_RAW.jsonl') for arm in ('V7','EQUAL_SLOT','COMPONENT_ANCHOR')}
 return reverify_raw_freeze(paths,out)

def verify_runtime_assets(binding):
 """Hash every adapter/base file before the first model import."""
 identities=binding['checkpoint_identities'];checks=[]
 for entry in identities['V7']['base_files']:
  path=Path(identities['V7']['base_path'])/entry['name']
  if not path.is_file() or path.stat().st_size!=entry['bytes'] or sha(path)!=entry['sha256']:fail('BASE_ASSET:'+entry['name'])
  checks.append({'path':str(path),'sha256':entry['sha256']})
 for arm in ('V7','EQUAL_SLOT','COMPONENT_ANCHOR'):
  item=identities[arm];path=Path(item['adapter_path'])/'adapter_model.safetensors'
  expected=next(x['sha256'] for x in item['adapter_files'] if x['name']=='adapter_model.safetensors') if arm=='V7' else item['adapter_model_sha256']
  if not path.is_file() or path.stat().st_size!=528550256 or sha(path)!=expected:fail('ADAPTER_ASSET:'+arm)
  checks.append({'arm':arm,'path':str(path),'sha256':expected})
 return checks

def native_feature_positions(tokenizer, task):
 """Find the final query digit span in the actual native chat token stream."""
 from inference.nvarc_native import native_messages,serialize_grid
 messages=native_messages(task,0);encoded=tokenizer.apply_chat_template(messages,add_generation_prompt=True,tokenize=True,return_tensors='pt',return_dict=True)
 ids=[int(x) for x in encoded['input_ids'][0].tolist()];grid=task['test'][0]['input'];grid_ids=[int(x) for x in tokenizer.encode(serialize_grid(grid),add_special_tokens=False)]
 starts=[i for i in range(len(ids)-len(grid_ids)+1) if ids[i:i+len(grid_ids)]==grid_ids]
 if not starts:fail('NATIVE_QUERY_GRID_NOT_FOUND')
 start=max(starts);h,w=len(grid),len(grid[0]);expected=[];local_positions=[]
 for y in range(h):
  for x in range(w):
   token=[int(v) for v in tokenizer.encode(str(grid[y][x]),add_special_tokens=False)]
   if len(token)!=1:fail('NATIVE_CELL_NOT_SINGLE_TOKEN')
   local_positions.append(len(expected));expected.extend(token)
  if y<h-1: expected.extend(int(v) for v in tokenizer.encode('\n',add_special_tokens=False))
 if expected!=grid_ids:fail('NATIVE_GRID_SERIALIZATION_MISMATCH')
 positions=[start+i for i in local_positions]
 if len(positions)!=h*w or any(ids[p]!=grid_ids[i] for p,i in zip(positions,local_positions,strict=True)):fail('NATIVE_QUERY_POSITION_MISMATCH')
 return encoded,positions

def native_tokenizer_audit(receipt_path):
 """Pre-model audit of the exact native chat-template tokenizer only."""
 from inference.nvarc_native import checkpoint_native_tokenizer
 identities=load(PKG/'CHECKPOINT_IDENTITIES.json');base=Path(identities['V7']['base_path'])
 tokenizer,metadata=checkpoint_native_tokenizer(base,ROOT/'configs/nvarc_native_846d0198');audit=[]
 for row in records():
  encoded,positions=native_feature_positions(tokenizer,row['task']);grid=row['task']['test'][0]['input'];audit.append({'record_id':row['record_id'],'split':row['split'],'prompt_sha256':row['prompt_sha256'],'input_tokens':int(encoded['input_ids'].shape[-1]),'query_shape':[len(grid),len(grid[0])],'position_count':len(positions),'first_position':positions[0],'last_position':positions[-1]})
 if len(audit)!=144 or sum(row['split']=='TRAIN_PROBE' for row in audit)!=48 or any(row['position_count']!=row['query_shape'][0]*row['query_shape'][1] for row in audit):fail('NATIVE_TOKENIZER_AUDIT_GRID')
 payload={'status':'PASS_E4D_REPRESENTATION_BINDING_NATIVE_TOKENIZER_AUDIT_NO_MODEL','source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'worker_sha256':sha(Path(__file__)),'tokenizer_metadata':metadata,'records':audit,'records_per_arm':144,'train_probe_records':48,'ood_records':96,'last_native_chat_span_selected':True,'model_loaded':False,'gpu_used':False,'targets_opened':False}
 write(receipt_path,payload);return payload

def load_arm(binding,arm):
 import torch
 from peft import PeftModel
 from transformers import AutoModelForCausalLM
 from inference.nvarc_native import checkpoint_native_tokenizer,NVARCNativeProvider
 identities=binding['checkpoint_identities'];base=Path(identities['V7']['base_path']);tokenizer,_=checkpoint_native_tokenizer(base,ROOT/'configs/nvarc_native_846d0198')
 model=AutoModelForCausalLM.from_pretrained(str(base),local_files_only=True,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True).to('cuda:0').eval()
 model=PeftModel.from_pretrained(model,identities[arm]['adapter_path']).eval();provider=NVARCNativeProvider(model_path=base,tokenizer_config_dir=ROOT/'configs/nvarc_native_846d0198',device='cuda:0');provider.model,provider.tokenizer=model,tokenizer
 return model,torch,tokenizer,provider

def generate_arm(binding,arm,out):
 """Write target-blind native predictions and mean-pooled frozen features."""
 from inference.nvarc_native import native_messages,parse_native_grid
 model,torch,tokenizer,provider=load_arm(binding,arm);rows=[]
 try:
  for index,row in enumerate(records()):
   encoded,positions=native_feature_positions(tokenizer,row['task']);encoded={k:v.to('cuda:0') for k,v in encoded.items()}
   with torch.inference_mode(): result=model(**encoded,output_hidden_states=True,use_cache=False)
   feature={str(layer):[float(x) for x in result.hidden_states[layer][0,positions].float().mean(dim=0).cpu().tolist()] for layer in LAYERS}
   generation=provider.generate(native_messages(row['task'],0),max_new_tokens=binding['runtime']['max_new_tokens'],context_window=binding['runtime']['context_window'],seed=20261013+index)
   grid=parse_native_grid(generation.text);native_count=None if grid is None or len(grid)!=3 or any(len(r)!=3 for r in grid) else sum(x!=0 for x in grid[1])
   rows.append({'record_id':row['record_id'],'split':row['split'],'count':row.get('count'),'prompt_sha256':row['prompt_sha256'],'layer_features':feature,'native_count':native_count,'generated_token_ids':list(generation.token_ids),'parser_valid':grid is not None})
   del result,encoded
  path=Path(out)/(arm+'_RAW.jsonl');write_jsonl(path,rows);return path
 finally:
  del model;torch.cuda.empty_cache()

def cpu_integration(binding_path):
 import tempfile,shutil
 b=load(binding_path);tmp=Path(tempfile.mkdtemp(prefix='e4d-repbind-'))
 try:
  pre=preclaim(binding_path,True,True);paths={}
  for arm in ('V7','EQUAL_SLOT','COMPONENT_ANCHOR'):
   rows=[]
   for row in records():
    pos,shape=final_query_digit_positions(row['task']);rows.append({'record_id':row['record_id'],'split':row['split'],'count':row.get('count'),'prompt_sha256':row['prompt_sha256'],'final_query_positions':pos,'query_shape':shape,'layer_features':{str(layer):[0.0,1.0] for layer in LAYERS},'native_count':None,'generated_token_ids':[],'parser_valid':False})
   paths[arm]=tmp/(arm+'_RAW.jsonl');write_jsonl(paths[arm],rows)
  frozen=raw_freeze(paths,tmp);sidecar=make_sidecar_after_freeze(tmp)
  from scripts.score_e4d_representation_binding_diagnostic_v1 import run as score
  score_result=score(PKG/'TRAIN_DERIVED_PROBE_SPLIT.json',PKG/'OOD_INPUTS_AND_COMMITMENTS.json',sidecar,paths,tmp/'SCORE_RECEIPT.json')
  return {'status':'PASS_E4D_REPRESENTATION_BINDING_REAL_TOKENIZER_CPU_INTEGRATION','preclaim_status':pre['status'],'records_per_arm':144,'three_arm_raw_hashes':frozen,'raw_freeze_before_sidecar':True,'required_layer_keys':[str(x) for x in LAYERS],'score_status':score_result['status'],'model_loaded':False,'gpu_used':False,'sidecar_opened_after_raw_freeze':True}
 finally: shutil.rmtree(tmp)

def make_sidecar_after_freeze(out):
 """Open only committed synthetic OOD targets after the three raw files freeze."""
 sidecar_allowed(out)
 from scripts.connected_components_object_extraction_semantic_oracle import target_for_family,validate_demonstrations
 panel=load(PKG/'OOD_INPUTS_AND_COMMITMENTS.json');targets={}
 for row in panel['records']:
  task=row['prompt'];validate_demonstrations(task,'CONNECTED_COMPONENT_COUNT')
  targets[row['record_id']]=target_for_family('CONNECTED_COMPONENT_COUNT',task['test'][0]['input'])
 commitments={row['record_id']:row['target_sha256'] for row in panel['target_commitments']}
 if set(targets)!=set(commitments):fail('SIDECAR_COMMITMENT_GRID')
 for record_id,target in targets.items():
  if hashlib.sha256(canon(target)).hexdigest()!=commitments[record_id]:fail('SIDECAR_TARGET_COMMITMENT:'+record_id)
 path=Path(out)/'SYNTHETIC_TARGET_SIDECAR.json'
 write(path,{'status':'DERIVED_AFTER_RAW_FREEZE','opened_after_raw_freeze':True,'oracle':'PASS_INDEPENDENT_CONNECTED_COMPONENT_COUNT','commitments_verified':len(commitments),'targets':targets})
 return path

def execute(binding_path):
 # Delayed production path.  The authorization gate calls this only after
 # preclaim.  It intentionally has no training branch; all checkpoint weights
 # remain frozen and only readout parameters are derived from train features.
 pre=preclaim(binding_path);b=load(binding_path);out=Path(b['output_root']);out.mkdir(parents=True);write(out/'PREFLIGHT_RECEIPT.json',{**pre,'runtime_asset_checks':verify_runtime_assets(b)})
 paths={arm:generate_arm(b,arm,out) for arm in ('V7','EQUAL_SLOT','COMPONENT_ANCHOR')};raw_freeze(paths,out)
 sidecar=make_sidecar_after_freeze(out)
 from scripts.score_e4d_representation_binding_diagnostic_v1 import run as score
 score(PKG/'TRAIN_DERIVED_PROBE_SPLIT.json',PKG/'OOD_INPUTS_AND_COMMITMENTS.json',sidecar,paths,out/'SCORE_RECEIPT.json')
 write(out/'TERMINAL_RECEIPT.json',{'status':'COMPLETE','raw_freeze_before_sidecar':True,'raw_freeze_sha256':sha(out/'RAW_FREEZE_RECEIPT.json'),'sidecar_sha256':sha(sidecar),'score_sha256':sha(out/'SCORE_RECEIPT.json')})
 return {'status':'COMPLETE_E4D_REPRESENTATION_BINDING_DIAGNOSTIC','model_loaded':True,'gpu_used':True,'raw_freeze_before_sidecar':True}

def main():
 p=argparse.ArgumentParser();p.add_argument('--binding',type=Path);p.add_argument('--make-binding',action='store_true');p.add_argument('--output-root');p.add_argument('--nonce');p.add_argument('--preclaim-only',action='store_true');p.add_argument('--cpu-integration',action='store_true');p.add_argument('--native-tokenizer-audit',action='store_true');p.add_argument('--execute',action='store_true');p.add_argument('--receipt',type=Path);a=p.parse_args()
 if not a.native_tokenizer_audit and a.binding is None:raise SystemExit('--binding required')
 if a.make_binding: r=make_binding(a.binding,a.output_root,a.nonce)
 elif a.preclaim_only:r=preclaim(a.binding)
 elif a.cpu_integration:r=cpu_integration(a.binding)
 elif a.native_tokenizer_audit:
  if a.receipt is None:raise SystemExit('--receipt required for native tokenizer audit')
  r=native_tokenizer_audit(a.receipt)
 elif a.execute:r=execute(a.binding)
 else:raise SystemExit('select one mode')
 if a.receipt:write(a.receipt,r)
 print(json.dumps(r,sort_keys=True))
if __name__=='__main__':main()
