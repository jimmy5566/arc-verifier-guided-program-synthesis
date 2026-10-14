"""Hash-bound worker for the E4D representation-versus-generation diagnostic.

Importing this module is CPU-only.  Model and CUDA imports are deliberately
deferred to ``execute`` after binding preclaim succeeds.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, os, subprocess, sys
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
 values={arm:sha(path) for arm,path in raw_paths.items()};write(Path(out)/'RAW_FREEZE_RECEIPT.json',{'status':'COMPLETE_RAW_FROZEN_NO_TARGETS','raw_sha256':values,'targets_accessed':False});return values

def sidecar_allowed(out):
 receipt=load(Path(out)/'RAW_FREEZE_RECEIPT.json')
 if receipt.get('status')!='COMPLETE_RAW_FROZEN_NO_TARGETS' or receipt.get('targets_accessed') is not False or set(receipt.get('raw_sha256',{}))!={'V7','EQUAL_SLOT','COMPONENT_ANCHOR'}:fail('SIDECAR_BEFORE_RAW_FREEZE')
 return receipt

def cpu_integration(binding_path):
 import tempfile,shutil
 b=load(binding_path);tmp=Path(tempfile.mkdtemp(prefix='e4d-repbind-'))
 try:
  pre=preclaim(binding_path,True,True);paths={}
  for arm in ('V7','EQUAL_SLOT','COMPONENT_ANCHOR'):
   rows=[]
   for row in records():
    pos,shape=final_query_digit_positions(row['task']);rows.append({'record_id':row['record_id'],'split':row['split'],'count':row.get('count'),'prompt_sha256':row['prompt_sha256'],'final_query_positions':pos,'query_shape':shape,'layer_features':{str(layer):[0.0,1.0] for layer in LAYERS},'native_count':None})
   paths[arm]=tmp/(arm+'_RAW.jsonl');write_jsonl(paths[arm],rows)
  frozen=raw_freeze(paths,tmp);sidecar_allowed(tmp)
  return {'status':'PASS_E4D_REPRESENTATION_BINDING_REAL_TOKENIZER_CPU_INTEGRATION','preclaim_status':pre['status'],'records_per_arm':144,'three_arm_raw_hashes':frozen,'raw_freeze_before_sidecar':True,'required_layer_keys':[str(x) for x in LAYERS],'model_loaded':False,'gpu_used':False,'sidecar_opened':False}
 finally: shutil.rmtree(tmp)

def execute(binding_path):
 # Delayed production path.  The authorization gate calls this only after
 # preclaim.  It intentionally has no training branch; all checkpoint weights
 # remain frozen and only readout parameters are derived from train features.
 pre=preclaim(binding_path); raise RuntimeError('AUTHORIZED_EXECUTOR_REQUIRES_BOUND_RUNTIME_ADAPTER_LOADER')

def main():
 p=argparse.ArgumentParser();p.add_argument('--binding',type=Path,required=True);p.add_argument('--make-binding',action='store_true');p.add_argument('--output-root');p.add_argument('--nonce');p.add_argument('--preclaim-only',action='store_true');p.add_argument('--cpu-integration',action='store_true');p.add_argument('--execute',action='store_true');p.add_argument('--receipt',type=Path);a=p.parse_args()
 if a.make_binding: r=make_binding(a.binding,a.output_root,a.nonce)
 elif a.preclaim_only:r=preclaim(a.binding)
 elif a.cpu_integration:r=cpu_integration(a.binding)
 elif a.execute:r=execute(a.binding)
 else:raise SystemExit('select one mode')
 if a.receipt:write(a.receipt,r)
 print(json.dumps(r,sort_keys=True))
if __name__=='__main__':main()
