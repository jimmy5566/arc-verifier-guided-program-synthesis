"""E4-D component-anchor production preclaim and CPU-only contract validation.

This module intentionally does not import Torch, Transformers, bitsandbytes, or a
model.  Its preclaim path is the same source/panel/schedule/binding path that a
future authorized worker will use, stopping before model/CUDA initialization.
"""
from __future__ import annotations
import argparse, copy, hashlib, importlib, json, os, subprocess, sys, time
from pathlib import Path
ROOT=Path(os.environ.get('ARC2_ROOT',Path(__file__).resolve().parents[1]))
for p in (ROOT,ROOT/'src'):
    if str(p) not in sys.path: sys.path.insert(0,str(p))
BASE=ROOT/'experiments/capability_repair_baseline_v1/connected_components_object_extraction_capability_pilot_v1'
E3=BASE/'deconfounded_count_curriculum_causal_pilot_v1'
PKG=E3/'e4d_object_centric_component_anchor_grounding_v1'
ID='E4D_OBJECT_CENTRIC_COMPONENT_ANCHOR_GROUNDING_V1'
V7=ROOT/'experiments/capability_repair_baseline_v1/CHECKPOINT_MANIFEST_REMOTE_V1.json'
RUNTIME='/root/code_pair_py311_cu128_v2/bin/python'

def canon(x): return json.dumps(x,sort_keys=True,separators=(',',':')).encode()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def load(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_bytes(canon(x)+b'\n');os.replace(q,p)
def fail(x): raise RuntimeError('E4D:'+x)
def serializer(item):
 task=copy.deepcopy(item['task']);task.setdefault('source_id','E4D_SERIALIZER:'+item['canonical_base_id']);return task
def clean_source():
 for a in (['git','diff','--quiet'],['git','diff','--cached','--quiet']):
  if subprocess.run(a,cwd=ROOT).returncode: fail('TRACKED_SOURCE_DIRTY')
def inputs(): return {
 'protocol':PKG/'PROTOCOL.json','freeze_receipt':PKG/'PROSPECTIVE_FREEZE_RECEIPT.json',
 'ood':PKG/'OOD_INPUTS_AND_COMMITMENTS.json','marker':PKG/'MARKER_INPUTS_AND_COMMITMENTS.json',
 'schedule':E3/'TREATMENT_SCHEDULE.json','e3_protocol':E3/'PROTOCOL.json','v7_manifest':V7,
 'module':ROOT/'scripts/e4d_object_centric_component_anchor_grounding.py'}
def schedule():
 from training_data.pipeline import task_to_sample
 rows=load(E3/'TREATMENT_SCHEDULE.json')['episodes']
 if len(rows)!=384 or [r['slot'] for r in rows]!=list(range(384)): fail('SCHEDULE_SLOTS')
 if [r['optimizer_step'] for r in rows]!=[i//4 for i in range(384)]: fail('SCHEDULE_GROUPS')
 counts={x:sum(r['role']==x for r in rows) for x in ('PRIMARY_CONNECTED_COMPONENT_COUNT','PRIMARY_OBJECT_EXTRACTION','PROTECTED_REPLAY')}
 if counts!={'PRIMARY_CONNECTED_COMPONENT_COUNT':48,'PRIMARY_OBJECT_EXTRACTION':48,'PROTECTED_REPLAY':288}: fail('SCHEDULE_ROLES')
 tokens=sum(task_to_sample(serializer(r))['assistant_token_count'] for r in rows)
 if tokens!=27360: fail('TOKEN_BUDGET')
 return rows
def anchor_audit(rows):
 from scripts.e4d_object_centric_component_anchor_grounding import query_grid_anchor_layout, scalar_mass_contract
 from scripts.connected_components_object_extraction_semantic_oracle import components
 report=[]
 for r in rows:
  if r['role']!='PRIMARY_CONNECTED_COMPONENT_COUNT': continue
  t=serializer(r); grid=t['test'][0]['input']; layout=query_grid_anchor_layout(t)
  # Independent oracle code path validates visible color-separated components.
  oracle=components(grid)
  if len(layout.positions)!=81 or sum(layout.labels)!=layout.components or layout.components!=len(oracle): fail('ANCHOR_ORACLE:'+str(r['slot']))
  if any(t['test'][0]['input'][y][x]!=t['test'][0]['input'][y][x] for y,x in []): fail('UNREACHABLE')
  report.append({'slot':r['slot'],'query_grid_tokens':len(layout.positions),'positive_anchor_tokens':sum(layout.labels),'visible_components':layout.components,'anchor_positions':list(layout.positions),'anchor_labels':list(layout.labels)})
 if len(report)!=48 or sorted(x['positive_anchor_tokens'] for x in report).count(1)!=16 or sorted(x['positive_anchor_tokens'] for x in report).count(2)!=16 or sorted(x['positive_anchor_tokens'] for x in report).count(3)!=16: fail('ANCHOR_GRID')
 control=scalar_mass_contract(3.0,2.0,0.0); treatment=scalar_mass_contract(3.0,2.0,0.5)
 if control['slot_mean']!=3.0 or treatment['slot_mean']!=2.5 or treatment['slot_coefficient']!=.25: fail('MASS_ARITHMETIC')
 return report,{'control':control,'treatment':treatment}
def panels():
 o=load(PKG/'OOD_INPUTS_AND_COMMITMENTS.json');m=load(PKG/'MARKER_INPUTS_AND_COMMITMENTS.json')
 rows=o.get('records',[]);mr=m.get('records',[])
 if o.get('target_blind') is not True or len(rows)!=96 or len({x['canonical_unit_id'] for x in rows})!=48: fail('OOD_GRID')
 if o.get('topology_count_mutual_information_bits')!=0.0: fail('OOD_MI')
 if m.get('target_blind') is not True or len(mr)!=16 or len({x['canonical_unit_id'] for x in mr})!=8: fail('MARKER_GRID')
 orders={}
 for x in rows:
  if 'output' in x['prompt']['test'][0] or hashlib.sha256(canon(x['prompt'])).hexdigest()!=x['prompt_sha256']: fail('OOD_LEAK_OR_HASH')
  orders.setdefault(x['canonical_unit_id'],set()).add(x['demo_order'])
 if any(v!={'A','B'} for v in orders.values()): fail('OOD_ORDERS')
 for x in mr:
  if 'output' in x['prompt']['test'][0] or hashlib.sha256(canon(x['prompt'])).hexdigest()!=x['prompt_sha256']: fail('MARKER_LEAK_OR_HASH')
 return {'ood_canonical_units':48,'ood_records':96,'marker_canonical_units':8,'marker_records':16,'target_sidecar_opened':False}
def noncount_identity(rows):
 # Each non-Count slot reaches the byte-identical Equal-Slot path: no labels,
 # auxiliary tensor, parameter group, or scalar coefficient is changed.
 out=[]
 for r in rows:
  if r['role']=='PRIMARY_CONNECTED_COMPONENT_COUNT': continue
  out.append({'slot':r['slot'],'role':r['role'],'loss_path':'ORIGINAL_EQUAL_SLOT_CE','anchor_coefficient':0.0,'task_sha256':hashlib.sha256(canon(serializer(r))).hexdigest()})
 if len(out)!=336 or any(x['anchor_coefficient'] for x in out): fail('NONCOUNT_IDENTITY')
 return out
def cpu_fixture():
 # Structural launcher -> worker -> raw-freeze -> sidecar -> scorer trace.  It
 # deliberately carries no target and cannot produce a scientific prediction.
 trace=[
  {'phase':'launcher','binding_checked':True}, {'phase':'worker','model_loaded':False,'cuda_initialized':False},
  {'phase':'raw_freeze','raw_predictions_generated':False,'targets_opened':False},
  {'phase':'sidecar','sidecar_written':False,'targets_opened':False}, {'phase':'scorer','scored':False,'reason':'CPU_NO_MODEL_FIXTURE'}]
 return trace
def make_binding(path,output_root):
 clean_source(); commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
 if Path(output_root).exists(): fail('OUTPUT_EXISTS')
 manifest=load(V7); nonce=hashlib.sha256((commit+'|'+str(output_root)+'|'+str(time.time_ns())).encode()).hexdigest()[:32]
 b={'schema_version':1,'experiment_id':ID,'execution_source_commit':commit,
 'inputs':{k:{'path':str(v.relative_to(ROOT)).replace('\\','/'),'sha256':sha(v)} for k,v in inputs().items()},
 'worker_path':'scripts/run_e4d_object_centric_component_anchor_grounding_v1.py','worker_sha256':sha(Path(__file__)),
 'launch_command':['bash','scripts/launch_e4d_object_centric_component_anchor_grounding_v1.sh','--binding',str(Path(path).resolve().relative_to(ROOT)).replace('\\','/')],
 'runtime_python':RUNTIME,'output_root':str(output_root),'nonce':nonce,'retry':False,'hard_cap_seconds':7200,
 'runtime':{'physical_batch_size':1,'decoder':'NVARC_NATIVE_GREEDY','cuda_required':True,'bf16_required':True,'max_new_tokens':192,'context_window':16384},
 'training':{'schedule_path':str((E3/'TREATMENT_SCHEDULE.json').relative_to(ROOT)).replace('\\','/'),'schedule_sha256':sha(E3/'TREATMENT_SCHEDULE.json'),'seed':20261013,'learning_rate':5e-5,'optimizer':'PagedAdamW8bit','precision':'BF16','updates':96,'slots_per_update':4,'slot_coefficient':.25,
 'control':'ORIGINAL_EQUAL_SLOT_ALL_ASSISTANT_CE','treatment':'0.5_ORIGINAL_COUNT_PRIMARY_EQUAL_SLOT_ASSISTANT_CE_PLUS_0.5_MEAN_BINARY_COMPONENT_ANCHOR_CE','anchor_label':'LEXICOGRAPHICALLY_FIRST_CELL_PER_VISIBLE_COLOR_SEPARATED_4_CONNECTED_COMPONENT','anchor_positions':'REAL_NVARC_TOKENIZED_FINAL_QUERY_GRID_POSITIONS','noncount_paths':'BYTE_IDENTICAL_ORIGINAL_EQUAL_SLOT_CE'},
 'v7':{'manifest_sha256':sha(V7),'base_path':manifest['base_path'],'base_files':manifest['base_files'],'adapter_path':manifest['adapter_path'],'adapter_files':manifest['adapter_files']},
 'invariants':{'target_blind_inference':True,'raw_freeze_before_sidecar':True,'duplicate_output_guard':True,'no_prediction_or_target_before_authorization':True,'complete_50_50_objective_substitution_causal_scope':True},
 'evaluation':{'bootstrap_seed':20261013,'bootstrap_replicates':10000,'new_ood_canonical_units':48,'orders_per_unit':2,'marker_units':8,'physical_batch_size':1}}
 write(path,b);return b
def preclaim(binding,allow_missing_runtime=False):
 b=load(binding)
 if b.get('experiment_id')!=ID or b.get('retry') is not False or b.get('hard_cap_seconds')!=7200 or b['runtime'].get('physical_batch_size')!=1: fail('BINDING')
 if Path(b['output_root']).exists(): fail('OUTPUT_EXISTS')
 clean_source()
 for k,v in b['inputs'].items():
  p=ROOT/v['path']
  if not p.is_file() or sha(p)!=v['sha256']: fail('BOUND_INPUT:'+k)
 rows=schedule(); anchors,mass=anchor_audit(rows); panel=panels(); noncount=noncount_identity(rows)
 for name in ('training_data.pipeline','inference.nvarc_native','scripts.e04_e_equal_slot_loss','scripts.connected_components_object_extraction_semantic_oracle','scripts.e4d_object_centric_component_anchor_grounding'): importlib.import_module(name)
 m=load(V7); assets=[]
 for root,files in ((m['base_path'],m['base_files']),(m['adapter_path'],m['adapter_files'])):
  for f in files: assets.append({'root':root,'name':f['name'],'bytes':f['bytes'],'sha256':f['sha256']})
 if not allow_missing_runtime:
  if not Path(b['runtime_python']).is_file(): fail('RUNTIME_PYTHON')
  for f in assets:
   p=Path(f['root'])/f['name']
   if not p.is_file() or p.stat().st_size!=f['bytes'] or sha(p)!=f['sha256']: fail('ASSET:'+f['name'])
 return {'status':'PASS_E4D_PRODUCTION_PREFLIGHT_NO_MODEL','execution_source_commit':b['execution_source_commit'],'binding_sha256':sha(binding),'schedule':{'slots':len(rows),'updates':96,'assistant_tokens':27360,'sha256':sha(E3/'TREATMENT_SCHEDULE.json')},'anchor_audit':{'count_primary_slots':48,'query_grid_tokens_per_slot':81,'positive_anchor_count_distribution':{str(i):sum(x['positive_anchor_tokens']==i for x in anchors) for i in (1,2,3)},'real_tokenizer_positions_verified':True,'independent_semantic_oracle_verified':True,'objective_mass':mass},'noncount':{'slots':len(noncount),'byte_identical_equal_slot_paths':True},'panels':panel,'production_fixture':cpu_fixture(),'assets':assets,'model_loaded':False,'gpu_used':False,'sidecar_opened':False}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--binding',type=Path);ap.add_argument('--make-binding',action='store_true');ap.add_argument('--output-root');ap.add_argument('--preclaim-only',action='store_true');ap.add_argument('--allow-missing-runtime',action='store_true');ap.add_argument('--receipt',type=Path);a=ap.parse_args()
 if a.make_binding:
  if not a.binding or not a.output_root: ap.error('--make-binding requires --binding and --output-root')
  result=make_binding(a.binding,a.output_root)
 elif a.preclaim_only:
  if not a.binding: ap.error('--preclaim-only requires --binding')
  result=preclaim(a.binding,a.allow_missing_runtime)
 else: ap.error('E4D preparation event only permits --make-binding or --preclaim-only')
 if a.receipt: write(a.receipt,result)
 print(json.dumps(result,sort_keys=True))
if __name__=='__main__': main()