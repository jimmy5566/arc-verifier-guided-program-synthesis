"""Production closure for the frozen post-V7 deconfounded Count pilot.

The preclaim path validates the exact production inputs without importing a
model library.  Targets are derived only after the raw-freeze receipt exists.
"""
from __future__ import annotations
import argparse, hashlib, importlib, json, os, random, subprocess, sys, time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
for _p in (ROOT,ROOT/'src'):
 if str(_p) not in sys.path: sys.path.insert(0,str(_p))
BASE=ROOT/'experiments/capability_repair_baseline_v1/connected_components_object_extraction_capability_pilot_v1'
PKG=BASE/'deconfounded_count_curriculum_causal_pilot_v1'
ID='POST_V7_DECONFOUNDED_COUNT_CURRICULUM_CAUSAL_PILOT_V1'
V7=ROOT/'experiments/capability_repair_baseline_v1/CHECKPOINT_MANIFEST_REMOTE_V1.json'
RUNTIME='/root/code_pair_py311_cu128_v2/bin/python'

def canon(x): return json.dumps(x,sort_keys=True,separators=(',',':')).encode()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def load(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_bytes(canon(x)+b'\n');os.replace(q,p)
def fail(x): raise RuntimeError('E3:'+x)
def task_id(t): return hashlib.sha256(canon(t)).hexdigest()
def clean_source():
 if subprocess.run(['git','status','--porcelain'],cwd=ROOT,capture_output=True,text=True).stdout: fail('SOURCE_DIRTY')

def inputs():
 return {
  'protocol':PKG/'PROTOCOL.json','manifest':PKG/'MANIFEST.json','cpu_validation':PKG/'CPU_VALIDATION.json',
  'control_schedule':PKG/'CONTROL_SCHEDULE.json','treatment_schedule':PKG/'TREATMENT_SCHEDULE.json',
  'train_prompts':PKG/'TRAIN_FIT_INPUT_PROMPTS.json',
  'ood':BASE/'post_v7_generalization_mechanism_audit_v1/FRESH_STRUCTURAL_OOD_INPUTS_AND_COMMITMENTS.json',
  'protected':BASE/'measurement_contract_v1/PROTECTED_CAPABILITY_INPUT_PROMPTS.json',
  'composition':BASE/'scientific_successor_v2/COMPOSITION_SENTINEL_INPUT_PROMPTS.json','v7_manifest':V7,
 }

def make_binding(path,output_root):
 clean_source(); commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
 protocol=load(PKG/'PROTOCOL.json'); assert protocol['experiment_id']==ID
 b={'schema_version':1,'experiment_id':ID,'scientific_package_commit':'a984e926191fa60c8ca0d40b00fadead35cc89c2','execution_source_commit':commit,
    'inputs':{k:{'path':str(v.relative_to(ROOT)).replace('\\','/'),'sha256':sha(v)} for k,v in inputs().items()},
    'worker_path':'scripts/run_post_v7_deconfounded_count_causal_pilot_v1.py','worker_sha256':sha(Path(__file__)),
    'output_root':str(output_root),'retry':False,'hard_cap_seconds':7200,'runtime_python':RUNTIME,
    'runtime':{'base_path':load(V7)['base_path'],'adapter_path':load(V7)['adapter_path'],'context_window':16384,'requested_batch_size':32,'fallback_ladder':[32,16,8,4,1],'max_new_tokens':192},
    'training':{'updates_per_arm':96,'slots_per_update':4,'loss':'ORIGINAL_EQUAL_SLOT_ALL_ASSISTANT_CE','seed':20261013,'learning_rate':5e-5},
    'nonce':hashlib.sha256((commit+str(output_root)+sha(PKG/'MANIFEST.json')+str(time.time_ns())).encode()).hexdigest()[:32],
    'sidecar_rule':'POST_RAW_FREEZE_ONLY_SYNTHETIC_TARGET_DERIVATION'}
 write(path,b);return b

def schedules_and_inputs(b):
 from training_data.pipeline import task_to_sample
 c,t=load(PKG/'CONTROL_SCHEDULE.json'),load(PKG/'TREATMENT_SCHEDULE.json')
 for name,x in [('control',c),('treatment',t)]:
  e=x['episodes'];
  if len(e)!=384 or [r['slot'] for r in e]!=list(range(384)):fail('SCHEDULE:'+name)
  if len({r['optimizer_step'] for r in e})!=96 or any(r['role']=='PRIMARY_CONNECTED_COMPONENT_COUNT' and 'source_id' not in r['task'] for r in e):fail('SCHEDULE_CONTENT:'+name)
  if sum(task_to_sample(r['task'])['assistant_token_count'] for r in e)!=27360:fail('TOKEN_BUDGET:'+name)
 for l,r in zip(c['episodes'],t['episodes'],strict=True):
  if l['role']!='PRIMARY_CONNECTED_COMPONENT_COUNT' and canon(l)!=canon(r):fail('NON_COUNT_DRIFT')
 return c,t

def preclaim(binding_path):
 b=load(binding_path)
 if b.get('experiment_id')!=ID or b.get('retry') is not False or b.get('worker_sha256')!=sha(Path(__file__)):fail('BINDING')
 if Path(b['output_root']).exists():fail('OUTPUT_EXISTS')
 clean_source()
 for key,e in b['inputs'].items():
  p=ROOT/e['path']
  if not p.is_file() or sha(p)!=e['sha256']:fail('INPUT:'+key)
 if load(PKG/'CPU_VALIDATION.json').get('status')!='PASS_CPU_ONLY_SCIENTIFIC_PACKAGE_VALIDATION':fail('CPU_VALIDATION')
 schedules_and_inputs(b)
 for name in ('inference.nvarc_native','training_data.pipeline','scripts.e04_e_equal_slot_loss','scripts.connected_components_object_extraction_semantic_oracle'):
  importlib.import_module(name)
 if not Path(b['runtime_python']).is_file():fail('RUNTIME_PYTHON')
 m=load(V7)
 for root,files in ((m['adapter_path'],m['adapter_files']),(m['base_path'],m['base_files'])):
  for f in files:
   p=Path(root)/f['name']
   if not p.is_file() or p.stat().st_size!=f['bytes'] or sha(p)!=f['sha256']:fail('ASSET:'+f['name'])
 gpu=subprocess.run(['nvidia-smi','--query-gpu=memory.total','--format=csv,noheader'],capture_output=True,text=True)
 if gpu.returncode or int(gpu.stdout.split()[0])<24576:fail('GPU_CAPACITY')
 return {'status':'PASS_E3_PRODUCTION_SHAPED_PRECLAIM_NO_MODEL','binding_sha256':sha(binding_path),'model_loaded':False,'gpu_used':False,'sidecar_opened':False,'schedules':{'control':384,'treatment':384}}

def records(arm=None):
 out=[]
 ood=load(inputs()['ood'])
 for r in ood['records']: out.append({'id':r['record_id'],'split':'OOD','canonical':r['canonical_unit_id'],'order':r['demo_order'],'task':r['prompt']})
 for r in load(inputs()['protected'])['records']:out.append({'id':'PROTECTED:'+r['canonical_base_id']+':'+r['condition']+':'+str(r['control_marker_turn']),'split':'RETENTION','task':r['prompt'],'protected':r})
 for r in load(inputs()['composition']):out.append({'id':'COMPOSITION:'+r['prompt_id'],'split':'COMPOSITION','task':r['task']})
 if arm:
  for r in load(PKG/'TRAIN_FIT_INPUT_PROMPTS.json')['records']:
   if r['arm']==arm:out.append({'id':r['record_id'],'split':'TRAIN_FIT','task':r['prompt']})
 if any('output' in x['task']['test'][0] for x in out):fail('TARGET_LEAK')
 return out

def raw_write(path,rows):
 p=Path(path);p.write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows),encoding='utf-8');return sha(p)

def execute(binding_path):
 pre=preclaim(binding_path);b=load(binding_path);out=Path(b['output_root']);out.mkdir(parents=True);started=time.monotonic();deadline=started+b['hard_cap_seconds'];write(out/'PREFLIGHT_RECEIPT.json',pre)
 try:
  from scripts import run_connected_components_object_extraction_pilot_v1 as real
  translated={'v7':{'checkpoint_manifest_path':str(V7.relative_to(ROOT)).replace('\\','/')},'training':{'torch_seed':b['training']['seed'],'learning_rate':b['training']['learning_rate']},'runtime':{'combined_cap_seconds':b['hard_cap_seconds'],'context_window':b['runtime']['context_window']},'evaluation':{'fallback_ladder':b['runtime']['fallback_ladder'],'max_new_tokens':b['runtime']['max_new_tokens']}}
  raw={}
  def generate(model,torch,manifest,items,tag):
   return real._generate(model,torch,manifest,[{'record_id':x['id'],'split':x['split'],'task':x['task']} for x in items],translated,out,deadline,tag)
  v,torch,m=real._load_v7(translated,trainable=False);raw['V7']=generate(v,torch,m,records(),'V7');del v
  for arm,fn in [('CONTROL_ORIGINAL_CONFOUNDED','CONTROL_SCHEDULE.json'),('TREATMENT_DECONFOUNDED','TREATMENT_SCHEDULE.json')]:
   model,torch,m,adapter=real._train(translated,arm,PKG/fn,out/arm,deadline);raw[arm]=generate(model,torch,m,records(arm),arm);write(out/arm/'ADAPTER_RECEIPT.json',{'adapter_sha256':adapter,'optimizer_steps':96});del model
  write(out/'RAW_FREEZE_RECEIPT.json',{'status':'COMPLETE_RAW_FROZEN_NO_TARGETS','raw_sha256':raw,'targets_accessed':False,'nonce':b['nonce']})
  sidecar=make_sidecar(binding_path,out);result=score(binding_path,out,sidecar)
  write(out/'TERMINAL_RECEIPT.json',{'status':'COMPLETE','result_sha256':sha(out/'SCORE_RECEIPT.json'),'raw_freeze_before_sidecar':True})
 except Exception as e:
  write(out/'TERMINAL_RECEIPT.json',{'status':'FAILED_OR_PARTIAL_NO_RETRY','error':type(e).__name__+':'+str(e),'raw_predictions_created':(out/'RAW_FREEZE_RECEIPT.json').exists()});raise

def make_sidecar(binding_path,out):
 if not (Path(out)/'RAW_FREEZE_RECEIPT.json').is_file():fail('RAW_FREEZE_REQUIRED')
 from scripts.connected_components_object_extraction_semantic_oracle import target_for_family
 from scripts import e04_orientation_fixed_demonstration_baseline_v3 as orient
 bases={x['canonical_base_id']:x for x in orient.v2.canonical_bases('VALIDATION')}
 targets={}
 for r in records('CONTROL_ORIGINAL_CONFOUNDED')+records('TREATMENT_DECONFOUNDED'):
  if r['split'] in {'TRAIN_FIT','OOD','COMPOSITION'}: targets[r['id']]=target_for_family('CONNECTED_COMPONENT_COUNT' if r['split'] in {'TRAIN_FIT','OOD'} else 'COUNT_THEN_OBJECT_EXTRACTION',r['task']['test'][0]['input'])
  elif r['split']=='RETENTION':
   q=r['protected'];targets[r['id']]=orient.target(bases[q['canonical_base_id']],q['control_marker_turn'],q['condition'],q['source_row_index'])['target']
 commitments={x['record_id']:x['target_sha256'] for x in load(PKG/'TRAIN_FIT_INPUT_PROMPTS.json')['target_commitments']}
 commitments|={x['record_id']:x['target_sha256'] for x in load(inputs()['ood'])['target_commitments']}
 for k,v in commitments.items():
  if k not in targets or hashlib.sha256(canon(targets[k])).hexdigest()!=v:fail('TARGET_COMMITMENT:'+k)
 p=Path(out)/'SYNTHETIC_TARGET_SIDECAR.json';write(p,{'status':'DERIVED_AFTER_RAW_FREEZE','targets':targets,'commitments_verified':len(commitments)});return p

def score(binding_path,out,sidecar):
 b=load(binding_path);freeze=load(Path(out)/'RAW_FREEZE_RECEIPT.json')
 if freeze.get('status')!='COMPLETE_RAW_FROZEN_NO_TARGETS' or freeze.get('targets_accessed') is not False:fail('FREEZE')
 targets=load(sidecar)['targets'];scored={}
 for tag in ('V7','CONTROL_ORIGINAL_CONFOUNDED','TREATMENT_DECONFOUNDED'):
  rows=[json.loads(x) for x in (Path(out)/(tag+'_RAW.jsonl')).read_text().splitlines() if x];scored[tag]={r['record_id']:{**r,'exact':bool(r.get('parser_valid')) and r.get('parsed_grid')==targets.get(r['record_id'])} for r in rows}
 c,t,v=scored['CONTROL_ORIGINAL_CONFOUNDED'],scored['TREATMENT_DECONFOUNDED'],scored['V7'];units=sorted({r['canonical'] for r in records() if r['split']=='OOD'})
 diff=[int(all(t[x['id']]['exact'] for x in records('TREATMENT_DECONFOUNDED') if x['split']=='OOD' and x['canonical']==u))-int(all(c[x['id']]['exact'] for x in records('CONTROL_ORIGINAL_CONFOUNDED') if x['split']=='OOD' and x['canonical']==u)) for u in units]
 rng=random.Random(b['training']['seed']);draws=sorted(sum(rng.choice(diff) for _ in diff) for _ in range(10000));delta=sum(diff);lo,hi=draws[249],draws[9749]
 fit=sum(z['exact'] for k,z in t.items() if k.startswith('TREATMENT_DECONFOUNDED:COUNT_TRAIN_FIT:'))
 treatment_any=[u for u in units if any(t[x['id']]['exact'] for x in records('TREATMENT_DECONFOUNDED') if x['split']=='OOD' and x['canonical']==u)]
 inv=sum(all(t[x['id']]['exact'] for x in records('TREATMENT_DECONFOUNDED') if x['split']=='OOD' and x['canonical']==u) for u in treatment_any)
 retention={}
 for cond in sorted({x['protected']['condition'] for x in records() if x['split']=='RETENTION'}):
  ids=[x['id'] for x in records() if x['split']=='RETENTION' and x['protected']['condition']==cond]
  retention[cond]={'control_minus_v7':sum(c[i]['exact']-v[i]['exact'] for i in ids),'treatment_minus_v7':sum(t[i]['exact']-v[i]['exact'] for i in ids),'rows':len(ids)}
 comp=[x['id'] for x in records() if x['split']=='COMPOSITION'];composition={'control':sum(c[i]['exact'] for i in comp),'treatment':sum(t[i]['exact'] for i in comp),'treatment_minus_control':sum(t[i]['exact']-c[i]['exact'] for i in comp),'rows':len(comp)}
 gates={'primary':delta>=6 and lo>0,'train_fit':fit>=43,'order_invariance':not treatment_any or inv/len(treatment_any)>=.8,'retention':all(x['control_minus_v7']>=-1 and x['treatment_minus_v7']>=-1 for x in retention.values()),'composition':composition['treatment_minus_control']>=-1}
 decision='PROMOTE' if all(gates.values()) else ('REJECT' if not gates['train_fit'] else 'INCONCLUSIVE')
 r={'status':'COMPLETE_E3_SCORE_AFTER_RAW_FREEZE','canonical_units':48,'treatment_minus_control':{'delta_correct':delta,'bootstrap_95_ci':[lo,hi],'replicates':10000},'train_fit_treatment':fit,'order_invariance':{'both_orders_correct':inv,'any_order_correct':len(treatment_any)},'retention':retention,'composition':composition,'gates':gates,'decision':decision,'targets_opened_after_raw_freeze':True}
 write(Path(out)/'SCORE_RECEIPT.json',r);return r

def main():
 p=argparse.ArgumentParser();p.add_argument('--binding',type=Path,required=True);p.add_argument('--output-root');p.add_argument('--make-binding',action='store_true');p.add_argument('--preclaim-only',action='store_true');p.add_argument('--execute',action='store_true');a=p.parse_args()
 if a.make_binding: print(json.dumps(make_binding(a.binding,a.output_root),sort_keys=True))
 elif a.preclaim_only: print(json.dumps(preclaim(a.binding),sort_keys=True))
 elif a.execute: execute(a.binding)
 else: raise SystemExit('select mode')
if __name__=='__main__':main()
