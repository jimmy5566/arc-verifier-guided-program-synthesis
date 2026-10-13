from __future__ import annotations
import argparse,hashlib,json,os,subprocess,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
 h=hashlib.sha256();
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def fail(x):raise RuntimeError('IC_V2_PRECLAIM:'+x)
def load(p):return json.loads(Path(p).read_text())
def verify(path,entry):
 p=ROOT/entry['path']
 if not p.is_file() or sha(p)!=entry['sha256']:fail('INPUT:'+path)
 return p
def validate(binding_path):
 b=load(binding_path); req={'experiment_id','scientific_source_commit','source_files','inputs','arms','v7','runtime','execution','training','evaluation','dataset_paths','inference','scoring'}
 if not req<=set(b):fail('BINDING_SCHEMA')
 if b['experiment_id']!='CONNECTED_COMPONENT_COUNT_CODE_PAIR_INDEPENDENT_CONFIRMATION_V2':fail('EXPERIMENT')
 for rel,h in b['source_files'].items():
  if not (ROOT/rel).is_file() or sha(ROOT/rel)!=h:fail('SOURCE:'+rel)
 obs=verify('observation',b['inputs']['observation']); recipes=verify('recipes',b['inputs']['recipes']); b1=verify('batch1',b['inputs']['batch1']); table=verify('decision_table',b['inputs']['decision_table']); schedule=verify('schedule',b['inputs']['diverse_schedule'])
 if b['arms']['control']['schedule_sha256']!=b['arms']['treatment']['schedule_sha256'] or b['arms']['control']['schedule_sha256']!=b['inputs']['diverse_schedule']['sha256']:fail('SCHEDULE')
 if b['arms']['control']['loss_allocation']!={'kind':'UNIFORM_SEQUENCE_LOSS'} or b['arms']['treatment']['loss_allocation']!={'kind':'COUNT_CODE_PAIR_REWEIGHTED','decision_token_multiplier':2.0}:fail('LOSS')
 manifest=ROOT/b['v7']['checkpoint_manifest_path']
 if not manifest.is_file() or sha(manifest)!=b['v7']['checkpoint_manifest_sha256']:fail('V7_MANIFEST')
 asset_manifest=load(manifest); verified=0
 for root_key,files_key in (('base_path','base_files'),('adapter_path','adapter_files')):
  asset_root=Path(asset_manifest[root_key])
  for asset in asset_manifest[files_key]:
   asset_path=asset_root/asset['name']
   if not asset_path.is_file() or asset_path.stat().st_size!=asset['bytes'] or sha(asset_path)!=asset['sha256']:fail('V7_ASSET:'+asset['name'])
   verified+=1
 ex=b['execution']; needed={'authorization_required','remote_output_root','cpu_integration_root','terminal_receipt','nonce','retry','worker_path','launcher_path'}
 if not needed<=set(ex) or ex['authorization_required'] is not True or ex['retry'] is not False:fail('EXECUTION')
 rows=load(obs); expect={'CONNECTED_COMPONENT_COUNT':24,'OBJECT_EXTRACTION':24,'FIXED_TURN_ROTATION_CONTROL':8,'MARKER_BINDING_CONTROL':8,'NO_TRANSFORM_RETENTION':24,'COUNT_THEN_OBJECT_EXTRACTION':16};got={}
 for r in rows:
  if 'output' in r['task']['test'][0]:fail('TARGET_LEAK')
  got[r['latent_condition']['family']]=got.get(r['latent_condition']['family'],0)+1
 if got!=expect:fail('INVENTORY')
 ids={r['prompt_id'] for r in rows}; subset=load(b1)['checkpoints']
 if set(subset)!={'V7','CONTROL_NARROW_STRUCTURE','TREATMENT_STRUCTURE_DIVERSE'} or any(len(x)!=12 or len(set(x))!=12 or not set(x)<=ids for x in subset.values()):fail('B1')
 if not all(x==subset['V7'] for x in subset.values()):fail('B1_SHARED')
 return b,rows,{'inventory':got,'stable_record_ids':['V2:'+x for x in subset['V7']],'v7_asset_files_verified':verified,'sealed_sidecar_opened':False}
def run_cpu(binding,out):
 b,rows,r=validate(binding);out=Path(out)
 if out.exists():fail('FRESH_OUTPUT')
 out.mkdir(parents=True); cmd=[sys.executable,str(ROOT/'scripts/score_connected_component_count_code_pair_independent_confirmation_v2.py'),'--binding',str(binding),'--validate-only'];c=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True)
 if c.returncode:fail('SCORER:'+c.stderr+c.stdout)
 rec={'status':'PASS_V2_LAUNCHER_WORKER_SCORER_CPU_INTEGRATION_STOP_BEFORE_MODEL','binding_sha256':sha(binding),'records':len(rows),'inventory':r['inventory'],'b1_record_ids':r['stable_record_ids'],'scorer_validation':json.loads(c.stdout),'model_loaded':False,'gpu_used':False,'cuda_initialized':False,'sealed_sidecar_opened':False,'authorization_claimed':False};(out/'CPU_INTEGRATION_RECEIPT.json').write_text(json.dumps(rec,sort_keys=True,separators=(',',':'))+'\n');return rec
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--binding',type=Path,required=True);a.add_argument('--output-root',type=Path,required=True);a.add_argument('--preclaim-only',action='store_true');a.add_argument('--cpu-integration',action='store_true');x=a.parse_args()
 x.binding=x.binding.resolve(); x.output_root=x.output_root.resolve()
 if x.preclaim_only==x.cpu_integration:raise SystemExit('choose one')
 if x.preclaim_only:
  b,rows,r=validate(x.binding);print(json.dumps({'status':'PASS_V2_PRODUCTION_PRECLAIM_STOP_BEFORE_MODEL','records':len(rows),**r,'model_loaded':False,'gpu_used':False,'sealed_sidecar_opened':False,'authorization_claimed':False},sort_keys=True))
 else:print(json.dumps(run_cpu(x.binding,x.output_root),sort_keys=True))
