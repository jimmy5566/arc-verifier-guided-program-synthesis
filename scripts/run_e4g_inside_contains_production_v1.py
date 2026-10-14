"""E4G production entrypoint; CUDA imports are deferred until --execute."""
from __future__ import annotations
import argparse, hashlib, importlib, json, os, subprocess, sys
from pathlib import Path
ROOT=Path(os.environ.get('ARC2_ROOT',Path(__file__).resolve().parents[1]))
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
PKG=ROOT/'experiments/capability_repair_baseline_v1/connected_components_object_extraction_capability_pilot_v1/deconfounded_count_curriculum_causal_pilot_v1/e4g_inside_contains_atomic_plasticity_v1'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,x):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(x,sort_keys=True,separators=(',',':'))+'\n',encoding='utf-8')
def fail(x):raise RuntimeError('E4G:'+x)
def resolve(binding):
 b=load(binding)
 if b['experiment_id']!='E4G_INSIDE_CONTAINS_ATOMIC_PLASTICITY_POSITIVE_CONTROL_V1' or b['retry'] is not False or b['hard_cap_seconds']!=7200:fail('BINDING')
 if subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()!=b['execution_source_commit']:fail('HEAD')
 for name,v in b['inputs'].items():
  p=ROOT/v['path']
  if not p.is_file() or sha(p)!=v['sha256']:fail('INPUT_'+name)
 p=load(ROOT/b['inputs']['ood']['path'])
 if len(p['records'])!=96 or len({r['canonical_unit_id'] for r in p['records']})!=48 or any('output' in r['input_task']['test'][0] for r in p['records']):fail('TARGET_BLIND_OOD')
 s=load(ROOT/b['inputs']['schedule']['path'])['episodes']; roles={k:sum(r['role']==k for r in s) for k in ('PRIMARY_CONNECTED_COMPONENT_COUNT','PRIMARY_OBJECT_EXTRACTION','PROTECTED_REPLAY')}
 if roles!={'PRIMARY_CONNECTED_COMPONENT_COUNT':48,'PRIMARY_OBJECT_EXTRACTION':48,'PROTECTED_REPLAY':288}:fail('SCHEDULE')
 for m in ('training_data.pipeline','inference.nvarc_native','scripts.e04_e_equal_slot_loss','scripts.e4g_inside_contains_atomic','scripts.score_e4g_inside_contains_v1'):importlib.import_module(m)
 return b,roles
def cpu(binding):
 b,roles=resolve(binding); out=Path(binding).parent/'E4G_CPU_WORKER_RECEIPT.json'
 write(out,{'status':'PASS_E4G_WORKER_CPU_PRECLAIM_NO_MODEL','binding_sha256':sha(binding),'roles':roles,'deferred_imports':'PASS','targets_opened':False,'model_loaded':False,'gpu_used':False,'output_root':b['output_root']});return out
def execute(binding):
 # Execution-only imports are deliberately unreachable in CPU validation.
 b,roles=resolve(binding); import torch
 if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():fail('CUDA_BF16')
 # The established Equal-Slot primitive is the only training loss; this call
 # is reached only after a bound runtime preclaim and uses the frozen schedule.
 from training_data.pipeline import task_to_sample
 from scripts.e04_e_equal_slot_loss import causal_slot_mean_ce
 schedule=load(ROOT/b['inputs']['schedule']['path'])['episodes']
 if len(schedule)!=384 or not all(task_to_sample(r['task'])['assistant_token_count']>0 for r in schedule):fail('SERIALIZER')
 # Model construction/training/native generation are intentionally kept in the
 # bound runtime launcher, after the scientific boundary receipt is written.
 write(Path(b['output_root'])/'SCIENTIFIC_BOUNDARY_RECEIPT.json',{'status':'MODEL_LOAD_AUTHORIZED','retry':False,'roles':roles})
 return {'status':'E4G_EXECUTION_ENTRY_VALIDATED','model_loaded':False}
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--binding',required=True);ap.add_argument('--preclaim-only',action='store_true');ap.add_argument('--cpu-self-test',action='store_true');ap.add_argument('--execute',action='store_true');a=ap.parse_args()
 if a.cpu_self_test or a.preclaim_only: print(cpu(a.binding))
 elif a.execute: print(execute(a.binding))
 else: ap.error('choose --preclaim-only, --cpu-self-test, or --execute')
