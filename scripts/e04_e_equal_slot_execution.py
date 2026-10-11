"""CPU schedule and identity contract for E04-E V2; no model imports."""
from __future__ import annotations
import hashlib, json, os, sys, tempfile
from pathlib import Path
from typing import Any
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from training_data.pipeline import IGNORE_INDEX, task_to_sample
BASE=ROOT/'experiments/capability_repair_baseline_v1/e04_e_equal_slot_weight_marker_replay_preservation_pilot_v2'
PROTOCOL_ID='E04_E_EQUAL_SLOT_WEIGHT_MARKER_REPLAY_PRESERVATION_PILOT_V2'
ARMS=('CONTROL_EQUAL_SLOT','TREATMENT_EQUAL_SLOT')
ARM_FILE={ARMS[0]:'CONTROL_SCHEDULE.json',ARMS[1]:'TREATMENT_SCHEDULE.json'}
PER_ARM_CAP_SECONDS=3600; JOINT_CAP_SECONDS=7200; STEP_SIZE=4
class E04EFailure(RuntimeError): pass
def fail(code:str)->None: raise E04EFailure(code)
def sha_path(path:Path)->str:
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()
def canon(x:Any)->bytes:return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()
def atomic_json(path:Path,value:Any)->None:
 path.parent.mkdir(parents=True,exist_ok=True);fd,tmp=tempfile.mkstemp(dir=path.parent,prefix=path.name+'.',suffix='.tmp')
 try:
  with os.fdopen(fd,'w',encoding='utf-8',newline='\n') as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
  os.replace(tmp,path)
 finally:
  if os.path.exists(tmp):os.unlink(tmp)
def read_json(path:Path)->dict[str,Any]:return json.loads(Path(path).read_text(encoding='utf-8'))
def relative(path:Path)->str:return path.resolve().relative_to(ROOT).as_posix()
def _schedules(): return {arm:read_json(BASE/ARM_FILE[arm]) for arm in ARMS}
def build_arm_samples(arm:str,schedule:dict[str,Any])->list[dict[str,Any]]:
 if arm not in ARMS or schedule.get('arm')!=arm:fail('E04E_ARM_IDENTITY')
 episodes=schedule.get('episodes')
 if not isinstance(episodes,list) or len(episodes)!=384:fail('E04E_SCHEDULE_ROWS')
 samples=[]
 for slot,row in enumerate(episodes):
  if row.get('slot')!=slot or row.get('optimizer_step')!=slot//STEP_SIZE or row.get('within_step_role_position')!=slot%STEP_SIZE:fail('E04E_SLOT_ORDER')
  task=row.get('task')
  if not isinstance(task,dict) or hashlib.sha256(canon(task)).hexdigest()!=row.get('task_sha256'):fail('E04E_TASK_HASH:'+str(slot))
  sample=task_to_sample({'source_id':f'e04e-{arm.lower()}-{slot}',**task})
  if len(sample['input_ids'])!=len(sample['labels']) or sample['attention_mask']!=[1]*len(sample['input_ids']):fail('E04E_NATIVE_SHAPE:'+str(slot))
  supervised=sum(x!=IGNORE_INDEX for x in sample['labels'])
  if supervised!=sample['assistant_token_count'] or supervised<1:fail('E04E_NATIVE_LABELS:'+str(slot))
  samples.append({'slot':slot,'episode':row,'sample':sample,'supervised':supervised})
 return samples
def static_schedule_preflight()->dict[str,Any]:
 protocol=read_json(BASE/'E04_E_V2_PRELAUNCH_PROTOCOL.json'); accounting=read_json(BASE/'TOKEN_ACCOUNTING.json'); manifest=read_json(BASE/'MANIFEST.json'); schedules=_schedules()
 if protocol.get('protocol_id')!=PROTOCOL_ID or manifest.get('protocol_id')!=PROTOCOL_ID:fail('E04E_PROTOCOL_ID')
 for name in ('CONTROL_SCHEDULE.json','TREATMENT_SCHEDULE.json','TOKEN_ACCOUNTING.json'):
  expected={'CONTROL_SCHEDULE.json':protocol['schedule']['control_sha256'],'TREATMENT_SCHEDULE.json':protocol['schedule']['treatment_sha256'],'TOKEN_ACCOUNTING.json':protocol['schedule']['accounting_sha256']}[name]
  if sha_path(BASE/name)!=expected:fail('E04E_FROZEN_HASH:'+name)
 data={arm:build_arm_samples(arm,schedules[arm]) for arm in ARMS}
 counts={'rotation':0,'fixed':0,'swap':0}; marker_counts={}; totals={}
 for c,t in zip(data[ARMS[0]],data[ARMS[1]],strict=True):
  slot=c['slot']; ce,te=c['episode'],t['episode']
  if t['slot']!=slot or any(ce[k]!=te[k] for k in ('optimizer_step','within_step_role_position','role','replay_source','canonical_base_id','turn','train_cell')):fail('E04E_CROSS_ARM_METADATA:'+str(slot))
  if slot<96:
   counts['rotation']+=1
   if ce['task']!=te['task'] or ce['task_kind']!='FIXED_TURN_ROTATION_CONTROL' or te['task_kind']!='FIXED_TURN_ROTATION_CONTROL':fail('E04E_ROTATION_IDENTITY:'+str(slot))
  elif ce['replay_source']=='E04_C_BYTE_IDENTICAL_NO_TRANSFORM':
   counts['fixed']+=1
   if ce['task']!=te['task']:fail('E04E_FIXED_REPLAY_IDENTITY:'+str(slot))
  elif ce['replay_source']=='E04_D_MATCHED_MARKER_REPLAY_SWAP':
   counts['swap']+=1; p=ce.get('marker_pair_id')
   if ce['task_kind']!='NO_TRANSFORM_RETENTION_CONTROL' or te['task_kind']!='MARKER_BINDING_CONTROL' or not p or p!=te.get('marker_pair_id') or ce['task']==te['task']:fail('E04E_SWAP_IDENTITY:'+str(slot))
   marker_counts[p]=marker_counts.get(p,0)+1
  else:fail('E04E_REPLAY_SOURCE:'+str(slot))
 for arm,rows in data.items():
  step_positions=[{x['episode']['within_step_role_position'] for x in rows[i:i+STEP_SIZE]} for i in range(0,384,STEP_SIZE)]
  if any(x!={0,1,2,3} for x in step_positions):fail('E04E_STEP_ROLE_POSITIONS')
  total={'raw_transformer_tokens':sum(len(x['sample']['input_ids']) for x in rows),'raw_supervised_tokens':sum(x['supervised'] for x in rows),'slot_count':len(rows),'optimizer_steps':96}
  accounting_arm = 'CONTROL' if arm == ARMS[0] else 'TREATMENT'
  if any(total[k]!=accounting['arms'][accounting_arm][k] for k in total):fail('E04E_ACCOUNTING_DRIFT:'+arm)
  totals[arm]=total
 if counts!={'rotation':96,'fixed':192,'swap':96} or len(marker_counts)!=48 or set(marker_counts.values())!={2}:fail('E04E_ALLOCATION')
 return {'status':'PASS_E04E_CPU_SCHEDULE_AND_NATIVE_SERIALIZER','protocol_id':PROTOCOL_ID,'arms':list(ARMS),'slots_per_arm':384,'optimizer_steps_per_arm':96,'four_unique_role_positions_per_step':True,'allocation':counts,'marker_candidates_exactly_twice':True,'arm_totals':totals,'model_imported':False,'tokenizer_loaded':False,'optimizer_constructed':False,'target_sidecar_accessed':False,'final_audit_opened':False}
