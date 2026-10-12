#!/usr/bin/env python3
"""CPU-only end-to-end regression for the V7 localization production entrypoint."""
from __future__ import annotations
import hashlib, importlib.util, json, os, subprocess, sys, tempfile
from pathlib import Path
from unittest import mock

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("prod",ROOT/"scripts"/"v7_teacher_forced_batch_stability_localization_production_v1.py")
M=importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value): Path(path).write_bytes(json.dumps(value,sort_keys=True,separators=(',',':')).encode()+b'\n')

class FakeRuntime:
 def __init__(self, *, cross=False, repeat=False, mask=False, boom=False): self.cross=cross;self.repeat=repeat;self.mask=mask;self.boom=boom;self.calls=[];self.admitted=[]
 def check_dependencies(self): self.calls.append('dependencies')
 def verify_identities(self, identities): self.calls.append('identities')
 def prompt_token_ids(self,row): return [11]*(3+int(row['cohort_record_id'].split(':')[1])%5)
 def open_worker_only_sidecar(self, descriptor):
  assert descriptor['path'].startswith('C:\\unusable\\') and Path(descriptor['path']).exists() is False
  assert descriptor['remote_path'].startswith('/') and Path(descriptor['fixture_runtime_path']).exists()
  return json.loads(Path(descriptor['fixture_runtime_path']).read_text())['records']
 def teacher_forced(self,packed,**identity):
  self.calls.append((identity['requested_batch_size'],identity['repeat_identity'],packed['effective_shape']))
  if self.mask and identity['requested_batch_size']==2 and identity['repeat_identity']==1: packed['records'][0]['attention_mask']=[1]*len(packed['records'][0]['attention_mask'])
  if self.boom: raise RuntimeError('FAKE_WORKER_FAILURE')
  out={}
  for row in packed['records']:
   self.admitted.append(row['cohort_record_id'])
   logits=[[0,0,4,1],[0,0,1,4]]
   if (self.cross and identity['requested_batch_size']==8) or (self.repeat and identity['requested_batch_size']==4 and identity['repeat_identity']==1): logits=[[0,4,0,1],[0,0,1,4]]
   out[row['cohort_record_id']]=logits
  return out

def fixture(base):
 base=Path(base); base.mkdir(); protocol=base/'protocol.json'; cohort=base/'cohort.json'; subset=base/'subset.json'; sidecar=base/'sidecar.json'; binding=base/'binding.json'; auth=base/'auth.json'
 p={'schema_version':1,'protocol_id':'V7_TEACHER_FORCED_BATCH_STABILITY_LOCALIZATION_V1','scope':'BATCH_STABILITY_ONLY_NOT_CAPABILITY_ESTIMATION','measurement':{'physical_batch_sizes':[1,2,4,8,12],'minimum_repeats_per_condition':2,'records':12,'teacher_forced_only':True,'generation_calls':0,'optimizer_steps':0,'backward_calls':0},'fresh_authorization_required':{'gpu_execution':True},'within_condition_tolerance':{'rank_abs_delta_max':0,'top_flag_abs_delta_max':0,'margin_abs_delta_max':1e-6,'nll_abs_delta_max':1e-6},'cross_batch_tolerance':{'rank_abs_delta_max':0,'top_flag_abs_delta_max':0,'margin_abs_delta_max':.02,'nll_abs_delta_max':.005}}
 records=[{'cohort_record_id':f'R:{i:02d}','task':{'train':[],'test':[{'input':[[i]]}]}} for i in range(72)]
 write(protocol,p);write(cohort,{'records':records});write(subset,{'record_ids':[f'R:{i:02d}' for i in range(12)]});write(sidecar,{'records':[{'cohort_record_id':f'R:{i:02d}','target_token_ids':[2,3]} for i in range(72)]})
 identities={'checkpoint':{'adapter_model_sha256':'1639bf6665ccfbb99cf547f59fcd581663c95b6bad770ae7846fee90f66be1c3','adapter_model_path':'/unused'},'tokenizer':{'runtime_path':'/unused','files':{}},'runtime':{'native_config_tree_sha256':'f2701b5c1458d29a4be62933ef7bcd910ad9ae75a671cb55341a85e43896a17b','native_config_dir':'/unused'}}
 b={'protocol_id':'V7_TEACHER_FORCED_BATCH_STABILITY_LOCALIZATION_V1','status':'PREAUTH_CPU_REPAIRED_NOT_GPU_AUTHORIZED','execution':{'generation_calls':0,'optimizer_steps':0,'backward_calls':0,'retry':False,'physical_batch_sizes':[1,2,4,8,12],'minimum_repeats_per_condition':2,'records':12,'hard_cap_seconds':3600,'mandatory_watchdog':True},'protocol':{'path':str(protocol),'sha256':sha(protocol)},'cohort':{'path':str(cohort),'sha256':sha(cohort)},'batch1_subset':{'path':str(subset),'sha256':sha(subset)},'sidecar':{'path':'C:\\unusable\\sealed_sidecar.json','remote_path':'/workspace/arc2/sealed_sidecars/frozen.json','fixture_runtime_path':str(sidecar),'sha256':sha(sidecar),'access':'TEACHER_FORCED_WORKER_ONLY'},'identities':identities,'source_hashes':{'scripts/v7_teacher_forced_batch_stability_localization_production_v1.py':sha(ROOT/'scripts/v7_teacher_forced_batch_stability_localization_production_v1.py')}}
 write(binding,b);write(auth,{'status':'VALID_UNCONSUMED','binding_sha256':sha(binding)});return binding,auth

def run(base,**kw):
 binding,auth=fixture(base);out=Path(base)/'out';ctrl=Path(base)/'control';runtime=FakeRuntime(**kw);return M.launch(binding,runtime,auth,out,ctrl),runtime,out,ctrl,binding,auth
def expected(kind,base,**kw): return run(base,**kw)[0]['classification']==kind

def main():
 with tempfile.TemporaryDirectory() as tmp:
  root=Path(tmp)
  result,runtime,out,ctrl,binding,auth=run(root/'success')
  assert result['status']=='COMPLETE' and result['classification']=='STABLE_ACROSS_BATCHES' and result['records']==120
  assert set(runtime.admitted)=={f'R:{i:02d}' for i in range(12)} and len(runtime.admitted)==120
  rows=[json.loads(x) for x in (out/'LOCALIZATION_METRICS.jsonl').read_text().splitlines()]
  assert len(rows)==120 and all('target_token_ids' not in row and 'prompt_token_ids' not in row for row in rows)
  assert {call[2][0] for call in runtime.calls if isinstance(call,tuple) and isinstance(call[2],list)}=={1,2,4,8,12}
  assert expected('REPRODUCIBLE_BATCH_DEPENDENT_NUMERICAL_EFFECT',root/'cross',cross=True)
  assert expected('RUNTIME_INSTABILITY',root/'repeat',repeat=True)
  assert expected('IMPLEMENTATION_DEFECT',root/'mask',mask=True)
  # Original authorization claim blocks a second execution even with a fresh output root.
  try: M.launch(binding,FakeRuntime(),auth,root/'success'/'out2',root/'success'/'control2')
  except RuntimeError as exc: assert 'DUPLICATE_OR_CONSUMED_ONE_SHOT' in str(exc)
  else: raise AssertionError('duplicate guard absent')
  binding,auth=fixture(root/'failure');out=root/'failure'/'out';ctrl=root/'failure'/'control'
  try: M.launch(binding,FakeRuntime(boom=True),auth,out,ctrl)
  except RuntimeError as exc: assert 'FAKE_WORKER_FAILURE' in str(exc)
  else: raise AssertionError('failure path absent')
  assert json.loads((ctrl/'TERMINAL_RECEIPT.json').read_text())['status']=='FAILED_NO_UPDATE'
  watchdog=root/'timeout.json';assert M.V5.enforce_hard_cap([sys.executable,'-c','import time;time.sleep(1)'],watchdog,seconds=.01) is None
  assert json.loads(watchdog.read_text())['status']=='FAILED_NO_UPDATE_TIMEOUT'
  # Exact child CLI refuses execution unless its watchdog marker exists; mocked runtime makes the child path CPU-testable.
  binding,auth=fixture(root/'cli'); out=root/'cli'/'out';ctrl=root/'cli'/'control'; old=os.environ.get('ARC2_V7_BATCH_LOCALIZATION_WATCHDOG_CHILD');os.environ['ARC2_V7_BATCH_LOCALIZATION_WATCHDOG_CHILD']='1'
  try:
   with mock.patch.object(M,'ProductionRuntime',lambda _config:FakeRuntime()),mock.patch.object(sys,'argv',[str(M.__file__),'--binding',str(binding),'--authorization',str(auth),'--output-root',str(out),'--control-root',str(ctrl),'--execute-claimed']): M.main()
  finally:
   if old is None: os.environ.pop('ARC2_V7_BATCH_LOCALIZATION_WATCHDOG_CHILD',None)
   else: os.environ['ARC2_V7_BATCH_LOCALIZATION_WATCHDOG_CHILD']=old
  assert json.loads((ctrl/'TERMINAL_RECEIPT.json').read_text())['status']=='COMPLETE'
 print('PASS_V7_BATCH_LOCALIZATION_PRODUCTION_SHAPED_CPU_E2E')
if __name__=='__main__': main()
