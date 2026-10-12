import hashlib, importlib.util, json, tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def load(name):
 p=ROOT/'scripts'/name;s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
L=load('launch_connected_components_object_extraction_external_auth_v3.py')
W=load('run_connected_components_object_extraction_pilot_v1.py')
SOURCE=ROOT/'experiments/capability_repair_baseline_v1/connected_components_object_extraction_capability_pilot_v1/execution_binding_v3/EXECUTION_BINDING_EXTERNAL_AUTH_V3.json'
def receipt(binding,output):
 b=json.loads(binding.read_text());payload={'binding_sha256':L.sha(binding),'execution_source_commit':b['execution_source_commit'],'nonce':b['output']['nonce'],'output_root':output.as_posix(),'detached_jobs':1,'retry':False,'training_cap_seconds':b['runtime']['training_cap_seconds'],'evaluation_cap_seconds':b['runtime']['evaluation_cap_seconds'],'combined_cap_seconds':b['runtime']['combined_cap_seconds'],'forbidden_targets':b['target_isolation']['forbidden']};p=output.parent/'auth.json';p.write_text(json.dumps({'state':'UNCONSUMED','payload':payload,'payload_sha256':hashlib.sha256(L.canon(payload)).hexdigest()}));return p
def current_binding(out):
 b=json.loads(SOURCE.read_text());b['output']['remote_output_root']=out.as_posix()
 for pathkey,shakey in [('worker_path','worker_sha256'),('launcher_path','launcher_sha256'),('score_entrypoint_path','score_entrypoint_sha256')]:b[shakey]=L.sha(ROOT/b[pathkey])
 return b
def test_external_auth_receipt_gates_and_claims_once():
 with tempfile.TemporaryDirectory() as raw:
  d=Path(raw);out=d/'out';binding=d/'binding.json';binding.write_text(json.dumps(current_binding(out)));r=receipt(binding,out)
  assert L.validate(binding,r,out)[0]['schema_version']==3
  tampered=json.loads(r.read_text());tampered['payload']['nonce']='different';r.write_text(json.dumps(tampered))
  try:L.validate(binding,r,out);assert False
  except RuntimeError as e:assert 'RECEIPT_' in str(e)
  r=receipt(binding,out);assert L.claim(r,binding,out).exists()
  try:L.claim(r,binding,out);assert False
  except RuntimeError as e:assert 'ALREADY_CLAIMED' in str(e)
  out.mkdir()
  try:L.validate(binding,r,out);assert False
  except RuntimeError as e:assert 'OUTPUT_GUARD' in str(e)

def test_launcher_keeps_worker_output_fresh_and_records_control_receipts(monkeypatch):
 with tempfile.TemporaryDirectory() as raw:
  d=Path(raw);out=d/'out';binding=d/'binding.json';binding.write_text(json.dumps(current_binding(out)));r=receipt(binding,out)
  class Child: pid=12345
  seen=[]
  def fake_popen(args,**kwargs):seen.append((args,kwargs));return Child()
  monkeypatch.setattr(L.subprocess,'Popen',fake_popen)
  control,pid=L.launch(binding,r,out)
  assert pid==12345 and not out.exists() and (control/'LAUNCH_RECEIPT.json').exists() and (control/'REMOTE_JOB.json').exists()
  assert seen[0][0][-1]=='--execute'

def test_worker_writes_terminal_receipt_when_early_freshness_gate_fails():
 with tempfile.TemporaryDirectory() as raw:
  d=Path(raw);out=d/'out';out.mkdir();binding=d/'binding.json';binding.write_text(json.dumps(current_binding(out)))
  try: W.execute(binding,out);assert False
  except RuntimeError as e: assert 'FRESH_OUTPUT_REQUIRED' in str(e)
  terminal=json.loads((out/'TERMINAL_RECEIPT.json').read_text())
  assert terminal['status']=='FAILED_OR_PARTIAL' and terminal['targets_accessed'] is False
