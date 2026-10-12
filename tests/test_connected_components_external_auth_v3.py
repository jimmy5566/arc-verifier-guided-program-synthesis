import hashlib, importlib.util, json, tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def load(name):
 p=ROOT/'scripts'/name;s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
L=load('launch_connected_components_object_extraction_external_auth_v3.py')
SOURCE=ROOT/'experiments/capability_repair_baseline_v1/connected_components_object_extraction_capability_pilot_v1/execution_binding_v3/EXECUTION_BINDING_EXTERNAL_AUTH_V3.json'
def receipt(binding,output):
 b=json.loads(binding.read_text());payload={'binding_sha256':L.sha(binding),'execution_source_commit':b['execution_source_commit'],'nonce':b['output']['nonce'],'output_root':output.as_posix(),'detached_jobs':1,'retry':False,'training_cap_seconds':b['runtime']['training_cap_seconds'],'evaluation_cap_seconds':b['runtime']['evaluation_cap_seconds'],'combined_cap_seconds':b['runtime']['combined_cap_seconds'],'forbidden_targets':b['target_isolation']['forbidden']};p=output.parent/'auth.json';p.write_text(json.dumps({'state':'UNCONSUMED','payload':payload,'payload_sha256':hashlib.sha256(L.canon(payload)).hexdigest()}));return p
def test_external_auth_receipt_gates_and_claims_once():
 with tempfile.TemporaryDirectory() as raw:
  d=Path(raw);out=d/'out';binding=d/'binding.json';b=json.loads(SOURCE.read_text());b['output']['remote_output_root']=out.as_posix();binding.write_text(json.dumps(b));r=receipt(binding,out)
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
