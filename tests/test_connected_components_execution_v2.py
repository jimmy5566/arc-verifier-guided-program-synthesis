import hashlib, importlib.util, json, tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def load(name):
 p=ROOT/'scripts'/name;sp=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
freeze=load('freeze_connected_components_object_extraction_execution_binding_v2.py');worker=load('run_connected_components_object_extraction_pilot_v1.py');launcher=load('launch_connected_components_object_extraction_pilot_v1.py');scorer=load('score_connected_components_object_extraction_pilot_v1.py')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def test_binding_cpu_preflight_and_matching_arms():
 with tempfile.TemporaryDirectory() as d:
  b=freeze.freeze(Path(d)); p=Path(d)/'EXECUTION_BINDING_PREAUTH_V2.json'; assert worker.preflight(p)['protocol_id'].endswith('V2'); assert launcher.verify(p,None)['worker_sha256']==sha(ROOT/b['worker_path']); assert b['evaluation']['fallback_ladder']==[32,16,8,4,1]
 control=json.loads((ROOT/'experiments/capability_repair_baseline_v1/connected_components_object_extraction_capability_pilot_v1/CONTROL_SCHEDULE.json').read_text())['episodes']; treatment=json.loads((ROOT/'experiments/capability_repair_baseline_v1/connected_components_object_extraction_capability_pilot_v1/TREATMENT_SCHEDULE.json').read_text())['episodes']; assert len(control)==len(treatment)==384; assert [x['optimizer_step'] for x in control]==[x['optimizer_step'] for x in treatment]
def test_fresh_output_and_frozen_raw_scorer_boundary():
 with tempfile.TemporaryDirectory() as d:
  b=freeze.freeze(Path(d)/'binding'); p=Path(d)/'binding/EXECUTION_BINDING_PREAUTH_V2.json'; output=Path(d)/'output'; output.mkdir(); b=json.loads(p.read_text());b['output']['remote_output_root']=output.as_posix();p.write_text(json.dumps(b));
  try: worker.preflight(p,output,require_fresh=True); assert False
  except RuntimeError as e: assert 'FRESH_OUTPUT' in str(e)
  p.write_text(json.dumps(freeze.freeze(Path(d)/'other'))); output.mkdir(exist_ok=True)
  try: scorer.score(p,output); assert False
  except FileNotFoundError: pass
  raw=output/'V7_RAW.jsonl';raw.write_text(json.dumps({'record_id':'OOD:0','split':'OOD','parser_valid':True,'parsed_grid':[[1]],'generated_token_ids':[1]})+'\n'); fr={'status':'COMPLETE_RAW_FROZEN_NO_TARGETS','targets_accessed':False,'raw_sha256':{'V7':sha(raw)}};(output/'RAW_FREEZE_RECEIPT.json').write_text(json.dumps(fr)); r=scorer.score(p,output);assert r['by_artifact']['V7']['by_split']['OOD']['parser_valid']==1



def test_protected_and_trainfit_target_reconstruction_after_freeze():
 with tempfile.TemporaryDirectory() as d:
  b=freeze.freeze(Path(d)/'binding'); p=Path(d)/'binding/EXECUTION_BINDING_PREAUTH_V2.json'; output=Path(d)/'output';output.mkdir()
  base=ROOT/'experiments/capability_repair_baseline_v1/connected_components_object_extraction_capability_pilot_v1'
  prot=json.loads((base/'measurement_contract_v1/PROTECTED_CAPABILITY_INPUT_PROMPTS.json').read_text())['records'][0]
  raw=output/'V7_RAW.jsonl';raw.write_text(json.dumps({'record_id':'PROTECTED:'+prot['protected_id'],'split':'PROTECTED','parser_valid':False,'parsed_grid':None,'generated_token_ids':[]})+'\n')
  fr={'status':'COMPLETE_RAW_FROZEN_NO_TARGETS','targets_accessed':False,'raw_sha256':{'V7':sha(raw)}};(output/'RAW_FREEZE_RECEIPT.json').write_text(json.dumps(fr))
  r=scorer.score(p,output);assert r['by_artifact']['V7']['by_split']['PROTECTED']['exact']==0
