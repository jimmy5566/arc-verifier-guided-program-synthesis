from __future__ import annotations
import importlib.util,json,subprocess,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/capability_repair_baseline_v1/atomic_ood_identifiability_pilot_v4_three_checkpoint_native_greedy_no_update_r4'
def load(rel,name):
 s=importlib.util.spec_from_file_location(name,ROOT/rel); m=importlib.util.module_from_spec(s); assert s.loader; s.loader.exec_module(m); return m
def worker(): return load('scripts/run_atomic_ood_identifiability_pilot_v4_three_checkpoint_native_greedy_r4.py','v4r2')
def test_preimport_cpu_only_and_r4_identity():
 r=worker().validate_preimport(); assert r['status']=='PASS_V4R4_PREIMPORT' and r['model_imported'] is False and r['prompt_rows']==192
def test_fake_generator_collects_primary_and_distinct_b1_rows_with_raw_identities():
 w=worker()
 def fake(arm,rows,batch): return [{'generated_token_ids':[1,2,3],'decoded_text':'1 2\\n3 4','parser_valid':True,'parsed_grid':[[1,2],[3,4]],'prompt_tokens':7} for _ in rows]
 with tempfile.TemporaryDirectory() as d:
  result=w.collect_with_generator(OUT/'EXECUTION_PACKAGE.json',Path(d),fake)
  assert result['raw_rows']==612 and result['b1_distinct_rows']==36
  rows=w.read_jsonl(Path(d)/'V7_RAW_TARGET_BLIND.jsonl'); b1=[x for x in rows if x['phase']=='B1_SENSITIVITY']
  assert len(rows)==204 and len(b1)==12 and all(x['effective_batch_size']==1 for x in b1)
  assert set(b1[0]) >= {'generated_token_ids','decoded_text','parser_valid','parsed_grid','canonical_grid_sha256','prompt_tokens','completion_tokens'}
def test_wrong_commit_and_dirty_source_fail_preimport():
 w=worker(); pkg=OUT/'EXECUTION_PACKAGE.json'; data=w.read_json(pkg)
 with tempfile.TemporaryDirectory() as d:
  output=Path(d)/'new'; binding={'package_sha256':w.sha(pkg),'worker_sha256':data['worker_sha256'],'launcher_sha256':data['launcher_sha256'],'authorization':'DIRECTOR_ONCE','retry':False,'nonce':data['execution_output']['nonce'],'output_root':str(output),'active_hard_cap_seconds':7200,'source_commit':'0'*40}
  try: w.validate_preimport(pkg,output_root=output,binding=binding)
  except w.V4ExecutionFailure as e: assert str(e)=='V4R4_SOURCE_COMMIT'
  else: raise AssertionError('wrong commit accepted')
  binding['source_commit']=w.git(['rev-parse','HEAD'])
  original=w.git
  w.git=lambda args: 'M tracked.py' if args[0]=='status' else original(args)
  try:
   try: w.validate_preimport(pkg,output_root=output,binding=binding)
   except w.V4ExecutionFailure as e: assert str(e)=='V4R4_DIRTY_TRACKED_SOURCE'
   else: raise AssertionError('dirty tracked source accepted')
  finally: w.git=original
def test_cross_path_nonce_duplicate_is_refused():
 w=worker(); data=w.read_json(OUT/'EXECUTION_PACKAGE.json')
 with tempfile.TemporaryDirectory() as d:
  original=w.ROOT; w.ROOT=Path(d)
  try:
   first=w.claim_nonce(data,Path(d)/'a')
   try: w.claim_nonce(data,Path(d)/'b')
   except w.V4ExecutionFailure as e: assert str(e)=='V4R4_DUPLICATE_NONCE'
   else: raise AssertionError('duplicate nonce accepted')
   w.finalize_nonce(first,'FAILED')
  finally: w.ROOT=original
def test_outer_watchdog_writes_terminal_receipt_on_cap():
 launcher=ROOT/'scripts/launch_atomic_ood_identifiability_pilot_v4_three_checkpoint_native_greedy_r4.py'
 with tempfile.TemporaryDirectory() as d:
  d=Path(d); sleeper=d/'sleep.py'; sleeper.write_text('import time; time.sleep(5)')
  result=subprocess.run([sys.executable,str(launcher),'--worker',str(sleeper),'--package',str(sleeper),'--runtime-binding',str(sleeper),'--output-root',str(d/'out'),'--cap','1'])
  assert result.returncode==124
  receipt=json.loads((d/'out'/'TERMINAL_RECEIPT.json').read_text()); assert receipt['error']=='V4R4_OUTER_CAP_EXCEEDED' and receipt['target_sidecar_accessed'] is False

def test_launcher_leaves_fresh_path_for_worker_and_receipts_nonzero_child():
 launcher=ROOT/'scripts/launch_atomic_ood_identifiability_pilot_v4_three_checkpoint_native_greedy_r4.py'
 with tempfile.TemporaryDirectory() as d:
  d=Path(d); good=d/'good.py'
  good.write_text("import pathlib,sys; o=pathlib.Path(sys.argv[-1]); assert not o.exists(); o.mkdir(); (o/'TERMINAL_RECEIPT.json').write_text('{\"status\":\"COMPLETE\"}')")
  out=d/'fresh'; r=subprocess.run([sys.executable,str(launcher),'--worker',str(good),'--package',str(good),'--runtime-binding',str(good),'--output-root',str(out),'--cap','5']); assert r.returncode==0 and (out/'TERMINAL_RECEIPT.json').exists()
  bad=d/'bad.py'; bad.write_text('raise SystemExit(7)'); out2=d/'failed'; r=subprocess.run([sys.executable,str(launcher),'--worker',str(bad),'--package',str(bad),'--runtime-binding',str(bad),'--output-root',str(out2),'--cap','5']); receipt=json.loads((out2/'TERMINAL_RECEIPT.json').read_text()); assert r.returncode==7 and receipt['child_returncode']==7

def test_model_lifecycle_hook_runs_after_each_frozen_arm():
 w=worker(); released=[]
 def fake(arm,rows,batch): return [{'generated_token_ids':[1],'decoded_text':'x','parser_valid':False,'parsed_grid':None,'prompt_tokens':1} for _ in rows]
 with tempfile.TemporaryDirectory() as d:
  w.collect_with_generator(OUT/'EXECUTION_PACKAGE.json',Path(d),fake,after_arm=released.append)
 assert released==['V7','E04_E_CONTROL','E04_E_TREATMENT']
