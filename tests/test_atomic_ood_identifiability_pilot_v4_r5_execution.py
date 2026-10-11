from __future__ import annotations
import importlib.util,json,subprocess,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/capability_repair_baseline_v1/atomic_ood_identifiability_pilot_v4_three_checkpoint_native_greedy_no_update_r5'
def load(rel,name):
 s=importlib.util.spec_from_file_location(name,ROOT/rel); m=importlib.util.module_from_spec(s); assert s.loader; s.loader.exec_module(m); return m
def worker(): return load('scripts/run_atomic_ood_identifiability_pilot_v4_three_checkpoint_native_greedy_r5.py','v4r2')
def test_preimport_cpu_only_and_r5_identity():
 r=worker().validate_preimport(); assert r['status']=='PASS_V4R5_PREIMPORT' and r['model_imported'] is False and r['prompt_rows']==192
def test_preimport_never_stats_or_hashes_sealed_sidecar():
 w=worker(); original_sha=w.sha; original_is_file=Path.is_file
 def guarded_sha(path):
  if Path(path).name=='SEALED_TARGET_SIDECAR.jsonl': raise AssertionError('sealed sidecar hashed before raw freeze')
  return original_sha(path)
 def guarded_is_file(path):
  if path.name=='SEALED_TARGET_SIDECAR.jsonl': raise AssertionError('sealed sidecar statted before raw freeze')
  return original_is_file(path)
 w.sha=guarded_sha; Path.is_file=guarded_is_file
 try:
  assert w.validate_preimport()['status']=='PASS_V4R5_PREIMPORT'
 finally:
  w.sha=original_sha; Path.is_file=original_is_file

def test_length_bucketing_is_deterministic_and_applied_to_primary_rows():
 w=worker(); prompts=w.read_jsonl(ROOT/'experiments/capability_repair_baseline_v1/atomic_ood_identifiability_pilot_v4/INPUT_PROMPTS.jsonl')
 order=w.length_bucketed_primary(prompts,lambda row: len(row['episode_id']) % 3)
 assert [x['episode_id'] for x in order]==[x['episode_id'] for x in sorted(prompts,key=lambda row:(len(row['episode_id']) % 3,row['episode_id']))]
 def fake(arm,rows,batch): return [{'generated_token_ids':[1],'decoded_text':'x','parser_valid':False,'parsed_grid':None,'prompt_tokens':1} for _ in rows]
 with tempfile.TemporaryDirectory() as d:
  w.collect_with_generator(OUT/'EXECUTION_PACKAGE.json',Path(d),fake,primary_rows=order)
  frozen=w.read_jsonl(Path(d)/'V7_RAW_TARGET_BLIND.jsonl')
  assert [x['episode_id'] for x in frozen if x['phase']=='PRIMARY']==[x['episode_id'] for x in order]

def test_corrupt_runtime_identity_components_fail_before_model_import():
 w=worker()
 with tempfile.TemporaryDirectory() as d:
  root=Path(d); base=root/'base'; adapter=root/'adapter'; config=root/'configs'/'native'; base.mkdir(parents=True); adapter.mkdir(); config.mkdir(parents=True)
  (base/'base.bin').write_bytes(b'base'); (adapter/'adapter.bin').write_bytes(b'adapter'); (config/'tokenizer.json').write_bytes(b'tokenizer')
  manifest={'base_path':str(base),'adapter_path':str(adapter),'base_files':[{'name':'base.bin','bytes':4,'sha256':w.sha(base/'base.bin')}],'adapter_files':[{'name':'adapter.bin','bytes':7,'sha256':w.sha(adapter/'adapter.bin')}]}
  (root/'manifest.json').write_text(json.dumps(manifest)); identity={'runtime_config_dir_name':'native','files':[{'runtime_relative_path':'tokenizer.json','bytes':9,'sha256':w.sha(config/'tokenizer.json')}]}
  (root/'native.json').write_text(json.dumps(identity)); package={'three_checkpoints':[{'base_manifest_path':'manifest.json','base_manifest_sha256':w.sha(root/'manifest.json')}],'native_runtime':{'config_identity_path':'native.json','config_identity_sha256':w.sha(root/'native.json')}}
  old_root=w.ROOT; w.ROOT=root
  try:
   bad=json.loads(json.dumps(package)); bad['three_checkpoints'][0]['base_manifest_sha256']='0'*64
   try: w.validate_runtime_identity(bad)
   except w.V4ExecutionFailure as e: assert str(e)=='V4R5_BASE_MANIFEST_IDENTITY'
   else: raise AssertionError('corrupt base manifest accepted')
   bad=json.loads(json.dumps(package)); bad_manifest=json.loads((root/'manifest.json').read_text()); bad_manifest['adapter_files'][0]['sha256']='0'*64; (root/'manifest.json').write_text(json.dumps(bad_manifest)); bad['three_checkpoints'][0]['base_manifest_sha256']=w.sha(root/'manifest.json')
   try: w.validate_runtime_identity(bad)
   except w.V4ExecutionFailure as e: assert str(e)=='V4R5_V7_ADAPTER_IDENTITY:adapter.bin'
   else: raise AssertionError('corrupt V7 adapter accepted')
   (root/'manifest.json').write_text(json.dumps(manifest)); bad=json.loads(json.dumps(package)); bad['native_runtime']['config_identity_sha256']='0'*64
   try: w.validate_runtime_identity(bad)
   except w.V4ExecutionFailure as e: assert str(e)=='V4R5_NATIVE_CONFIG_IDENTITY'
   else: raise AssertionError('corrupt native config accepted')
  finally:
   w.ROOT=old_root
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
  except w.V4ExecutionFailure as e: assert str(e)=='V4R5_SOURCE_COMMIT'
  else: raise AssertionError('wrong commit accepted')
  binding['source_commit']=w.git(['rev-parse','HEAD'])
  original=w.git
  w.git=lambda args: 'M tracked.py' if args[0]=='status' else original(args)
  try:
   try: w.validate_preimport(pkg,output_root=output,binding=binding)
   except w.V4ExecutionFailure as e: assert str(e)=='V4R5_DIRTY_TRACKED_SOURCE'
   else: raise AssertionError('dirty tracked source accepted')
  finally: w.git=original
def test_cross_path_nonce_duplicate_is_refused():
 w=worker(); data=w.read_json(OUT/'EXECUTION_PACKAGE.json')
 with tempfile.TemporaryDirectory() as d:
  original=w.ROOT; w.ROOT=Path(d)
  try:
   first=w.claim_nonce(data,Path(d)/'a')
   try: w.claim_nonce(data,Path(d)/'b')
   except w.V4ExecutionFailure as e: assert str(e)=='V4R5_DUPLICATE_NONCE'
   else: raise AssertionError('duplicate nonce accepted')
   w.finalize_nonce(first,'FAILED')
  finally: w.ROOT=original
def test_outer_watchdog_writes_terminal_receipt_on_cap():
 launcher=ROOT/'scripts/launch_atomic_ood_identifiability_pilot_v4_three_checkpoint_native_greedy_r5.py'
 with tempfile.TemporaryDirectory() as d:
  d=Path(d); sleeper=d/'sleep.py'; sleeper.write_text('import time; time.sleep(5)')
  l=launcher_module(); original=l.preflight_runtime_import; l.preflight_runtime_import=lambda: None
  try: result=type('R',(),{'returncode':l.run(sleeper,sleeper,sleeper,d/'out',1)})()
  finally: l.preflight_runtime_import=original
  assert result.returncode==124
  receipt=json.loads((d/'out'/'TERMINAL_RECEIPT.json').read_text()); assert receipt['error']=='V4R5_OUTER_CAP_EXCEEDED' and receipt['target_sidecar_accessed'] is False

def test_launcher_leaves_fresh_path_for_worker_and_receipts_nonzero_child():
 launcher=ROOT/'scripts/launch_atomic_ood_identifiability_pilot_v4_three_checkpoint_native_greedy_r5.py'
 with tempfile.TemporaryDirectory() as d:
  d=Path(d); good=d/'good.py'
  good.write_text("import pathlib,sys; o=pathlib.Path(sys.argv[-1]); assert not o.exists(); o.mkdir(); (o/'TERMINAL_RECEIPT.json').write_text('{\"status\":\"COMPLETE\"}')")
  l=launcher_module(); original=l.preflight_runtime_import; l.preflight_runtime_import=lambda: None
  try:
   out=d/'fresh'; rc=l.run(good,good,good,out,5); assert rc==0 and (out/'TERMINAL_RECEIPT.json').exists()
   bad=d/'bad.py'; bad.write_text('raise SystemExit(7)'); out2=d/'failed'; rc=l.run(bad,bad,bad,out2,5); receipt=json.loads((out2/'TERMINAL_RECEIPT.json').read_text()); assert rc==7 and receipt['child_returncode']==7
  finally: l.preflight_runtime_import=original

def test_model_lifecycle_hook_runs_after_each_frozen_arm():
 w=worker(); released=[]
 def fake(arm,rows,batch): return [{'generated_token_ids':[1],'decoded_text':'x','parser_valid':False,'parsed_grid':None,'prompt_tokens':1} for _ in rows]
 with tempfile.TemporaryDirectory() as d:
  w.collect_with_generator(OUT/'EXECUTION_PACKAGE.json',Path(d),fake,after_arm=released.append)
 assert released==['V7','E04_E_CONTROL','E04_E_TREATMENT']


def launcher_module():
 return load('scripts/launch_atomic_ood_identifiability_pilot_v4_three_checkpoint_native_greedy_r5.py','v4r5_launcher')

def test_launcher_preflight_uses_production_cwd_and_src_path():
 l=launcher_module(); calls=[]; original=l.subprocess.run
 class Done: returncode=0; stdout=''; stderr=''
 def fake(args,**kwargs): calls.append((args,kwargs)); return Done()
 l.subprocess.run=fake
 try: l.preflight_runtime_import()
 finally: l.subprocess.run=original
 args,kwargs=calls.pop()
 assert args[:2]==[sys.executable,'-c'] and 'inference.nvarc_native' in args[2]
 assert Path(kwargs['cwd'])==ROOT
 assert kwargs['env']['PYTHONPATH'].split(__import__('os').pathsep)[0]==str(ROOT/'src')

def test_launcher_import_failure_prevents_worker_and_nonce_claim():
 l=launcher_module()
 with tempfile.TemporaryDirectory() as d:
  d=Path(d); sentinel=d/'worker_started'; worker=d/'worker.py'
  worker.write_text(f"from pathlib import Path; Path({str(sentinel)!r}).write_text('started')")
  original=l.preflight_runtime_import; l.preflight_runtime_import=lambda: (_ for _ in ()).throw(RuntimeError('forced'))
  try: code=l.run(worker,worker,worker,d/'fresh',5)
  finally: l.preflight_runtime_import=original
  receipt=json.loads((d/'fresh'/'TERMINAL_RECEIPT.json').read_text())
  assert code==125 and receipt['error']=='V4R5_IMPORT_PREFLIGHT_FAILED'
  assert not sentinel.exists()

def test_launcher_passes_same_root_cwd_and_env_to_worker():
 l=launcher_module(); seen=[]
 class Child:
  returncode=0
  def poll(self): return 0
  def kill(self): raise AssertionError('unexpected kill')
  def wait(self): return 0
 original_probe=l.preflight_runtime_import; original_popen=l.subprocess.Popen
 l.preflight_runtime_import=lambda: None
 def fake_popen(args,**kwargs): seen.append((args,kwargs)); return Child()
 l.subprocess.Popen=fake_popen
 try:
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'x'; assert l.run(p,p,p,Path(d)/'out',5)==0
 finally: l.preflight_runtime_import=original_probe; l.subprocess.Popen=original_popen
 args,kwargs=seen.pop(); assert Path(kwargs['cwd'])==ROOT
 assert kwargs['env']['PYTHONPATH'].split(__import__('os').pathsep)[0]==str(ROOT/'src')
