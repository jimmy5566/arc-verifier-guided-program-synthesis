import hashlib, importlib.util, json, subprocess, sys, tempfile, unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def load(name, file):
 s=importlib.util.spec_from_file_location(name,ROOT/'scripts'/file);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
WORKER=load('worker','run_paired_fixed_b32_fit_transfer_v1.py')
ANALYZER=load('analyzer','analyze_paired_fixed_b32_fit_transfer_v1.py')

class ExecutionContractTest(unittest.TestCase):
 def test_family_aligned_classifier_changes_with_frozen_dev_counts(self):
  delta={f:-.1 for f in ANALYZER.WEAK}
  self.assertEqual(ANALYZER.classify(delta,{f:0 for f in ANALYZER.WEAK}),'H3_SUPPORTED_CANDIDATE')
  dev={f:0 for f in ANALYZER.WEAK};dev['width']=1
  self.assertEqual(ANALYZER.classify(delta,dev),'INCONCLUSIVE')
  self.assertEqual(ANALYZER.classify({'width':-.1},dev),'H1_SUPPORTED_CANDIDATE')
 def test_b1_b32_reversal_is_inconclusive(self):
  self.assertFalse(WORKER.conclusion_stable({'width':-.1},{'width':.1},'H3_SUPPORTED_CANDIDATE','H3_SUPPORTED_CANDIDATE'))
  self.assertFalse(WORKER.conclusion_stable({'width':-.1},{'width':-.1},'H3_SUPPORTED_CANDIDATE','H1_SUPPORTED_CANDIDATE'))
  delta={f:-.1 for f in ANALYZER.WEAK};dev={f:0 for f in ANALYZER.WEAK};reversed_delta=dict(delta);reversed_delta['width']=.1
  self.assertEqual(ANALYZER.resolve_decision(delta,delta,reversed_delta,dev),'NUMERICALLY_INCONCLUSIVE')
 def test_authorization_noncontinue_and_mismatch_fail(self):
  with tempfile.TemporaryDirectory() as d:
   d=Path(d); config=d/'config.json';config.write_text('{}')
   bad=d/'bad.json';bad.write_text(json.dumps({'decision':'PAUSED','bindings':{}}))
   with self.assertRaisesRegex(RuntimeError,'AUTHORIZATION_DECISION_NOT_CONTINUE'):WORKER.validate_authorization(bad,config)
   mismatch=d/'mismatch.json';mismatch.write_text(json.dumps({'decision':'CONTINUE_CONTROLLER','bindings':{}}))
   with self.assertRaisesRegex(RuntimeError,'AUTHORIZATION_BINDING_MISMATCH'):WORKER.validate_authorization(mismatch,config)
 def test_stale_output_fails_before_model_work(self):
  with tempfile.TemporaryDirectory() as d:
   d=Path(d);out=d/'out';out.mkdir();config=d/'config.json';config.write_text('{}');auth=d/'auth.json';auth.write_text('{}')
   with self.assertRaisesRegex(RuntimeError,'FRESH_OUTPUT_REQUIRED'):WORKER.execute(config,{},d/'train.jsonl',auth,out)
 def test_hard_cap_emits_timeout_receipt(self):
  with tempfile.TemporaryDirectory() as d:
   receipt=Path(d)/'receipt.json'
   p=subprocess.run([sys.executable,str(ROOT/'scripts/arc2_hard_cap_launcher.py'),'--cap-seconds','1','--receipt',str(receipt),'--',sys.executable,'-c','import time; time.sleep(2)'],capture_output=True,text=True,timeout=15)
   self.assertNotEqual(p.returncode,0);self.assertEqual(json.loads(receipt.read_text())['status'],'RUNTIME_CAP_REACHED')
 def test_package_binding_verifier_rejects_bound_byte_drift(self):
  with tempfile.TemporaryDirectory() as d:
   d=Path(d); package=d/'package.json'; target='scripts/analyze_paired_fixed_b32_fit_transfer_v1.py'
   good={'authoritative_classifier':{'mutually_exclusive_rules':['a','b','c','d']},'execution':{'worker_path':target,'worker_sha256':hashlib.sha256((ROOT/target).read_bytes()).hexdigest(),'tested_source_hashes':{target:hashlib.sha256((ROOT/target).read_bytes()).hexdigest()}}}
   package.write_text(json.dumps(good))
   p=subprocess.run([sys.executable,str(ROOT/'scripts/verify_paired_fixed_b32_package_bindings.py'),'--package',str(package)],capture_output=True,text=True)
   self.assertEqual(p.returncode,0,p.stderr)
   good['execution']['tested_source_hashes'][target]='0'*64;package.write_text(json.dumps(good))
   self.assertNotEqual(subprocess.run([sys.executable,str(ROOT/'scripts/verify_paired_fixed_b32_package_bindings.py'),'--package',str(package)],capture_output=True,text=True).returncode,0)

if __name__=='__main__':unittest.main()
