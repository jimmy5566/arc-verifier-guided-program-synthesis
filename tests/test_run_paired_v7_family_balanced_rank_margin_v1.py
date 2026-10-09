from __future__ import annotations
import hashlib,json,subprocess,sys,tempfile,unittest
from pathlib import Path
from scripts.run_paired_v7_family_balanced_rank_margin_v1 import first_free_running_error, select_raw_pairs, validate_prompt_reconstruction
class WorkerEntryTests(unittest.TestCase):
 def test_missing_binding_fails_before_model_import(self):
  with tempfile.TemporaryDirectory() as d:
   r=subprocess.run([sys.executable,'scripts/run_paired_v7_family_balanced_rank_margin_v1.py','--binding',str(Path(d)/'missing'),'--output',str(Path(d)/'o'),'--receipt',str(Path(d)/'r')],capture_output=True,text=True)
   self.assertNotEqual(r.returncode,0);self.assertIn('LAUNCH_BINDING_MISSING',r.stderr)
 def test_invalid_binding_schema_fails_before_model_import(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'b.json';p.write_text('{}');r=subprocess.run([sys.executable,'scripts/run_paired_v7_family_balanced_rank_margin_v1.py','--binding',str(p),'--output',str(Path(d)/'o'),'--receipt',str(Path(d)/'r')],capture_output=True,text=True)
   self.assertNotEqual(r.returncode,0);self.assertIn('LAUNCH_BINDING_SCHEMA_INVALID',r.stderr)
 def test_identity_mismatch_writes_no_update_failure_before_model_import(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d); paths={}
   for key in ('raw_predictions','input_manifest','sealed_sidecar','v7_manifest','family_balanced_manifest'):
    path=root/f'{key}.json';path.write_text('{}',encoding='utf8');paths[key]=str(path)
   binding={'protocol_id':'PAIRED_V7_FAMILY_BALANCED_CORRECT_TOKEN_RANK_MARGIN_V1','runtime_cap_seconds':900,'checkpoints':{},'input_paths':paths,'expected_hashes':{key:hashlib.sha256(Path(value).read_bytes()).hexdigest() for key,value in paths.items()}}
   binding['expected_hashes']['raw_predictions']='0'*64
   bp=root/'binding.json';bp.write_text(json.dumps(binding),encoding='utf8');output=root/'fresh'/'result';receipt=root/'fresh'/'TERMINAL_RECEIPT.json'
   r=subprocess.run([sys.executable,'scripts/run_paired_v7_family_balanced_rank_margin_v1.py','--binding',str(bp),'--output',str(output),'--receipt',str(receipt)],capture_output=True,text=True)
   self.assertNotEqual(r.returncode,0);self.assertIn('LAUNCH_IDENTITY_MISMATCH',r.stderr)
   saved=json.loads(receipt.read_text(encoding='utf8'));self.assertEqual(saved['status'],'FAILED_NO_UPDATE');self.assertEqual(saved['optimizer_steps'],0);self.assertFalse(saved['generation'])
 def test_first_error_uses_each_checkpoint_own_frozen_continuation(self):
  row={'target_ids':[1,2,15],'free_running':{'RECONSTRUCTED_FOUNDATION_V2_V7':{'generated_token_ids':[1,2,15],'parse_valid':True},'FAMILY_BALANCED':{'generated_token_ids':[1,9,15],'parse_valid':False,'parse_reason':'INVALID_ARC_GRID'}}}
  self.assertEqual(first_free_running_error(row,'RECONSTRUCTED_FOUNDATION_V2_V7'),(None,'PARSE_VALID'))
  self.assertEqual(first_free_running_error(row,'FAMILY_BALANCED'),(1,'PARSE_INVALID:INVALID_ARC_GRID'))
 def test_all_60_frozen_native_prompts_reconstruct_byte_exactly(self):
  root=Path(__file__).resolve().parents[1]
  raw=root/'experiments/capability_repair_baseline_v1/unified_native_model_capability_baseline_v1/unified_native_model_capability_baseline_v3_batch16/run_001_5e46a8ba7794cbec09d838c84d26a32a/RAW_UNSCORED_BATCH16.jsonl'
  manifest=root/'experiments/capability_repair_baseline_v1/unified_native_model_capability_baseline_v1/SYNTHETIC_BENCHMARK_INPUT_MANIFEST_V1.json'
  validate_prompt_reconstruction(json.loads(manifest.read_text(encoding='utf8')),select_raw_pairs(raw))
if __name__=='__main__':unittest.main()
