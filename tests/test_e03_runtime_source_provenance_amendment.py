import hashlib, json, os, tempfile, unittest
from pathlib import Path
from scripts.bootstrap_private_https_asset import fetch
ROOT=Path(__file__).resolve().parents[1]
E03=ROOT/'experiments/capability_repair_baseline_v1/e03_v7_lora_gradient_interference_diagnostic_v1'
class E03AmendmentTests(unittest.TestCase):
 def test_manifest_is_self_contained_and_provenance_bound(self):
  m=json.loads((E03/'E03_TRAIN_MICROBATCH_MANIFEST_V1.json').read_text(encoding='utf-8'))
  c=json.loads((E03/'E03_V7_LORA_GRADIENT_INTERFERENCE_DIAGNOSTIC_V1_CONFIG_V7.json').read_text(encoding='utf-8'))
  rows=[x for f in m['families'] for b in f['microbatches'] for x in b['members']]
  self.assertEqual(len(rows),288); self.assertEqual(m['source'],c['source_provenance'])
  self.assertTrue(all(x['input_ids'] and len(x['input_ids'])==len(x['labels']) and any(v!=-100 for v in x['labels']) for x in rows))
 def test_draft_binding_removes_only_redundant_runtime_source_file(self):
  b=json.loads((E03/'E03_V7_LORA_GRADIENT_INTERFERENCE_DIAGNOSTIC_V1_RUN_001_AMENDMENT_V1_DRAFT_BINDING.json').read_text(encoding='utf-8'))
  self.assertFalse(b['execution_authorized']); self.assertNotIn('experiments/base_only_targeted_repair_remote_first_v2/remote_dataset_expected/TRAIN.jsonl',b['bound_files'])
  self.assertEqual(b['runtime_source_provenance']['sha256'],'fb615a681503bf1e8db013ffcf1d552cf51696e878b1e0da3b97ef296970e945')
 def test_v3_parent_initialization_and_source_contract(self):
  worker=(ROOT/'scripts/run_e03_v7_lora_gradient_interference_v3.py').read_text(encoding='utf-8'); launcher=(ROOT/'scripts/launch_e03_v7_lora_gradient_interference_v3.py').read_text(encoding='utf-8')
  self.assertIn('E03_MANIFEST_SOURCE_PROVENANCE_MISMATCH',worker);self.assertIn('x.output_root.parent.mkdir(parents=True,exist_ok=True)',launcher)
class PrivateBootstrapTests(unittest.TestCase):
 def test_cache_publish_and_non_https_rejection(self):
  with tempfile.TemporaryDirectory() as d:
   d=Path(d); raw=b'exact private asset'; h=hashlib.sha256(raw).hexdigest(); cache=d/'cache';cache.mkdir();(cache/h).write_bytes(raw)
   out=fetch(url='https://assets.example/revisions/r1/blob',revision='r1',expected_sha256=h,cache_dir=cache,destination=d/'out'/'asset',token_env='NO_TOKEN')
   self.assertEqual(out['source'],'CACHE');self.assertEqual((d/'out'/'asset').read_bytes(),raw)
   with self.assertRaisesRegex(RuntimeError,'HTTPS_REQUIRED'):fetch(url='http://assets.example/x',revision='r1',expected_sha256=h,cache_dir=cache,destination=d/'x',token_env='NO_TOKEN')
if __name__=='__main__':unittest.main()

