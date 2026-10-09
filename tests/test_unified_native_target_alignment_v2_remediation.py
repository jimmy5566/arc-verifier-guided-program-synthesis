from __future__ import annotations
import importlib.util,json,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('c',ROOT/'scripts/unified_native_target_alignment_v2_contract.py');m=importlib.util.module_from_spec(spec);assert spec.loader;spec.loader.exec_module(m)
class RemediationTests(unittest.TestCase):
 def test_top2_coverage_includes_greedy(self):
  x=m.exact_scores(greedy_grid=[[1]],alternate_grid=[[2]],target_grid=[[1]])
  self.assertTrue(x['greedy_exact_grid_match']);self.assertFalse(x['alternate_only_exact_grid_match']);self.assertTrue(x['complete_output_top2_coverage'])
 def test_summary_has_composition_and_retention(self):
  rows=[]
  for c in m.CONDITIONS:
   for f in m.FAMILIES:
    for i in range(12):rows.append({'checkpoint_condition':c,'family':f,'greedy_exact_grid_match':False,'alternate_only_exact_grid_match':False,'complete_output_top2_coverage':False})
  s=m.summarize_exact(rows);self.assertEqual(s['compositional_aggregate']['denominator'],144);self.assertEqual(s['protected_same_color_retention']['denominator'],72);self.assertEqual(s['by_condition'][m.CONDITIONS[0]]['denominator'],60)
 def test_manifest_sha_and_fresh_output_fail_closed(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'manifest.json';p.write_text('{}');self.assertEqual(m.require_manifest_identity(p,m.sha(p)),{})
   with self.assertRaisesRegex(RuntimeError,'MANIFEST_SHA'):m.require_manifest_identity(p,'0'*64)
   with self.assertRaisesRegex(RuntimeError,'FRESH_OUTPUT'):m.require_fresh_output(Path(d))
 def test_atomic_failure_receipt_write(self):
  import importlib.util
  spec=importlib.util.spec_from_file_location('launcher',ROOT/'scripts/launch_unified_native_target_aligned_scoring_v2_batch16.py');mod=importlib.util.module_from_spec(spec);assert spec.loader;spec.loader.exec_module(mod)
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'receipt.json';mod.atomic(p,{'status':'FAILURE'});self.assertEqual(json.loads(p.read_text())['status'],'FAILURE');self.assertFalse((Path(d)/'receipt.json.tmp').exists())
if __name__=='__main__':unittest.main()
