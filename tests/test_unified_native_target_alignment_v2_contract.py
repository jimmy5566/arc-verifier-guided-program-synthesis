from __future__ import annotations
import importlib.util,json,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('c',ROOT/'scripts/unified_native_target_alignment_v2_contract.py');m=importlib.util.module_from_spec(spec);assert spec.loader;spec.loader.exec_module(m)
class ContractTests(unittest.TestCase):
 def row(self,c,e):return {'checkpoint_condition':c,'episode_id':e,'family':'STRUCTURAL_TRANSLATION','prompt_sha256':'x','generated_token_ids':[1],'requested_batch_size':16,'effective_batch_size':16}
 def test_exact_scores_and_first_error(self):
  self.assertTrue(m.exact_scores(greedy_grid=[[1]],alternate_grid=None,target_grid=[[1]])['greedy_exact_grid_match'])
  self.assertEqual(m.first_error([1,2],[1,3]),1);self.assertEqual(m.first_error([1],[1,2]),2)
 def test_sidecar_requires_exact_episode_mapping(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'x.json';p.write_text(json.dumps({'target_token_ids_by_episode':{'e':[1]}}));h=m.sha(p)
   self.assertEqual(m.read_sidecar(p,h,{'e'}),{'e':[1]})
   with self.assertRaisesRegex(RuntimeError,'EPISODE_MAPPING'):m.read_sidecar(p,h,{'other'})
 def test_raw_rejects_target_fields(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'x.jsonl';p.write_text(json.dumps({'target':1})+'\n')
   with self.assertRaisesRegex(RuntimeError,'RAW_'):m.read_raw(p,m.sha(p))
if __name__=='__main__':unittest.main()

