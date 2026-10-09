from __future__ import annotations
import importlib.util,json,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('target_score',ROOT/'scripts/score_unified_native_target_alignment_v1.py');mod=importlib.util.module_from_spec(spec);assert spec.loader;spec.loader.exec_module(mod)
class TargetAlignmentTests(unittest.TestCase):
 def test_frozen_rows_reject_missing_generation_evidence(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'raw.jsonl';p.write_text(json.dumps({'episode_id':'e','prompt_sha256':'p','greedy_token_ids':[1]})+'\n')
   self.assertEqual(mod.frozen_rows(p)[0]['episode_id'],'e')
 def test_target_sidecar_rejects_malformed_or_missing(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'sidecar.json';p.write_text(json.dumps({'target_token_ids_by_episode':{'e':[1,2]}}))
   self.assertEqual(mod.sidecar_targets(p),{'e':[1,2]})
 def test_target_alignment_never_selects_candidate(self):
  item=mod.align_target_evidence(generated_token_ids=[4],target_token_ids=[7],target_ranks=[2],target_margins=[-0.1])
  self.assertEqual(item['first_free_running_error_index'],0);self.assertFalse(item['target_alignment_used_for_selection'])
if __name__=='__main__':unittest.main()
