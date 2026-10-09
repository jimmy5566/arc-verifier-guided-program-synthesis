from __future__ import annotations
import importlib.util,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('batch_contract',ROOT/'scripts/unified_native_batch_sensitivity_localization_contract.py');m=importlib.util.module_from_spec(spec);assert spec.loader;spec.loader.exec_module(m)
class BatchLocalizationTests(unittest.TestCase):
 def record(self,episode,token=1):
  return {'episode_id':episode,'generated_token_ids':[token],'physical_generated_token_ids':[token,13],'framework_trailing_pad_count':0,'nonterminal_trailing_pad_count':1,'eos_terminated_before_padding':False,'termination_status':'MAX_TOKENS','eos_observed':False,'trailing_pad_count':0,'parse_valid':False,'canonical_prediction_sha256':'x'}
 def test_ladder_and_semantic_fields_are_frozen(self):
  self.assertEqual(m.BATCH_MODES,(32,16,8,4,1));self.assertEqual([len(x) for x in m.length_bucketed_batches_for_mode([{'episode_id':str(i),'prompt_tokens':i} for i in range(17)],8)],[8,8,1])
  self.assertEqual(m.compare_episode_records([self.record('e')],[self.record('e')])['status'],'PASS')
  padded=self.record('e');padded['physical_generated_token_ids']=[1,13,13];padded['framework_trailing_pad_count']=2
  out=m.compare_episode_records([padded],[self.record('e')]);self.assertEqual(out['status'],'PASS');self.assertEqual(out['physical_deltas'][0]['field'],'physical_generated_token_ids')
  changed=self.record('e');changed['generated_token_ids']=[2]
  self.assertEqual(m.compare_episode_records([changed],[self.record('e')])['status'],'FAIL_EXACT_MISMATCH')
 def test_validation_groups_are_selected_from_full_cohort(self):
  rows=[{'episode_id':f'e{i:02d}','prompt_tokens':i} for i in range(20)]
  groups=m.full_cohort_groups_for_validation(rows,8,{'e01','e17'})
  self.assertEqual([[x['episode_id'] for x in g] for g in groups],[['e00','e01','e02','e03','e04','e05','e06','e07'],['e16','e17','e18','e19']])
 def test_decision_is_common_exact_and_nondeterminism_fails_closed(self):
  ok={'repeatability':{'status':'PASS'},'vs_batch1':{'status':'PASS'}};bad={'repeatability':{'status':'PASS'},'vs_batch1':{'status':'FAIL_EXACT_MISMATCH'}}
  table={'a':{32:bad,16:ok,8:ok,4:ok,1:ok},'b':{32:bad,16:bad,8:bad,4:bad,1:ok}}
  self.assertEqual(m.select_largest_common_batch(table,('a','b'))['status'],'ONLY_BATCH1_VALID')
  table['b'][4]=ok;table['b'][8]=ok;self.assertEqual(m.select_largest_common_batch(table,('a','b'))['selected_batch_size'],8)
  table['a'][8]={'repeatability':{'status':'FAIL'},'vs_batch1':{'status':'PASS'}};self.assertEqual(m.select_largest_common_batch(table,('a','b'))['status'],'RUNTIME_NONDETERMINISTIC_PAUSE')
 def test_worker_is_target_blind_and_update_free(self):
  text=(ROOT/'scripts/run_unified_native_batch_sensitivity_localization_v1.py').read_text(encoding='utf-8')
  self.assertIn("BATCH_MODES",text);self.assertIn("compare_episode_records",text);self.assertIn("full_cohort_groups_for_validation",text);self.assertIn("fixed_validation_subset",text)
  self.assertNotIn('torch.optim',text);self.assertNotIn('.backward(',text);self.assertNotIn('sealed_target_sidecar',text)
if __name__=='__main__':unittest.main()
