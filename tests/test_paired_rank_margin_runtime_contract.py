from __future__ import annotations
import unittest
from scripts.paired_rank_margin_runtime_contract import correct_token_metrics,extract_target_logit_indices,left_pad_teacher_forced,require_complete_paired_rows,slice_teacher_forced_logits

class RuntimeContractTests(unittest.TestCase):
 def test_left_padding_keeps_per_row_causal_target_boundary(self):
  batch=left_pad_teacher_forced([{'episode_id':'a','prompt_ids':[7,8],'target_ids':[1,15]},{'episode_id':'b','prompt_ids':[4,5,6,7],'target_ids':[2,3,15]}])
  self.assertEqual(batch['attention_mask'][0],[0,0,0,1,1,1,1])
  self.assertEqual(batch['position_ids'][0],[0,0,0,0,1,2,3])
  self.assertEqual(extract_target_logit_indices(batch['boundaries'][0]),[4,5])
  self.assertEqual(extract_target_logit_indices(batch['boundaries'][1]),[3,4,5])
 def test_invalid_alignment_fails_closed(self):
  with self.assertRaisesRegex(RuntimeError,'CAUSAL_ALIGNMENT'):
   extract_target_logit_indices({'target_logit_start':4,'target_length':2,'sequence_width':5})
 def test_correct_token_metrics_are_tie_safe(self):
  m=correct_token_metrics([1.0,1.0,0.0],1);self.assertEqual(m['rank'],1);self.assertTrue(m['top1']);self.assertEqual(m['margin'],0.0)
 def test_complete_pairs_require_two_conditions_by_sixty(self):
  rows=[{'checkpoint_condition':c,'episode_id':str(i)} for c in ('RECONSTRUCTED_FOUNDATION_V2_V7','FAMILY_BALANCED') for i in range(60)]
  require_complete_paired_rows(rows)
 def test_teacher_forced_logit_slices_follow_boundaries(self):
  self.assertEqual(slice_teacher_forced_logits([[['a'],['b'],['c']]], [{'target_logit_start':1,'target_length':2,'sequence_width':3}]), [[['b'],['c']]])
if __name__=='__main__':unittest.main()
