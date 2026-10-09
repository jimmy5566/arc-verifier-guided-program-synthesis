from __future__ import annotations
import unittest
from scripts.paired_rank_margin_runtime_contract import extract_target_logit_indices,left_pad_teacher_forced

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
if __name__=='__main__':unittest.main()
