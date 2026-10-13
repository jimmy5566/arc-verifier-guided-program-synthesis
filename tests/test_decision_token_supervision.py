import json
import sys
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts.decision_token_supervision import count_code_pair_label_indices,matched_weights
from training_data.pipeline import task_to_sample
class T(unittest.TestCase):
 def test_real_schedule_has_384_unambiguous_count_code_tokens(self):
  rows=json.loads((ROOT/'experiments/capability_repair_baseline_v1/connected_components_object_extraction_capability_pilot_v1/TREATMENT_SCHEDULE.json').read_text())['episodes']; focused=0
  for row in rows:
   if row['role']!='PRIMARY_CONNECTED_COMPONENT_COUNT': continue
   sample=task_to_sample(row['task']); indices,counts=count_code_pair_label_indices(row['task']);control,treatment=matched_weights(sample['labels'],indices)
   self.assertEqual(len(indices),8);self.assertEqual(len(counts),4);self.assertEqual(sum(control.values()),60);self.assertEqual(sum(treatment.values()),60)
   self.assertEqual(sum(treatment[i]==2 for i in indices),8);self.assertTrue(all(abs(treatment[i]-11/13)<1e-12 for i in treatment if i not in indices));focused+=len(indices)
  self.assertEqual(focused,384)
if __name__=='__main__':unittest.main()
