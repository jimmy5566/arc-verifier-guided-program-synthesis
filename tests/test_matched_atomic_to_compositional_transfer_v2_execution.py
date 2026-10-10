import json, tempfile, unittest
from pathlib import Path
from scripts.prepare_matched_atomic_to_compositional_transfer_v2 import prepare
class ExecutionContractTests(unittest.TestCase):
 def test_public_cohort_has_no_test_targets_and_fixed_validation_is_exact(self):
  with tempfile.TemporaryDirectory() as d:
   r=prepare(Path(d)/'out',Path(d)/'sealed');m=json.loads(r['manifest'].read_text())
   self.assertEqual(len(m['episodes']),168)
   self.assertEqual(m['fixed_batch1_validation']['episode_count'],42)
   self.assertTrue(all('output' not in x['observation']['task']['test'][0] for x in m['episodes']))
 def test_worker_has_no_target_sidecar_argument_or_update_operations(self):
  text=(Path('scripts/run_matched_atomic_to_compositional_transfer_v2.py')).read_text()
  self.assertNotIn("--sidecar",text);self.assertNotIn('optimizer.step',text);self.assertNotIn('.backward(',text)
  self.assertLess(text.index('checkpoints={condition:load_checkpoint'),text.index('from transformers import'))
if __name__=='__main__':unittest.main()
