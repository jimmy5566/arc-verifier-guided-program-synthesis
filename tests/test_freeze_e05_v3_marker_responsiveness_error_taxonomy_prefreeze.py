import json,subprocess,sys,unittest
from pathlib import Path
ROOT=Path(__file__).parents[1];OUT=ROOT/'experiments/capability_repair_baseline_v1/e05_v3_semantically_identifiable_demonstrations_and_adapter_storage_recovery'
class TestPrefreeze(unittest.TestCase):
 def test_frozen_taxonomy_is_exhaustive_and_cpu_only(self):
  subprocess.run([sys.executable,str(ROOT/'scripts/freeze_e05_v3_marker_responsiveness_error_taxonomy_prefreeze.py')],check=True)
  x=json.loads((OUT/'E05_V3_MARKER_RESPONSIVENESS_ERROR_TAXONOMY_PREFREEZE_SPEC_V1.json').read_text())
  self.assertEqual(len(x['taxonomy_order']),4);self.assertEqual(set(x['taxonomy_order']),set(x['taxonomy']))
  self.assertFalse(x['execution_authorization']);self.assertTrue(x['preexecution_review_required'])
  self.assertIn('NO_GPU_WORK',x['prohibitions'])
if __name__=='__main__':unittest.main()
