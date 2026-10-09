import json,tempfile,unittest
from pathlib import Path
from scripts.freeze_cpu_only_prospective_target_blind_beam4_selector_protocol_v1 import metadata_only
class SelectorProtocolTests(unittest.TestCase):
 def test_metadata_only_prefix_extracts_identity(self):
  line='{"episode_id":"E","family":"F","task":{"test":[{"output":[[9]]}]}}'
  self.assertEqual(metadata_only(line),('E','F'))
 def test_metadata_only_rejects_target_in_prefix(self):
  with self.assertRaises(ValueError): metadata_only('{"episode_id":"E","output":[[9]],"family":"F","task":{}}')
 def test_selector_contract_has_no_target_feature(self):
  p=Path('experiments/capability_repair_baseline_v1/cpu_only_prospective_target_blind_beam4_selector_protocol_v1/EXECUTION_PACKAGE.json')
  if not p.exists(): self.skipTest('package not frozen yet')
  x=json.loads(p.read_text()); a=x['selector']['allowed_features']; f=x['selector']['forbidden_features']
  self.assertFalse(any('target' in v or 'exact' in v for v in a)); self.assertIn('evaluation_target',f); self.assertTrue(x['candidate_freeze_then_score']['candidate_freeze_required_before_target_access'])
if __name__=='__main__': unittest.main()