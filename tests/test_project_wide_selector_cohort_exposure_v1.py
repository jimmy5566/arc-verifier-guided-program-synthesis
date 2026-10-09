import json,tempfile,unittest
from pathlib import Path
class ExposureAuditTests(unittest.TestCase):
 def test_frozen_audit_declares_no_unexposed_target_dev_cohort(self):
  p=Path('experiments/capability_repair_baseline_v1/cpu_only_prospective_target_blind_beam4_selector_protocol_v1/PROJECT_WIDE_EXPOSURE_AUDIT.json')
  if not p.exists():self.skipTest('audit not frozen yet')
  x=json.loads(p.read_text());self.assertEqual(x['status'],'NO_UNEXPOSED_NON_GOLD_SELECTOR_EVALUATION_COHORT');self.assertEqual(x['episode_level_findings']['prior_exact_grid_scored_or_emitted'],192);self.assertEqual(x['episode_level_findings']['not_proven_unexposed'],0)
if __name__=='__main__':unittest.main()