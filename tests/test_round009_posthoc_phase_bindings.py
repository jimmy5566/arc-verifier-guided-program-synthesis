import json
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'experiments/foundation_v2_reconstruction_and_targeted_repair_v2/round_009_post_hoc_capability_characterization'
class PhaseBindingTests(unittest.TestCase):
 def test_phase_bindings_are_isolated_and_fail_closed(self):
  a=json.loads((D/'ROUND_009_POST_HOC_PHASE_A_BINDING_V1.json').read_text()); b=json.loads((D/'ROUND_009_POST_HOC_PHASE_B_BINDING_V1.json').read_text())
  self.assertFalse(a['execution_authorized']); self.assertFalse(b['execution_authorized'])
  self.assertNotEqual(a['output_root'],b['output_root']); self.assertTrue(a['sealed_mapping_must_be_unreadable'])
  self.assertNotIn('sealed_scorer',' '.join(a['argv'])); self.assertTrue(b['requires_phase_a_terminal_immutable'])
  self.assertEqual(2000031,a['decode']['seed']); self.assertEqual('EXACT_128_64_64_NO_SHRINKING',b['denominator_rules'])
if __name__=='__main__': unittest.main()
