import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from scripts import e04_orientation_fixed_demonstration_baseline_v3 as e
class V3Test(unittest.TestCase):
 def test_freeze_validate_regenerate(self):
  with tempfile.TemporaryDirectory() as tmp:
   out=Path(tmp)/'out'; self.assertEqual(e.freeze(out)['validation_prompts'],768); self.assertEqual(e.validate_frozen(out)['status'],'PASS'); self.assertTrue(e.regeneration(out)['byte_identical'])
 def test_v2_turn_three_duplicate_is_rejected(self):
  with tempfile.TemporaryDirectory() as tmp:
   out=Path(tmp)/'out'; e.freeze(out)
   prompts=[json.loads(x) for x in (out/'VALIDATION_INPUT_PROMPTS.jsonl').read_text().splitlines()]
   targets=[json.loads(x) for x in (out/'TARGET_SCORER_SIDECAR.jsonl').read_text().splitlines()]
   prompts[3]['train'][3]={'input':prompts[3]['test'][0]['input'],'output':targets[3]['target']}
   with self.assertRaisesRegex(e.V3Failure,'QUERY_DEMONSTRATION_INPUT_COLLISION'):
    e.validate_rows(prompts,targets)
 def test_reserved_demo_exposure_is_rejected(self):
  reserved=e.v2.canonical_bases('VALIDATION')[:4]
  with patch.object(e,'selected_bank_bases',return_value=reserved):
   with self.assertRaisesRegex(e.V3Failure,'DEMONSTRATION_SELECTOR_CELL|RESERVED_CELL_EXPOSURE'):
    e.validate_bank()
 def test_fixed_bank_and_swatch(self):
  e.validate_bank(); self.assertEqual(len(e.demonstration_bank('ROTATION_TARGET')),4)
if __name__=='__main__': unittest.main()
