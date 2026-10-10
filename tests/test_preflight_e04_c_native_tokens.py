import json,tempfile,unittest
from pathlib import Path
from scripts import freeze_e04_c_matched_rotation_pilot as cohort
from scripts import freeze_e04_c_matched_rotation_schedule as schedule
from scripts import preflight_e04_c_native_tokens as preflight

class E04CNativeTokenPreflightTests(unittest.TestCase):
 def test_counts_are_matched_under_native_contract(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);source=root/"source";sched=root/"schedule";out=root/"result.json"
   cohort.freeze(source);schedule.build(source,sched);r=preflight.main(sched,out)
   self.assertEqual(r["status"],"PASS_NATIVE_RUNTIME_SERIALIZATION_CONTRACT")
   self.assertTrue(r["per_slot_equal_token_and_supervision"])
   self.assertEqual(r["rows"],384);self.assertEqual(r["intervention_rows"],96)
   self.assertEqual(r["control_total_tokens"],r["treatment_total_tokens"])
   self.assertEqual(r["control_supervised_tokens"],r["treatment_supervised_tokens"])
   self.assertFalse(r["full_hf_tokenizer_json"]["byte_equal"])
 def test_refuses_overwrite(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);source=root/"source";sched=root/"schedule";out=root/"result.json"
   cohort.freeze(source);schedule.build(source,sched);preflight.main(sched,out)
   with self.assertRaises(preflight.PreflightFailure):preflight.main(sched,out)
if __name__=="__main__":unittest.main()
