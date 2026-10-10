import json, tempfile, unittest
from pathlib import Path
from scripts import freeze_e04_c_matched_rotation_pilot as cohort
from scripts import freeze_e04_c_matched_rotation_schedule as schedule

class E04CMatchedScheduleTests(unittest.TestCase):
    def test_structural_arm_matching(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); source=root/"source"; out=root/"schedule"
            cohort.freeze(source); binding=schedule.build(source,out)
            self.assertEqual(binding["optimizer_steps"],96)
            self.assertEqual(binding["token_equality"],"NOT_RUN_REQUIRES_FROZEN_V7_TOKENIZER_STATIC_PREFLIGHT")
            control=json.loads((out/"CONTROL_SCHEDULE.json").read_text())["episodes"]
            treatment=json.loads((out/"TREATMENT_SCHEDULE.json").read_text())["episodes"]
            self.assertEqual(len(control),384); self.assertEqual(len(treatment),384)
            self.assertEqual([x["pair_id"] for x in control],[x["pair_id"] for x in treatment])
            self.assertEqual(sum(x["intervention_slot"] for x in control),96)
            self.assertTrue(all(c["control_task"]==t["treatment_task"] for c,t in zip(control[96:],treatment[96:])))
            self.assertTrue(all(c["control_task"]!=t["treatment_task"] for c,t in zip(control[:96],treatment[:96])))
    def test_refuses_rewrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); source=root/"source"; out=root/"schedule"; cohort.freeze(source); schedule.build(source,out)
            with self.assertRaises(schedule.FreezeFailure): schedule.build(source,out)

if __name__=="__main__": unittest.main()
