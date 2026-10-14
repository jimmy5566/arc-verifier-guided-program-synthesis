from __future__ import annotations
import json
import unittest

from scripts.e4g_inside_contains_atomic import OUT, build, cpu_preflight, paired_bootstrap


class E4GInsideContainsTests(unittest.TestCase):
    def test_freeze_is_deterministic_and_target_blind(self):
        self.assertEqual(build(), build())
        for name in ("TRAIN_FIT_INPUTS_AND_COMMITMENTS.json", "STRUCTURAL_OOD_INPUTS_AND_COMMITMENTS.json"):
            panel=json.loads((OUT/name).read_text())
            self.assertEqual(len(panel["records"]),96)
            self.assertTrue(all("output" not in r["input_task"]["test"][0] and r["oracle_valid"] for r in panel["records"]))
            groups={}
            for r in panel["records"]: groups.setdefault(r["canonical_unit_id"],set()).add(r["demo_order"])
            self.assertEqual(len(groups),48); self.assertTrue(all(v=={"A","B"} for v in groups.values()))

    def test_schedule_preserves_replay_and_balances_visible_relation(self):
        build(); schedule=json.loads((OUT/"TRAINING_SCHEDULE.json").read_text())["episodes"]
        self.assertEqual(len(schedule),384)
        primary=[r for r in schedule if r["role"]=="PRIMARY_CONNECTED_COMPONENT_COUNT"]
        replay=[r for r in schedule if r["role"]!="PRIMARY_CONNECTED_COMPONENT_COUNT"]
        self.assertEqual((len(primary),len(replay)),(48,336))
        commands=[r["task"]["test"][0]["input"][0][0] for r in primary]
        self.assertEqual((commands.count(8),commands.count(9)),(24,24))
        self.assertTrue(all(len(r["task"]["train"])==2 for r in primary))

    def test_panels_have_no_train_or_cross_panel_prompt_or_query_overlap(self):
        build(); a=json.loads((OUT/"TRAIN_FIT_INPUTS_AND_COMMITMENTS.json").read_text())["records"]; b=json.loads((OUT/"STRUCTURAL_OOD_INPUTS_AND_COMMITMENTS.json").read_text())["records"]
        self.assertFalse({x["full_prompt_sha256"] for x in a}&{x["full_prompt_sha256"] for x in b})
        self.assertFalse({x["query_input_sha256"] for x in a}&{x["query_input_sha256"] for x in b})
        self.assertEqual({x["construct"]["topology"] for x in a},{"SINGLE_FRAME"})
        self.assertEqual({x["construct"]["topology"] for x in b},{"DOUBLE_FRAME"})

    def test_production_path_and_paired_unit_contract_without_model(self):
        receipt=cpu_preflight(); self.assertTrue(receipt["launcher_to_worker_to_raw_freeze"])
        self.assertFalse(receipt["model_loaded"]); self.assertFalse(receipt["targets_opened"])
        self.assertEqual(paired_bootstrap([0]*48,[1]*48)["delta"],48)
        with self.assertRaises(ValueError): paired_bootstrap([],[])

if __name__ == "__main__": unittest.main()
