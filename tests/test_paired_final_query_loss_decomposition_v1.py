import json
import sys
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from paired_final_query_loss_common import COMPONENTS, component_sums, masks_for_task

def task():
    row=next(json.loads(x) for x in (ROOT/"experiments/targeted_capability_repair_v1/data/TARGET_DEV.jsonl").read_text().splitlines() if x.strip() and json.loads(x)["family"]=="orientation")
    return {"source_id":row["episode_id"],**row["task"]}

class FinalQueryMaskTests(unittest.TestCase):
    def test_masks_are_disjoint_exhaustive_and_query_has_eos(self):
        item=masks_for_task(task())
        self.assertEqual(sum(item["component_token_counts"].values()),item["assistant_token_count"])
        self.assertEqual(item["component_token_counts"]["FINAL_ASSISTANT_PREFIX"],3)
        self.assertEqual(item["component_token_counts"]["FINAL_EOS"],1)
        self.assertEqual(item["final_grid_plus_eos_token_count"],item["component_token_counts"]["FINAL_GRID_CONTENT"]+1)
        for pos in range(len(item["ids"])-1):
            self.assertEqual(sum(int(item["loss_masks"][name][pos]) for name in COMPONENTS),int(item["labels"][pos+1]!=-100))

    def test_one_loss_vector_reconstructs_legacy_all_assistant_sum(self):
        item=masks_for_task(task()); losses=[(i+1)/17.0 for i in range(len(item["ids"])-1)]
        values=component_sums(losses,item["loss_masks"])
        self.assertLess(abs(sum(values[name]["nll_sum"] for name in COMPONENTS)-values["ALL_ASSISTANT_LEGACY"]["nll_sum"]),1e-9)
        self.assertEqual(values["FINAL_GRID_PLUS_EOS"]["token_count"],values["FINAL_GRID_CONTENT"]["token_count"]+values["FINAL_EOS"]["token_count"])

if __name__ == "__main__":
    unittest.main()
