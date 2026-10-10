import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from freeze_e03_v7_lora_gradient_interference_v1 import FAMILIES, build_manifest


class E03FreezeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        train = ROOT / "experiments/base_only_targeted_repair_remote_first_v2/remote_dataset_expected/TRAIN.jsonl"
        cls.rows = [json.loads(line) for line in train.read_text(encoding="utf-8").splitlines() if line.strip()]
        cls.doc = build_manifest(cls.rows)

    def test_train_only_disjoint_complete_shape(self):
        self.assertEqual(self.doc["rows"], 288)
        self.assertEqual(len(self.doc["families"]), 9)
        ids = []
        for family in self.doc["families"]:
            self.assertEqual(len(family["microbatches"]), 4)
            for batch in family["microbatches"]:
                self.assertEqual(len(batch["members"]), 8)
                ids.extend(member["episode_id"] for member in batch["members"])
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(all(episode.startswith("TRAIN:") for episode in ids))

    def test_source_mapping_and_serialization_are_bound(self):
        wanted = {name: (role, family, program) for name, _, role, family, program in FAMILIES}
        for family in self.doc["families"]:
            name = family["canonical_family"]
            self.assertEqual(tuple(family["source_mapping"].values()), wanted[name])
            for batch in family["microbatches"]:
                for member in batch["members"]:
                    self.assertEqual(len(member["input_ids"]), len(member["labels"]))
                    self.assertGreater(member["supervised_token_count"], 0)
                    self.assertEqual(len(member["token_label_sha256"]), 64)
                    self.assertEqual(len(member["prompt_serialization_sha256"]), 64)
                    self.assertEqual(len(member["target_serialization_sha256"]), 64)

    def test_fixed_batch1_subset_is_one_predeclared_microbatch_per_family(self):
        sensitivity = self.doc["fixed_batch1_sensitivity_subset"]
        self.assertEqual(sensitivity["rows"], 72)
        self.assertEqual(len(sensitivity["families"]), 9)
        self.assertTrue(all(item["microbatch_index"] == 0 and len(item["episode_ids"]) == 8 for item in sensitivity["families"]))


if __name__ == "__main__":
    unittest.main()
