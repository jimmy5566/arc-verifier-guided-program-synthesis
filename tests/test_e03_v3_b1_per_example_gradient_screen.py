from __future__ import annotations

import hashlib
import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/capability_repair_baseline_v1/e03_v7_lora_gradient_interference_diagnostic_v1"
CONFIG = BASE / "E03_V3_B1_PER_EXAMPLE_GRADIENT_SCREEN_CONFIG_V1.json"
MANIFEST = BASE / "E03_V3_B1_PER_EXAMPLE_GRADIENT_SCREEN_MANIFEST_V1.json"
SOURCE = BASE / "E03_TRAIN_MICROBATCH_MANIFEST_V1.json"
SPEC = importlib.util.spec_from_file_location("e03v3post", ROOT / "scripts" / "postprocess_e03_v3_b1_per_example_gradient_screen.py")
assert SPEC and SPEC.loader
post = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(post)


class E03V3FrozenContractTests(unittest.TestCase):
    def test_exact_b1_subset_is_reused_train_only(self):
        data = json.loads(MANIFEST.read_text())
        source = json.loads(SOURCE.read_text())
        expected = [item for family in source["fixed_batch1_sensitivity_subset"]["families"] for item in family["episode_ids"]]
        actual = [item["episode_id"] for family in data["families"] for item in family["members"]]
        self.assertEqual(actual, expected)
        self.assertEqual(len(actual), 72)
        self.assertEqual(len(set(actual)), 72)
        self.assertTrue(all(item.startswith("TRAIN:") for item in actual))
        self.assertEqual(data["source_manifest_sha256"], hashlib.sha256(SOURCE.read_bytes()).hexdigest())

    def test_no_update_and_bootstrap_contract_is_frozen(self):
        config = json.loads(CONFIG.read_text())
        self.assertFalse(config["execution_authorized"])
        self.assertEqual(config["batch_size"], 1)
        self.assertEqual(config["bootstrap"], {"interval": "2.5 and 97.5 percentile", "replicates": 10000, "seed": 20261010, "stratification": "resample eight frozen episodes with replacement within each family", "unit": "episode"})
        self.assertLessEqual(config["runtime_cap_seconds"], 1800)
        for forbidden in ("NO_OPTIMIZER", "NO_PARAMETER_UPDATES", "NO_GENERATION", "NO_GOLD", "NO_DGOLD", "NO_FINAL_AUDIT", "NO_AUTOMATIC_RETRY"):
            self.assertIn(forbidden, config["forbidden"])
        self.assertEqual(config["numerics"]["repeatability_contract"]["rows"], 9)
        self.assertEqual(config["numerics"]["repeatability_contract"]["max_relative_l2"], 0.0001)
        self.assertEqual(config["numerics"]["repeatability_contract"]["min_cosine"], 0.9999)

    def test_worker_fails_closed_on_extra_trainable_parameters(self):
        source = (ROOT / "scripts" / "run_e03_v3_b1_per_example_gradient_screen.py").read_text()
        self.assertIn("parameter.requires_grad_(name in allow)", source)
        self.assertIn("E03_V3_TRAINABLE_PARAMETER_SET_MISMATCH", source)

    def test_gram_bootstrap_is_deterministic_and_finite(self):
        families = [name for name in post.WEAK + post.PROTECTED for _ in range(8)]
        gram = [[1.0 if i == j else 0.0 for j in range(56)] for i in range(56)]
        result_a = post.bootstrap(gram, families, [1] * 56, replicates=16, seed=20261010)
        result_b = post.bootstrap(gram, families, [1] * 56, replicates=16, seed=20261010)
        self.assertEqual(result_a, result_b)
        self.assertTrue(all(abs(value) < float("inf") for values in result_a.values() for value in values))

    def test_secondary_branch_and_repeatability_are_decision_driving(self):
        samples = {name: [-0.2] * 20 for name in post.WEAK + ("secondary_combined",)}
        point = {name: -0.2 for name in post.WEAK + ("secondary_combined",)}
        self.assertEqual(post.classify(point, samples, {"status": "PASS"}), "LOCAL_GRADIENT_INTERFERENCE_SUPPORTED")
        positive = {name: [0.1] * 20 for name in post.WEAK + ("secondary_combined",)}
        self.assertEqual(post.classify({name: 0.1 for name in point}, positive, {"status": "PASS"}), "GRADIENT_INTERFERENCE_DEPRIORITIZED")
        self.assertEqual(post.classify(point, samples, {"status": "INCONCLUSIVE"}), "INCONCLUSIVE")


if __name__ == "__main__":
    unittest.main()
