from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/capability_repair_baseline_v1/e03_v7_lora_gradient_interference_diagnostic_v1"
CONFIG = BASE / "E03_V3_B1_PER_EXAMPLE_GRADIENT_SCREEN_CONFIG_V2.json"
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
        self.assertNotIn("repeatability_tolerance", config["numerics"])

    def test_worker_fails_closed_on_extra_trainable_parameters(self):
        source = (ROOT / "scripts" / "run_e03_v3_b1_per_example_gradient_screen.py").read_text()
        self.assertIn("parameter.requires_grad_(name in allow)", source)
        self.assertIn("E03_V3_TRAINABLE_PARAMETER_SET_MISMATCH", source)
        self.assertIn("allow = tuple(sorted(config", source)
        config = json.loads(CONFIG.read_text())
        configured = tuple(config["lora_parameter_name_allowlist"])
        runtime_same_names_different_order = tuple(reversed(configured))
        self.assertEqual(tuple(sorted(configured)), tuple(sorted(runtime_same_names_different_order)))
        self.assertNotEqual(configured, tuple(sorted(configured)))

    def test_gram_bootstrap_is_deterministic_and_finite(self):
        families = [name for name in post.FAMILY_ORDER for _ in range(8)]
        gram = [[1.0 if i == j else 0.0 for j in range(72)] for i in range(72)]
        result_a = post.bootstrap(gram, families, [1] * 72, replicates=16, seed=20261010)
        result_b = post.bootstrap(gram, families, [1] * 72, replicates=16, seed=20261010)
        self.assertEqual(result_a, result_b)
        self.assertTrue(all(abs(value) < float("inf") for values in result_a.values() for value in values))

    def test_bootstrap_is_process_stable_across_python_hash_seeds(self):
        source = (
            "import importlib.util,json; "
            f"s=importlib.util.spec_from_file_location('p',{str(ROOT / 'scripts' / 'postprocess_e03_v3_b1_per_example_gradient_screen.py')!r}); "
            "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
            "f=[x for x in m.FAMILY_ORDER for _ in range(8)]; "
            "g=[[1.0 if i==j else 0.0 for j in range(72)] for i in range(72)]; "
            "print(json.dumps(m.bootstrap(g,f,[1]*72,16,20261010),sort_keys=True))"
        )
        outputs = []
        for seed in ("1", "2"):
            env = dict(os.environ, PYTHONHASHSEED=seed)
            outputs.append(subprocess.check_output([sys.executable, "-c", source], env=env, text=True).strip())
        self.assertEqual(outputs[0], outputs[1])

    def test_raw_identity_and_positive_token_counts_fail_closed(self):
        data = json.loads(MANIFEST.read_text())
        raw = {
            "episode_ids": [member["episode_id"] for family in data["families"] for member in family["members"]],
            "families": [family["canonical_family"] for family in data["families"] for _ in family["members"]],
            "supervised_token_counts": [member["supervised_token_count"] for family in data["families"] for member in family["members"]],
        }
        self.assertEqual(post.validate_raw_identity(raw, MANIFEST)[0], raw["families"])
        raw["episode_ids"] = raw["episode_ids"][1:]
        with self.assertRaises(ValueError):
            post.validate_raw_identity(raw, MANIFEST)

    def test_production_bootstrap_contract_is_not_cli_overridable(self):
        source = (ROOT / "scripts" / "postprocess_e03_v3_b1_per_example_gradient_screen.py").read_text()
        self.assertIn("E03_V3_BOOTSTRAP_CONTRACT_INVALID", source)
        self.assertIn("args.replicates != FROZEN_REPLICATES", source)

    def test_secondary_branch_and_repeatability_are_decision_driving(self):
        samples = {name: [-0.2] * 20 for name in post.WEAK + ("secondary_combined",)}
        point = {name: -0.2 for name in post.WEAK + ("secondary_combined",)}
        self.assertEqual(post.classify(point, samples, {"status": "PASS"}), "LOCAL_GRADIENT_INTERFERENCE_SUPPORTED")
        positive = {name: [0.1] * 20 for name in post.WEAK + ("secondary_combined",)}
        self.assertEqual(post.classify({name: 0.1 for name in point}, positive, {"status": "PASS"}), "GRADIENT_INTERFERENCE_DEPRIORITIZED")
        self.assertEqual(post.classify(point, samples, {"status": "INCONCLUSIVE"}), "INCONCLUSIVE")


if __name__ == "__main__":
    unittest.main()
