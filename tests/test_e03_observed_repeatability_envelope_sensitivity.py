import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from postprocess_e03_v3_observed_repeatability_envelope_sensitivity import (
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_RAW_SHA256,
    analyse,
    conservative_cosine_interval,
)

BASE = ROOT / "experiments" / "capability_repair_baseline_v1" / "e03_v7_lora_gradient_interference_diagnostic_v1"
RAW = BASE / "run_003_e6651046ee8d0c6ac3ff911c28f02d5f" / "RAW_GRAM_STATISTICS.json"
PUBLISHED = BASE / "run_003_e6651046ee8d0c6ac3ff911c28f02d5f" / "POSTPROCESS_RESULT.json"
MANIFEST = BASE / "E03_V3_B1_PER_EXAMPLE_GRADIENT_SCREEN_MANIFEST_V1.json"


class E03ObservedEnvelopeSensitivityTests(unittest.TestCase):
    def test_frozen_e03_reconstructs_all_published_points(self):
        result = analyse(RAW, PUBLISHED, MANIFEST)
        self.assertEqual(result["status"], "COMPLETE_CPU_ONLY_POST_HOC")
        self.assertEqual(result["input_identity"]["raw_gram_sha256"], EXPECTED_RAW_SHA256)
        self.assertEqual(result["input_identity"]["manifest_sha256"], EXPECTED_MANIFEST_SHA256)
        self.assertEqual(set(result["contrasts"]), {"connected_components", "inside_contains", "width", "difference", "orientation", "secondary_combined"})
        for value in result["contrasts"].values():
            self.assertLessEqual(value["lower"], value["point_cosine"])
            self.assertGreaterEqual(value["upper"], value["point_cosine"])

    def test_conservative_bound_contains_unperturbed_cosine(self):
        gram = [[1.0, 0.5], [0.5, 1.0]]
        bound = conservative_cosine_interval([1.0, 0.0], [0], [1.0], [0.0, 1.0], [1], [1.0], gram, 0.01)
        self.assertLessEqual(bound["lower"], 0.5)
        self.assertGreaterEqual(bound["upper"], 0.5)

    def test_wrong_raw_identity_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "RAW_SHA256_MISMATCH"):
            analyse(RAW, PUBLISHED, MANIFEST, expected_raw_sha256="0" * 64)


if __name__ == "__main__":
    unittest.main()
