import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/capability_repair_baseline_v1/connected_components_object_extraction_capability_pilot_v1"
BINDING = BASE / "connected_component_count_code_pair_supervision_pilot_v1/EXECUTION_BINDING_V3.json"


def load(relative, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


COMPAT = load("scripts/score_connected_component_count_code_pair_supervision_pilot_v1.py", "code_pair_score_compat_test")


class CountCodePairPostfreezeScoreCompatibilityTest(unittest.TestCase):
    def test_projection_is_derived_from_the_bound_source_and_inputs(self):
        projected, receipt = COMPAT.project(BINDING)
        original = json.loads(BINDING.read_text())
        self.assertEqual(receipt["original_binding_sha256"], hashlib.sha256(BINDING.read_bytes()).hexdigest())
        self.assertEqual(projected["score_entrypoint_sha256"], original["source_files"]["scripts/score_connected_components_object_extraction_pilot_v1.py"])
        self.assertEqual(projected["dataset_paths"], {
            "OOD": original["inputs"]["ood_prompts"]["path"],
            "RETENTION": original["inputs"]["retention_prompts"]["path"],
            "COMPOSITION": original["inputs"]["composition_prompts"]["path"],
        })
        self.assertFalse(receipt["targets_opened"])
        self.assertFalse(receipt["model_loaded"])
        self.assertFalse(receipt["gpu_used"])

    def test_projection_validation_fails_before_target_context_when_raw_is_incomplete(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "RAW_FREEZE_RECEIPT.json").write_text(json.dumps({
                "status": "COMPLETE_RAW_FROZEN_NO_TARGETS", "targets_accessed": False, "raw_sha256": {}
            }))
            with self.assertRaisesRegex(RuntimeError, "RAW_ARTIFACT_SET"):
                COMPAT.validate_projection(BINDING, output)
            self.assertFalse((output / "POSTFREEZE_SCORE_COMPATIBILITY_RECEIPT.json").exists())


if __name__ == "__main__":
    unittest.main()
