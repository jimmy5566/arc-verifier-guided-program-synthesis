from __future__ import annotations
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_connected_components_pilot_integrity():
    path = ROOT / "scripts/validate_connected_components_object_extraction_capability_pilot_v1.py"
    spec = importlib.util.spec_from_file_location("cc_pilot_validator", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    result = module.run()
    assert result["status"] == "PASS_CPU_ONLY_CONSTRUCT_REPLAY_EXPOSURE_AND_SERIALIZER_VALIDATION"
    assert result["byte_identical_protected_replay_slots"] == 288
    assert result["independent_target_blind_prompts"] == {"ood": 48, "retention": 24, "composition_sentinel": 16}
