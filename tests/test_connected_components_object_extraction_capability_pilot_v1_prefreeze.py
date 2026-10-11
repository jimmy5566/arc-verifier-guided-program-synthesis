from __future__ import annotations
import importlib.util
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def test_prefreeze_draft_is_matched_and_cpu_only():
    path=ROOT/'scripts/validate_connected_components_object_extraction_capability_pilot_v1_prefreeze.py'
    spec=importlib.util.spec_from_file_location('cc_prefreeze',path); module=importlib.util.module_from_spec(spec); assert spec.loader; spec.loader.exec_module(module)
    assert module.validate()['status']=='PASS_CPU_ONLY_PREFREEZE_DRAFT'
