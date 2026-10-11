from __future__ import annotations
import importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('e05v3',ROOT/'scripts/freeze_e05_v3_semantic_demo_and_asset_recovery.py');m=importlib.util.module_from_spec(spec);assert spec and spec.loader;spec.loader.exec_module(m)
def test_marker_rotation_demo_has_four_distinct_inputs_and_outputs():
 bank=m.banks()['MARKER_CONDITIONED_ROTATION'];assert len({str(x['input']) for x in bank})==4;assert len({str(x['output']) for x in bank})==4
def test_marker_binding_demo_has_four_distinct_outputs():
 bank=m.banks()['MARKER_BINDING_CONTROL'];assert len({str(x['output']) for x in bank})==4
def test_semantic_oracle_audit_passes():assert m.sem_audit(m.banks())['status']=='PASS'
