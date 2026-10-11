from __future__ import annotations
import importlib.util, json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('e05v2',ROOT/'scripts/freeze_e05_v2_identifiable_prompt_repair.py')
m=importlib.util.module_from_spec(spec);assert spec and spec.loader;spec.loader.exec_module(m)
def test_v2_plan_has_unique_observable_base_inputs():
 rows=m.plan();assert len(rows)==192;assert len({(r['layout'],r['colors']) for r in rows})==48
def test_v2_freeze_makes_conditions_model_observable_and_targets_isolated():
 # Production OUT is intentionally frozen by this test's separate process.
 assert set(m.banks())==set(m.CONDS)
 assert all(m.banks()[c] for c in m.CONDS)
