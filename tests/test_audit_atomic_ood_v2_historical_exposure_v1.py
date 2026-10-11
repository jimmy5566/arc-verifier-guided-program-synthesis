from __future__ import annotations
import importlib.util,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'experiments/capability_repair_baseline_v1/atomic_ood_identifiability_pilot_v2_historical_exposure_audit_v1'
spec=importlib.util.spec_from_file_location('audit',ROOT/'scripts/audit_atomic_ood_v2_historical_exposure_v1.py');m=importlib.util.module_from_spec(spec);assert spec.loader;spec.loader.exec_module(m)
def test_reconstructs_complete_role_aware_audit_from_bound_assets():
 result=m.audit();frozen=json.loads((OUT/'HISTORICAL_ROLE_AWARE_EXPOSURE_AUDIT.json').read_text())
 assert result==frozen and result['status']=='PASS' and len(result['historical_asset_coverage'])==len(m.ASSETS)
 assert result['historical_tasks']>0 and result['latent_tuple_intersection']==0 and not any(result['direct_exposure_intersections'].values())
def test_missing_asset_fails_closed(monkeypatch):
 monkeypatch.setattr(m,'ASSETS',('not/a/real/frozen-asset.jsonl',))
 try:m.audit()
 except RuntimeError as e:assert 'HISTORICAL_ASSET_MISSING' in str(e)
 else:raise AssertionError('missing asset accepted')
def test_refuses_overwrite():
 p=subprocess.run([sys.executable,str(ROOT/'scripts/audit_atomic_ood_v2_historical_exposure_v1.py')],capture_output=True,text=True)
 assert p.returncode and 'REFUSE_OVERWRITE_HISTORICAL_AUDIT' in p.stderr
