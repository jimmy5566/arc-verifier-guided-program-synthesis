from __future__ import annotations
import importlib.util, json, tempfile, unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("unified_preflight",ROOT/"scripts/preflight_unified_native_model_capability_baseline_v1.py")
mod=importlib.util.module_from_spec(spec); assert spec.loader; spec.loader.exec_module(mod)

class UnifiedPreflightTests(unittest.TestCase):
 def test_current_cpu_package_passes(self):
  result=mod.validate(mod.OUT)
  self.assertEqual(result["status"],"CPU_PREFLIGHT_PASS_RUNTIME_IDENTITY_PENDING")
  self.assertFalse(result["gpu_used"])
 def test_test_target_output_fails_closed(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d); (root/"SYNTHETIC_BENCHMARK_INPUT_MANIFEST_V1.json").write_text(json.dumps({"status":"INPUTS_FROZEN_TARGETS_SEALED_CPU_ONLY","episodes":[{"episode_id":str(i),"observation_sha256":str(i),"observation":{"task":{"test":[{"input":[],"output":[]}]}}} for i in range(60)],"content_level_audit":{"gold_dgold_final_audit_accessed":False},"sealed_target_sidecar":{"sha256":"x","not_committed":True}}))
   (root/"CHECKPOINT_PROVENANCE_DISCOVERY_V1.json").write_text(json.dumps({"checkpoint_records":{"HISTORICAL_FOUNDATION_V2":{"status":"UNAVAILABLE_NOT_SUBSTITUTED"}}}))
   (root/"EXECUTION_CONTRACT_DRAFT_V1.json").write_text("{}")
   with self.assertRaisesRegex(RuntimeError,"SEALED_TARGET_LEAK"): mod.validate(root)
if __name__=="__main__": unittest.main()
