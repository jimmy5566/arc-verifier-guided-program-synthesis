from __future__ import annotations
import importlib.util, unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("unified_v2",ROOT/"scripts/run_unified_native_model_capability_baseline_v2.py")
mod=importlib.util.module_from_spec(spec); assert spec.loader; spec.loader.exec_module(mod)
class WorkerV2Tests(unittest.TestCase):
 def test_v2_freezes_batch32_and_target_blind_primary(self):
  text=(ROOT/"scripts/run_unified_native_model_capability_baseline_v2.py").read_text(encoding="utf-8-sig")
  self.assertEqual(mod.PROTOCOL,"UNIFIED_NATIVE_MODEL_CAPABILITY_BASELINE_V2")
  self.assertIn("length_bucketed_batches(contexts)",text)
  self.assertIn("fixed_validation_subset(contexts)",text)
  self.assertIn("BATCH1_BATCH32_MATERIAL_DRIFT",text)
  self.assertNotIn("torch.optim",text); self.assertNotIn(".backward(",text)
if __name__=="__main__": unittest.main()
