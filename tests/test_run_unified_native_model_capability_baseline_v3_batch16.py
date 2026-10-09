from __future__ import annotations
import importlib.util, unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("unified_v3",ROOT/"scripts/run_unified_native_model_capability_baseline_v3_batch16.py")
mod=importlib.util.module_from_spec(spec); assert spec.loader; spec.loader.exec_module(mod)
class WorkerV3Batch16Tests(unittest.TestCase):
 def test_v3_freezes_batch16_and_target_blind_primary(self):
  text=(ROOT/"scripts/run_unified_native_model_capability_baseline_v3_batch16.py").read_text(encoding="utf-8-sig")
  self.assertEqual(mod.PROTOCOL,"UNIFIED_NATIVE_MODEL_CAPABILITY_BASELINE_V3_BATCH16")
  self.assertIn("length_bucketed_batch16(contexts)",text)
  self.assertIn("fixed_validation_subset(contexts)",text)
  self.assertIn("BATCH1_BATCH16_SEMANTIC_DRIFT",text)
  self.assertIn("generation_evidence",text)
  self.assertIn('tokenizer.padding_side="left"',text)
  self.assertIn("RANK2_FROM_EXACT_BATCH16_PRIMARY_PREFIX_AND_TRACE",text)
  self.assertIn("BATCH16_OOM_FAIL_CLOSED",text)
  self.assertNotIn("torch.optim",text); self.assertNotIn(".backward(",text)
 def test_generation_evidence_uses_token_level_parser(self):
  from scripts.arc2_token_grid_parser import TokenGridContract
  c=TokenGridContract(tuple(range(10)),10,15,13)
  evidence=mod.generation_evidence("e",[1,2,10,3,4,15,13],c)
  self.assertTrue(evidence["parse_valid"]); self.assertEqual(evidence["termination_status"],"EOS")
if __name__=="__main__": unittest.main()

