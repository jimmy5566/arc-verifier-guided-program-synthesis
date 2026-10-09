from __future__ import annotations
import importlib.util, unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("unified_worker",ROOT/"scripts/run_unified_native_model_capability_baseline_v1.py")
mod=importlib.util.module_from_spec(spec); assert spec.loader; spec.loader.exec_module(mod)
class UnifiedWorkerTests(unittest.TestCase):
 def test_native_prompt_rejects_sealed_target(self):
  task={"train":[{"input":[[1]],"output":[[2]]}],"test":[{"input":[[3]],"output":[[4]]}]}
  with self.assertRaisesRegex(RuntimeError,"SEALED_TARGET_LEAK"): mod.prompt(task)
 def test_native_prompt_has_one_generation_turn(self):
  task={"train":[{"input":[[1]],"output":[[2]]}],"test":[{"input":[[3]]}]}
  self.assertTrue(mod.prompt(task).endswith("<|im_start|>assistant\n"))
 def test_fixed_runtime_and_no_training_imports(self):
  text=(ROOT/"scripts/run_unified_native_model_capability_baseline_v1.py").read_text(encoding="utf-8")
  self.assertEqual(mod.RUNTIME_CAP_SECONDS,9000); self.assertEqual(mod.MAX_NEW_TOKENS,128)
  self.assertNotIn("torch.optim",text); self.assertNotIn(".backward(",text)
if __name__=="__main__": unittest.main()
