import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class E03BoundedWorkerTests(unittest.TestCase):
 def test_worker_has_no_optimizer_generation_or_json_vector_persistence(self):
  s=(ROOT/'scripts/run_e03_v7_lora_gradient_interference_v2.py').read_text(encoding='utf-8')
  self.assertNotIn('torch.optim',s);self.assertNotIn('.generate(',s);self.assertNotIn("'gradient':vector",s);self.assertIn('np.memmap',s);self.assertIn('pmm_path.unlink()',s);self.assertIn('copy_gradient(bmm,fi,params,offsets,torch,count,True)',s)
 def test_worker_requires_exact_allowlist_and_immutable_binding(self):
  s=(ROOT/'scripts/run_e03_v7_lora_gradient_interference_v2.py').read_text(encoding='utf-8')
  self.assertIn('E03_LORA_ALLOWLIST_EXACT_SET_MISMATCH',s);self.assertIn('E03_DIRECTOR_RESPONSE_HASH_MISMATCH',s);self.assertIn('verify_files(ck)',s);self.assertIn('E03_EXTERNAL_CAP_LAUNCHER_REQUIRED',s);self.assertIn('E03_DIRECTOR_DECISION_NOT_AUTHORIZING',s)
 def test_external_cap_is_process_level_and_has_no_retry(self):
  s=(ROOT/'scripts/launch_e03_v7_lora_gradient_interference_v2.py').read_text(encoding='utf-8')
  self.assertIn('os.killpg',s);self.assertIn('E03_EXTERNAL_RUNTIME_CAP_EXCEEDED',s);self.assertIn("'retry':False",s)
if __name__=='__main__':unittest.main()

