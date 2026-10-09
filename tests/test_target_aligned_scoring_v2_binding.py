from __future__ import annotations
import hashlib,json,subprocess,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'experiments/capability_repair_baseline_v1/unified_native_model_capability_baseline_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
class FrozenBindingTests(unittest.TestCase):
 def test_bound_files_and_runtime_policy(self):
  b=json.loads((BASE/'TARGET_ALIGNED_SCORING_V2_BATCH16_LAUNCH_BINDING_V1.json').read_text())
  self.assertEqual(b['runtime_cap_seconds'],9000);self.assertEqual(b['raw']['sha256'],'4be1741f0b59f132efc7e19aa86f119fb49717cc37ef874c39b344654004e26e')
  self.assertEqual(sha(ROOT/b['worker']['path']),b['worker']['sha256']);self.assertEqual(sha(ROOT/b['launcher']['path']),b['launcher']['sha256']);self.assertEqual(len(b['nonce']),32)
  self.assertIn('generation',b['forbidden']);self.assertIn('backward',b['forbidden'])
 def test_worker_entrypoint_and_no_generation(self):
  worker=ROOT/'scripts/run_unified_native_target_aligned_scoring_v2_batch16.py';source=worker.read_text(encoding='utf8');self.assertNotIn('.generate(',source)
  p=subprocess.run([sys.executable,'-m','scripts.run_unified_native_target_aligned_scoring_v2_batch16','--help'],cwd=ROOT,capture_output=True,text=True)
  self.assertEqual(p.returncode,0);self.assertIn('--sidecar-sha256',p.stdout)
if __name__=='__main__':unittest.main()

