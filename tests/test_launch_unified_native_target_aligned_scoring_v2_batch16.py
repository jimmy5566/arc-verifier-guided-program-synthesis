from __future__ import annotations
import importlib.util,json,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('l',ROOT/'scripts/launch_unified_native_target_aligned_scoring_v2_batch16.py');m=importlib.util.module_from_spec(spec);assert spec.loader;spec.loader.exec_module(m)
class LauncherTests(unittest.TestCase):
 def test_binding_rejects_missing_sidecar(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'x.json';p.write_text('{}')
   with self.assertRaisesRegex(RuntimeError,'LAUNCH_BINDING_INVALID'):m.load_binding(p,'0'*40)
 def test_entrypoint_help_is_cpu_only(self):
  import subprocess,sys
  x=subprocess.run([sys.executable,'-m','scripts.launch_unified_native_target_aligned_scoring_v2_batch16','--help'],cwd=ROOT,capture_output=True,text=True)
  self.assertEqual(x.returncode,0);self.assertIn('--binding',x.stdout)
if __name__=='__main__':unittest.main()
