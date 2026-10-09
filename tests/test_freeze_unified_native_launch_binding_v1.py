from __future__ import annotations
import importlib.util,json,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("freeze_binding",ROOT/"scripts/freeze_unified_native_launch_binding_v1.py");mod=importlib.util.module_from_spec(spec);assert spec.loader;spec.loader.exec_module(mod)
class FreezeBindingTests(unittest.TestCase):
 def test_contract_fields_are_frozen(self):
  import subprocess
  with tempfile.TemporaryDirectory() as d:
   out=Path(d)/"binding.json"; subprocess.run(["py","-3",str(ROOT/"scripts/freeze_unified_native_launch_binding_v1.py"),"--out",str(out),"--nonce","0123456789abcdef0123456789abcdef"],check=True,capture_output=True,text=True)
   x=json.loads(out.read_text());self.assertEqual(x["runtime_cap_seconds"],9000);self.assertFalse(x["authorization"]["scientific_training_authorized"]);self.assertIn("training",x["forbidden"]);self.assertIn("all_available_adapter_sha256",x["machine_gates"])
if __name__=="__main__":unittest.main()
