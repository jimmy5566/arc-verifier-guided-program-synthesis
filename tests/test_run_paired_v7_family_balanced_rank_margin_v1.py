from __future__ import annotations
import subprocess,sys,tempfile,unittest
from pathlib import Path
class WorkerEntryTests(unittest.TestCase):
 def test_missing_binding_fails_before_model_import(self):
  with tempfile.TemporaryDirectory() as d:
   r=subprocess.run([sys.executable,'scripts/run_paired_v7_family_balanced_rank_margin_v1.py','--binding',str(Path(d)/'missing'),'--output',str(Path(d)/'o'),'--receipt',str(Path(d)/'r')],capture_output=True,text=True)
   self.assertNotEqual(r.returncode,0);self.assertIn('LAUNCH_BINDING_MISSING',r.stderr)
 def test_invalid_binding_schema_fails_before_model_import(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'b.json';p.write_text('{}');r=subprocess.run([sys.executable,'scripts/run_paired_v7_family_balanced_rank_margin_v1.py','--binding',str(p),'--output',str(Path(d)/'o'),'--receipt',str(Path(d)/'r')],capture_output=True,text=True)
   self.assertNotEqual(r.returncode,0);self.assertIn('LAUNCH_BINDING_SCHEMA_INVALID',r.stderr)
if __name__=='__main__':unittest.main()
