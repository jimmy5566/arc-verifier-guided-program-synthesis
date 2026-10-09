import importlib.util, json, subprocess, sys, tempfile, unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def load(name,file):
 s=importlib.util.spec_from_file_location(name,ROOT/'scripts'/file);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
W=load('beam4','run_paired_target_blind_beam4_candidate_recall_v1.py')
class Beam4Tests(unittest.TestCase):
 def test_target_blind_parser(self):
  self.assertEqual(W.candidate_parse([1,2,10,3,4,15,13]),('VALID',[[1,2],[3,4]]))
  for ids in ([15,1],[1,15,2],[1,10,2,3],[11,1],[]): self.assertNotEqual(W.candidate_parse(ids)[0],'VALID')
 def test_freeze_package_and_self_test(self):
  p=subprocess.run([sys.executable,str(ROOT/'scripts/freeze_paired_target_blind_beam4_candidate_recall_v1.py')],capture_output=True,text=True)
  self.assertEqual(p.returncode,0,p.stderr)
  cfg=ROOT/'experiments/capability_repair_baseline_v1/paired_target_blind_beam4_candidate_recall_v1/EXECUTION_PACKAGE.json'
  p=subprocess.run([sys.executable,str(ROOT/'scripts/run_paired_target_blind_beam4_candidate_recall_v1.py'),'--config',str(cfg),'--self-test'],capture_output=True,text=True)
  self.assertEqual(p.returncode,0,p.stderr); self.assertEqual(json.loads(p.stdout)['rows'],60)
if __name__=='__main__':unittest.main()
