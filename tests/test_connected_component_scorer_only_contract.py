import importlib.util,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('score',ROOT/'scripts/score_connected_component_fit_generation_diagnostic_v1.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class TestScorerOnly(unittest.TestCase):
 def test_top_level_sidecar_hash_is_accepted(self): self.assertEqual(m.sidecar_sha256({'sidecar_sha256':'a'*64,'execution':{}}),'a'*64)
 def test_conflicting_hashes_fail_closed(self):
  with self.assertRaisesRegex(RuntimeError,'CONFLICT'): m.sidecar_sha256({'sidecar_sha256':'a'*64,'execution':{'sidecar_sha256':'b'*64}})
if __name__=='__main__':unittest.main()
