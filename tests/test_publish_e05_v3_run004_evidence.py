import importlib.util
from pathlib import Path
import unittest
P=Path(__file__).parents[1]/'scripts'/'publish_e05_v3_run004_evidence.py'
spec=importlib.util.spec_from_file_location('pub',P); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
class TestPublishE05Evidence(unittest.TestCase):
 def test_missing_evidence_fails_closed(self):
  import tempfile
  with tempfile.TemporaryDirectory() as d:
   with self.assertRaisesRegex(RuntimeError,'MISSING_FROZEN'):
    m.validate_source(Path(d))
 def test_manifest_has_no_target_sidecar(self):
  got={'a':{'sha256':'x','bytes':1}}
  out=m.manifest(got)
  self.assertEqual(out['transport'],'GITHUB_NON_FORCE_COMMIT')
  self.assertFalse(out['target_sidecar_accessed_by_transport'])
  self.assertEqual(out['files'],got)
if __name__=='__main__': unittest.main()
