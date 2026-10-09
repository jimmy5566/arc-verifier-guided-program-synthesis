from __future__ import annotations
import tempfile,unittest
from pathlib import Path
from scripts.paired_rank_margin_launch_contract import acquire_live_lock,failure_receipt,release_live_lock,require_checkpoint_records,require_launch,sha
class LaunchContractTests(unittest.TestCase):
 def test_identity_cap_and_fresh_output(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d); inp=root/'i';inp.write_text('x');out=root/'run'/'result.json';rec=root/'run'/'receipt.json'
   require_launch(output=out,receipt=rec,cap_seconds=900,expected_hashes={'i':sha(inp)},actual_paths={'i':inp})
   (root/'run').mkdir()
   require_launch(output=out,receipt=rec,cap_seconds=900,expected_hashes={'i':sha(inp)},actual_paths={'i':inp})
   out.write_text('done')
   with self.assertRaisesRegex(RuntimeError,'FRESH'):
    require_launch(output=out,receipt=rec,cap_seconds=900,expected_hashes={'i':sha(inp)},actual_paths={'i':inp})
 def test_live_lock_rejects_duplicate_and_releases(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'stable-parent'/'.nonce.live.lock';acquire_live_lock(p)
   with self.assertRaisesRegex(RuntimeError,'DUPLICATE'):acquire_live_lock(p)
   release_live_lock(p);acquire_live_lock(p)
 def test_failure_receipt_is_atomic_and_nonoverwrite(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'r.json';failure_receipt(receipt=p,reason='X');self.assertEqual(__import__('json').loads(p.read_text())['status'],'FAILED_NO_UPDATE')
   with self.assertRaisesRegex(RuntimeError,'NON_OVERWRITE'):failure_receipt(receipt=p,reason='X')
 def test_checkpoint_discovery_requires_manifests_and_adapters(self):
  d={'checkpoint_records':{'V7':{'manifest_path':'m','manifest_sha256':'a','adapter_model_sha256':'b'},'FB':{'manifest_path':'m','manifest_sha256':'c','adapter_model_sha256':'d'}}}
  self.assertEqual(set(require_checkpoint_records(d,('V7','FB'))),{'V7','FB'})
if __name__=='__main__':unittest.main()
