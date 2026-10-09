from __future__ import annotations
import tempfile,unittest
from pathlib import Path
from scripts.paired_rank_margin_launch_contract import failure_receipt,require_launch,sha
class LaunchContractTests(unittest.TestCase):
 def test_identity_cap_and_fresh_output(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d); inp=root/'i';inp.write_text('x');out=root/'run'/'result.json';rec=root/'run'/'receipt.json'
   require_launch(output=out,receipt=rec,cap_seconds=900,expected_hashes={'i':sha(inp)},actual_paths={'i':inp})
   (root/'run').mkdir()
   with self.assertRaisesRegex(RuntimeError,'FRESH'):
    require_launch(output=out,receipt=rec,cap_seconds=900,expected_hashes={'i':sha(inp)},actual_paths={'i':inp})
 def test_failure_receipt_is_atomic_and_nonoverwrite(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'r.json';failure_receipt(receipt=p,reason='X');self.assertEqual(__import__('json').loads(p.read_text())['status'],'FAILED_NO_UPDATE')
   with self.assertRaisesRegex(RuntimeError,'NON_OVERWRITE'):failure_receipt(receipt=p,reason='X')
if __name__=='__main__':unittest.main()
