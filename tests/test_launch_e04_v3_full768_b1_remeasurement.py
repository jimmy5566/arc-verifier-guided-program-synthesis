import unittest
from scripts.launch_e04_v3_full768_b1_remeasurement import remote_script
class LaunchTest(unittest.TestCase):
 def test_remote_contract_is_detached_and_bounded(self):
  b={'output_root':'/workspace/arc2/e04_full768/run_x','source_commit':'a'*40,'nonce':'n'}
  s=remote_script(b,'b'*40,'x/binding.json','c'*64,'host')
  for needle in ('git checkout --detach','test "$(git rev-parse HEAD)"','git status --porcelain','test ! -e','pgrep -af','--self-test','--cap-seconds 1800','run_e04_v3_full768_b1_remeasurement.py','LAUNCH_BINDING.json.tmp','expected_terminal_receipt'):
   self.assertIn(needle,s)
  self.assertNotIn('score_e04_v3_full768',s)
if __name__=='__main__':unittest.main()
