import unittest
from reconstruction_v2_launch_contract import require_exact,reserve_budget,terminal_receipt
class T(unittest.TestCase):
 def test_mismatch_fails(self):
  c={'launch_nonce':'n','worker_binding':{'argv':['p'],'environment':{'PYTHONPATH':'x'}}}; g={'authorization':{'exact_source_commit':'a','contract_sha256':'c'}}; d={'exact_source_commit':'a','contract_sha256':'c','launch_nonce':'n'}; p={'source_head':'a'}
  require_exact(c,g,d,p,'a',['p'],{'PYTHONPATH':'x'})
  with self.assertRaisesRegex(RuntimeError,'WORKER_ARGUMENTS_MISMATCH'): require_exact(c,g,d,p,'a',['q'],{'PYTHONPATH':'x'})
  with self.assertRaisesRegex(RuntimeError,'STALE_PREFLIGHT_SOURCE_COMMIT'): require_exact(c,g,d,{'source_head':'b'},'a',['p'],{'PYTHONPATH':'x'})
 def test_budget_and_receipt_fail_closed(self):
  import tempfile
  from pathlib import Path
  with tempfile.TemporaryDirectory() as x:
   p=Path(x)/'l'; p.write_text('{"scientific_gpu_seconds":28800}\n')
   with self.assertRaisesRegex(RuntimeError,'INSUFFICIENT_CUMULATIVE_BUDGET'): reserve_budget(p,28800,1)
  with self.assertRaisesRegex(RuntimeError,'TERMINAL_RECEIPT_FIELDS_MISSING'): terminal_receipt({})
if __name__=='__main__': unittest.main()
