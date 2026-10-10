import hashlib,json,tempfile,unittest
from pathlib import Path
from scripts import e04_v3_full768_b1_remeasurement as e
from scripts.e04_orientation_v3_no_update import canon
class Full768B1Tests(unittest.TestCase):
 def _rows(self,prompts):
  return [{'row_index':i,'input_sha256':hashlib.sha256(canon(p)).hexdigest(),'generated_token_ids':[1],'text':'1','parser_valid':True,'parsed_grid':[[1]],'prompt_tokens':1,'completion_tokens':1,'elapsed_seconds':0.0,'effective_batch_size':1,'physical_returned_token_ids':[1],'pad_token_count':0} for i,p in enumerate(prompts)]
 def test_cpu_preflight_is_target_blind(self):
  r=e.cpu_preflight(); self.assertEqual(r['status'],'PASS_NO_MODEL_IMPORT');self.assertFalse(r['target_sidecar_accessed']);self.assertEqual(r['physical_batch_size'],1)
 def test_full_raw_and_reference_gate(self):
  c=e.load_config(); prompts=e.read_jsonl(e.ROOT/c['prompt_path']); rows=self._rows(prompts); refs=e.read_jsonl(e.ROOT/c['sensitivity_reference_raw_path'])
  for ref in refs:
   for k in ('generated_token_ids','text','parser_valid','parsed_grid','prompt_tokens','completion_tokens','elapsed_seconds','physical_returned_token_ids','pad_token_count'):
    rows[ref['row_index']][k]=ref[k]
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'raw.jsonl';e.atomic_jsonl(p,rows); loaded=e.validate_full_raw(p,prompts);e.verify_embedded_reference(loaded,c)
   rows[refs[0]['row_index']]['generated_token_ids']=[999]; rows[refs[0]['row_index']]['physical_returned_token_ids']=[999]; rows[refs[0]['row_index']]['completion_tokens']=1; e.atomic_jsonl(p,rows);loaded=e.validate_full_raw(p,prompts)
   with self.assertRaisesRegex(e.FullB1Failure,'E04_FULL768_B1_REPEATABILITY'):e.verify_embedded_reference(loaded,c)
if __name__=='__main__':unittest.main()
