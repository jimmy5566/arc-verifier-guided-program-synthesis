import hashlib,json,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'experiments/foundation_v2_reconstruction_and_targeted_repair_v2/round_009_post_hoc_capability_characterization/ROUND_009_POST_HOC_TARGET_FREE_MAPPING_PROOF.json'
class TargetFreeProofTests(unittest.TestCase):
 def test_proof_is_exact_target_free_256_row_mapping(self):
  raw=P.read_bytes();v=json.loads(raw)
  self.assertEqual('e4c0a4e4a60a45bb5cd02be4229fff1d3e084bed4cb3dcc6b83a7fc390c37133',hashlib.sha256(raw).hexdigest())
  self.assertFalse(v['target_payloads_included']);self.assertEqual({'NOVEL':128,'REPLAY':64,'PROTECTED':64},v['counts']);self.assertEqual(256,len(v['rows']))
  self.assertEqual(256,len({(r['surface'],r['manifest_item_id']) for r in v['rows']}))
  forbidden=('gold_grid','reference_output','target_tokens','labels','prompt_ids')
  self.assertFalse(any(any(k in r for k in forbidden) for r in v['rows']))
if __name__=='__main__': unittest.main()
