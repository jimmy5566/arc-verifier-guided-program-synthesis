import json,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from prepare_matched_atomic_to_compositional_transfer_v1 import prepare,transform
class MatchedTransferPreparationTests(unittest.TestCase):
 def test_freeze_has_exact_tuple_structure_and_sealed_targets(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);r=prepare(root/'out',root/'sealed');m=json.loads(r['manifest'].read_text());p=json.loads(r['protocol'].read_text());side=json.loads(r['sidecar'].read_text())
   self.assertEqual(len(m['episodes']),168);self.assertEqual(len(m['latent_parameter_tuples']),48);self.assertEqual(len(side['targets']),168);self.assertEqual(m['fixed_batch1_validation']['episode_count'],42);self.assertFalse(any('output' in e['observation']['task']['test'][0] for e in m['episodes']));self.assertEqual(p['inference']['batch_primary'],16)
 def test_selector_recolor_composition_is_exact(self):
  g=[[0]*9 for _ in range(9)];g[1][2]=7
  for y,x in ((2,2),(3,2),(3,3),(4,2)):g[y][x]=2
  for y,x in ((5,5),(6,5),(6,6),(7,5)):g[y][x]=2
  self.assertEqual(sum(x==7 for r in transform(g,2,7,'COMPOSITION_SELECT_RECOLOR') for x in r),4)
if __name__=='__main__':unittest.main()
