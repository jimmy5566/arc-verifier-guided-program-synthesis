import importlib.util,json,unittest
from pathlib import Path
P=Path(__file__).parents[1]/'scripts'/'run_e05_v3_marker_responsiveness_error_taxonomy.py'
s=importlib.util.spec_from_file_location('t',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def row(e,grid,valid=True):return {'episode_id':e,'parser_valid':valid,'parsed_grid':grid}
class TestTaxonomy(unittest.TestCase):
 def setUp(self):self.targets={str(i):[[i]] for i in range(4)}
 def test_precedence_and_legal_classes(self):
  good=[row(str(i),[[i]]) for i in range(4)];self.assertEqual(m.classify(good,self.targets),'CORRECT_MAPPING')
  invalid=[row(str(i),[[i]],i!=0) for i in range(4)];self.assertEqual(m.classify(invalid,self.targets)[0],'STRUCTURAL_OR_FORMAT_FAILURE')
  nonoracle=[row(str(i),[[9]]) for i in range(4)];self.assertEqual(m.classify(nonoracle,self.targets)[0],'STRUCTURAL_OR_FORMAT_FAILURE')
  invariant=[row(str(i),[[0]]) for i in range(4)];self.assertEqual(m.classify(invariant,self.targets),'MARKER_INVARIANT_LEGAL_ROTATION')
  wrong=[row(str(i),[[i%2]]) for i in range(4)];self.assertEqual(m.classify(wrong,self.targets),'MARKER_RESPONSIVE_WRONG_MAPPING')
 def test_frozen_artifact_and_grouping_gates(self):
  spec=json.loads(m.SPEC.read_text(encoding='utf-8-sig'));self.assertEqual(m.sha(m.SPEC),m.SPEC_SHA)
  for name,expected in spec['raw_evidence_sha256'].items():self.assertEqual(m.sha(m.EVIDENCE/name),expected)
  side=m.read_rows(m.SIDE);self.assertEqual(len(side),768)
  for condition in spec['primary_scope']:
   groups={}
   for item in [x for x in side if x['condition']==condition]:groups.setdefault(item['canonical_base_id'],[]).append(item)
   self.assertEqual(len(groups),48);self.assertTrue(all(sorted(x['turn'] for x in rows)==[0,1,2,3] for rows in groups.values()))
 def test_fixed_percentile(self):self.assertEqual(m.percentile_ci([0]*500+[1]*9500),[0,1])
if __name__=='__main__':unittest.main()
