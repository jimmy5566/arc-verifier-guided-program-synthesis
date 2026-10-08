import importlib.util, unittest
from pathlib import Path
spec=importlib.util.spec_from_file_location('a',Path('scripts/analyze_paired_capability_repair.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def row(i,surface,ok):return {'episode_id':i,'surface':surface,'exact_grid_match':ok}
class TestAnalysis(unittest.TestCase):
 def test_gates_pass(self):
  base={'status':'COLLECTED_PASS','predictions':[row('TARGET_DEV:TARGETED_COMPOSITION:a','TARGET_DEV',False),row('TARGET_DEV:TARGETED_COMPOSITION:b','TARGET_DEV',False),row('TARGET_DEV:TARGETED_COMPOSITION:c','TARGET_DEV',False)]+[row(f'TARGET_DEV:TARGETED_COMPOSITION:x{i}','TARGET_DEV',True) for i in range(31)]+[row(f'RETENTION_SENTINEL:r{i}','RETENTION_SENTINEL',True) for i in range(70)]+[row(f'RETENTION_SENTINEL:n{i}','RETENTION_SENTINEL',False) for i in range(26)]}
  cand={'status':'COLLECTED_PASS','predictions':[dict(x) for x in base['predictions']]}
  cand['predictions'][0]['exact_grid_match']=True;cand['predictions'][1]['exact_grid_match']=True;cand['predictions'][2]['exact_grid_match']=True
  x=m.analyze(base,cand);self.assertEqual(x['prospective_r2_gate_status'],'PASS')
 def test_harm_fails_retention(self):
  base={'status':'COLLECTED_PASS','predictions':[row(f'TARGET_DEV:TARGETED_COMPOSITION:a{i}','TARGET_DEV',True) for i in range(34)]+[row(f'RETENTION_SENTINEL:r{i}','RETENTION_SENTINEL',True) for i in range(70)]}
  cand={'status':'COLLECTED_PASS','predictions':[dict(x) for x in base['predictions']]};cand['predictions'][-1]['exact_grid_match']=False
  self.assertFalse(m.analyze(base,cand)['prospective_r2_gates']['retention_recovery'])
if __name__=='__main__':unittest.main()
