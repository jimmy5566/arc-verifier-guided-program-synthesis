import importlib.util,unittest
from pathlib import Path
spec=importlib.util.spec_from_file_location('m',Path('scripts/analyze_r1_r2_per_family.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def row(n,a,b,c): return ({'status':'COLLECTED_PASS','predictions':[{'episode_id':'TARGET_DEV:TARGETED_EVALUATION:connected components:000000','exact_grid_match':a}]},{'status':'COLLECTED_PASS','predictions':[{'episode_id':'TARGET_DEV:TARGETED_EVALUATION:connected components:000000','exact_grid_match':b}]},{'status':'COLLECTED_PASS','predictions':[{'episode_id':'TARGET_DEV:TARGETED_EVALUATION:connected components:000000','exact_grid_match':c}]})
class T(unittest.TestCase):
 def test_improved_then_regressed(self):
  x=m.analyze(*row(0,False,True,False));f=next(iter(x['families'].values()));self.assertEqual(f['classification'],'IMPROVED_THEN_REGRESSED');self.assertEqual(f['repeated_flip_examples']['improved_r1_then_regressed_r2'],1)
 def test_recovered(self):
  x=m.analyze(*row(0,True,False,True));f=next(iter(x['families'].values()));self.assertEqual(f['classification'],'RECOVERED_IN_R2')
if __name__=='__main__':unittest.main()
