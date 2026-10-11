import importlib.util,unittest
from pathlib import Path
P=Path(__file__).parents[1]/'scripts'/'score_e05_v3_three_arm_native_greedy.py'
s=importlib.util.spec_from_file_location('score',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class TestScore(unittest.TestCase):
 def test_paired_ci_is_seeded_and_episode_scaled(self):
  x=[1,0,-1,2]*12
  self.assertEqual(m.ci_count(x),m.ci_count(x))
  lo,hi=m.ci_count(x);self.assertLessEqual(lo,hi);self.assertGreaterEqual(lo,-48);self.assertLessEqual(hi,96)
if __name__=='__main__':unittest.main()
