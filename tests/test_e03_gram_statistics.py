import sys, unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from e03_gram_statistics import FAMILIES,WEAK,PROTECTED,summary,bootstrap,sensitivity
from postprocess_e03_v7_lora_gradient_interference_v2 import classify

def gram(n): return [[1.0 if i==j else 0.0 for j in range(n)] for i in range(n)]
def primary():
 rows=[]
 for fi,f in enumerate(FAMILIES):
  for mb in range(4):rows.append({'canonical_family':f,'microbatch_index':mb,'supervised_token_count':mb+1,'basis_index':len(rows)})
 return rows
class GramContractTests(unittest.TestCase):
 def test_token_weighted_summary_and_bootstrap_are_deterministic(self):
  rows=primary();g=gram(len(rows));s=summary(rows,g,4);a=bootstrap(rows,g,8,20261010);b=bootstrap(rows,g,8,20261010)
  self.assertEqual(a,b);self.assertEqual(set(s['weak_protected_macro_by_weak']),set(WEAK))
 def test_b1_sensitivity_uses_reconstructed_family_estimands(self):
  rows=primary();g=gram(36);sub=[{'canonical_family':r['canonical_family'],'microbatch_index':0,'supervised_token_count':r['supervised_token_count'],'basis_index':i} for i,r in enumerate(r for r in rows if r['microbatch_index']==0)]
  b1=[{'canonical_family':f,'microbatch_index':0,'supervised_token_count':1,'basis_index':i} for i,f in enumerate(FAMILIES)]
  env,_,_=sensitivity(sub,gram(9),b1,gram(9));self.assertEqual(set(env).issuperset({'negative_pair_frequency','historical_protected_macro','equal_protected_macro'}),True)
 def test_point_estimate_gate_is_not_interval_lower_bound(self):
  # A wide interval alone cannot satisfy the frozen -0.05 effect condition.
  point={'weak_protected_macro_by_weak':{f:-.01 for f in WEAK},'historical_protected_macro':-.01,'negative_pair_frequency':.2}
  intervals={'weak_by_equal_protected_macro':{f:{'lower':-.2,'upper':-.01} for f in WEAK},'historical_macro':{'lower':-.2,'upper':-.01},'equal_macro':{'lower':-.2,'upper':-.01},'historical_by_protected':{p:{'lower':-.2,'upper':-.01} for p in PROTECTED},'equal_by_protected':{p:{'lower':-.2,'upper':-.01} for p in PROTECTED}}
  env={f'weak:{f}':0 for f in WEAK}|{'historical_protected_macro':0,'equal_protected_macro':0,'negative_pair_frequency':0}|{f'historical_protected:{p}':0 for p in PROTECTED}|{f'equal_protected:{p}':0 for p in PROTECTED}
  self.assertNotEqual(classify(point,intervals,env)[0],'LOCAL_INTERFERENCE_SUPPORTED')
 def test_historical_branch_not_equal_counterfactual_controls_support(self):
  point={'weak_protected_macro_by_weak':{f:0.1 for f in WEAK},'historical_protected_macro':0.1,'negative_pair_frequency':.2}
  intervals={'weak_by_equal_protected_macro':{f:{'lower':.05,'upper':.2} for f in WEAK},'historical_macro':{'lower':.05,'upper':.2},'equal_macro':{'lower':-.3,'upper':-.1},'historical_by_protected':{p:{'lower':.05,'upper':.2} for p in PROTECTED},'equal_by_protected':{p:{'lower':-.3,'upper':-.1} for p in PROTECTED}}
  env={f'weak:{f}':0 for f in WEAK}|{'historical_protected_macro':0,'equal_protected_macro':0,'negative_pair_frequency':0}|{f'historical_protected:{p}':0 for p in PROTECTED}|{f'equal_protected:{p}':0 for p in PROTECTED}
  self.assertNotEqual(classify(point,intervals,env)[0],'LOCAL_INTERFERENCE_SUPPORTED')
 def test_negative_frequency_reversal_is_inconclusive(self):
  point={'weak_protected_macro_by_weak':{f:.1 for f in WEAK},'historical_protected_macro':.1,'negative_pair_frequency':.09}
  intervals={'weak_by_equal_protected_macro':{f:{'lower':.05,'upper':.2} for f in WEAK},'historical_macro':{'lower':.05,'upper':.2},'equal_macro':{'lower':.05,'upper':.2},'historical_by_protected':{p:{'lower':.05,'upper':.2} for p in PROTECTED},'equal_by_protected':{p:{'lower':.05,'upper':.2} for p in PROTECTED}}
  env={f'weak:{f}':0 for f in WEAK}|{'historical_protected_macro':0,'equal_protected_macro':0,'negative_pair_frequency':.02}|{f'historical_protected:{p}':0 for p in PROTECTED}|{f'equal_protected:{p}':0 for p in PROTECTED}
  self.assertEqual(classify(point,intervals,env)[0],'INCONCLUSIVE_NEGATIVE_FREQUENCY_SENSITIVITY')
if __name__=='__main__':unittest.main()
