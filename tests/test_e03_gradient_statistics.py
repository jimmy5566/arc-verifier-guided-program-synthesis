import json
import sys
import tempfile
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from e03_gradient_statistics import FAMILIES,bootstrap,summary_from_records,b1_reconstruction,sensitivity_envelope,protected_sign_consistent
from postprocess_e03_v7_lora_gradient_interference_v1 import load,process

def rows(sign=1.0):
 out=[]
 for fi,family in enumerate(FAMILIES):
  for mb in range(4):
   value=(-1.0 if sign<0 and family in FAMILIES[:5] else 1.0)
   out.append({'canonical_family':family,'microbatch_index':mb,'supervised_token_count':mb+1,'gradient':[value,.1*(fi+1),.01*(mb+1)]})
 return out
class E03StatisticsTests(unittest.TestCase):
 def test_token_mean_and_deterministic_clustered_bootstrap(self):
  r=rows(-1);a=summary_from_records(r);b=bootstrap(r,replicates=100,seed=20261010);c=bootstrap(r,replicates=100,seed=20261010)
  self.assertEqual(b,c);self.assertEqual(set(a['weak_protected_cosines']),set(FAMILIES[:5]));self.assertIn('weak_by_equal_protected_macro',b)
 def test_b1_reconstruction_uses_token_numerator_weights(self):
  examples=[]
  for fi,family in enumerate(FAMILIES):
   for i in range(8):examples.append({'canonical_family':family,'supervised_token_count':i+1,'gradient':[float(i+1),float(fi+1)]})
  reconstructed=b1_reconstruction(examples);self.assertEqual(len(reconstructed),9);self.assertAlmostEqual(reconstructed[0]['gradient'][0],204/36)
 def test_sensitivity_and_opposite_protected_signs(self):
  r=rows();subset=[x for x in r if x['microbatch_index']==0]
  b1=[{'canonical_family':x['canonical_family'],'supervised_token_count':x['supervised_token_count'],'gradient':x['gradient']} for x in subset]
  self.assertTrue(all(v==0 for v in sensitivity_envelope(subset,b1).values()))
  self.assertFalse(protected_sign_consistent({'same_color':{'lower':-1,'upper':-.1},'color_mapping':{'lower':.1,'upper':1}}))
 def test_sensitivity_covers_each_protected_decision_metric(self):
  r=rows();subset=[x for x in r if x['microbatch_index']==0]
  b1=[]
  for x in subset:
   for i in range(8): b1.append({'canonical_family':x['canonical_family'],'supervised_token_count':i+1,'gradient':x['gradient']})
  keys=sensitivity_envelope(subset,b1)
  for protected in ('same_color','color_mapping'):
   self.assertIn(f'equal_protected:{protected}',keys)
   self.assertIn(f'historical_protected:{protected}',keys)
 def test_zero_norm_fails_closed(self):
  r=rows();r[0]['gradient']=[0.,0.,0.]
  with self.assertRaisesRegex(RuntimeError,'ZERO_NORM'):summary_from_records(r)
 def test_worker_no_optimizer_or_generation(self):
  source=(ROOT/'scripts/run_e03_v7_lora_gradient_interference_v1.py').read_text(encoding='utf-8')
  self.assertNotIn('torch.optim',source);self.assertNotIn('.generate(',source);self.assertIn('model.zero_grad(set_to_none=True)',source)
 def test_postprocessor_accepts_complete_synthetic_statistics(self):
  primary=rows();b1=[]
  for fi,family in enumerate(FAMILIES):
   for i in range(8):b1.append({'canonical_family':family,'supervised_token_count':i+1,'gradient':[1.,.1*(fi+1),.01]})
  raw={'status':'COMPLETE_NO_UPDATE_E03_SUFFICIENT_STATISTICS','manifest_rows':288,'batch1_rows':72,'optimizer_steps':0,'parameter_updates':0,'generation_calls':0,'final_audit_opened':False,'primary_b8_microbatch_gradients':primary,'batch1_per_example_gradients':b1}
  with tempfile.TemporaryDirectory() as td:
   p=Path(td)/'raw.json';p.write_text(json.dumps(raw),encoding='utf-8');result=process(load(p))
  self.assertEqual(result['bootstrap']['replicates'],10000)
  self.assertIn(result['classification'],{'LOCAL_INTERFERENCE_SUPPORTED','GRADIENT_CONFLICT_DEPRIORITIZED','INCONCLUSIVE','INCONCLUSIVE_PROTECTED_SIGNS_OPPOSE'})
if __name__=='__main__':unittest.main()
