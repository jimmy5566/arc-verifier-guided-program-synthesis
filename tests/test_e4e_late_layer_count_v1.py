from __future__ import annotations
import json, os, subprocess, sys
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts.e4e_late_layer_global_count import AUXILIARY_COEFFICIENT,loss_contract,multiclass_target_index,visible_component_count

class E4ELateLayerCountTest(unittest.TestCase):
 def test_visible_count_and_loss_preserves_answer_ce(self):
  self.assertEqual(3,visible_component_count([[1,0,2],[0,0,0],[3,3,0]]))
  self.assertEqual(2,multiclass_target_index(3))
  control=loss_contract(2.0,4.0,False);treatment=loss_contract(2.0,4.0,True)
  self.assertEqual(1.0,control['answer_coefficient']);self.assertEqual(1.0,treatment['answer_coefficient'])
  self.assertEqual(0.0,control['auxiliary_coefficient']);self.assertEqual(AUXILIARY_COEFFICIENT,treatment['auxiliary_coefficient'])
  self.assertEqual(.5,control['step_term']);self.assertEqual(.75,treatment['step_term'])
 def test_freeze_preclaim_and_full_cpu_path(self):
  env={**os.environ,'ARC2_ROOT':str(ROOT)};pkg=ROOT/'experiments/capability_repair_baseline_v1/connected_components_object_extraction_capability_pilot_v1/deconfounded_count_curriculum_causal_pilot_v1/e4e_late_count_v1';binding=pkg/'CPU_TEST_BINDING.json';receipt=pkg/'CPU_PRODUCTION_PATH_RECEIPT.json'
  subprocess.run([sys.executable,str(ROOT/'scripts/freeze_e4e_late_layer_count_v1.py')],env=env,check=True,capture_output=True,text=True)
  subprocess.run([sys.executable,str(ROOT/'scripts/run_e4e_late_layer_count_v1.py'),'--make-cpu-binding',str(binding)],env=env,check=True,capture_output=True,text=True)
  subprocess.run([sys.executable,str(ROOT/'scripts/run_e4e_late_layer_count_v1.py'),'--binding',str(binding),'--cpu-production-path','--receipt',str(receipt)],env=env,check=True,capture_output=True,text=True)
  got=json.loads(receipt.read_text());self.assertEqual('PASS_E4E_LAUNCHER_WORKER_RAW_FREEZE_SIDECAR_SCORER_CPU_ONLY',got['status']);self.assertFalse(got['model_loaded']);self.assertFalse(got['gpu_used']);self.assertTrue(got['sidecar_opened_after_freeze'])

if __name__=='__main__':unittest.main()
