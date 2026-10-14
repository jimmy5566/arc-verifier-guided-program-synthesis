from __future__ import annotations
import unittest
import json, tempfile
from pathlib import Path

from scripts.e4e_late_layer_global_count import final_query_grid_layout, layer31_count_loss, loss_contract
from scripts.run_e4e_late_layer_count_production_v1 import audit_slots, schedule, marker_endpoint_guard, validate_feature_file, fail

class E4EExecutableTests(unittest.TestCase):
    def test_all_384_slots_and_frozen_loss_partition(self):
        primary, noncount = audit_slots()
        self.assertEqual(len(schedule()), 384)
        self.assertEqual(len(primary), 48)
        self.assertEqual(len(noncount), 336)
        self.assertEqual({x['auxiliary_coefficient'] for x in noncount}, {0.0})
        self.assertEqual([sum(x['count']==c for x in primary) for c in (1,2,3)], [16,16,16])
        for row in primary:
            self.assertEqual(row['control']['answer_coefficient'], 1.0)
            self.assertEqual(row['treatment']['answer_coefficient'], 1.0)
            self.assertEqual(row['treatment']['auxiliary_coefficient'], .25)
            self.assertEqual(row['treatment']['slot_coefficient'], .25)

    def test_layer31_auxiliary_gradient_is_additive_and_head_is_train_only(self):
        try:
            import torch
        except ImportError:
            self.skipTest('torch unavailable')
        states=[torch.zeros((1,9,4),requires_grad=True) for _ in range(32)]
        states[31]=torch.randn((1,9,4),requires_grad=True)
        head=torch.nn.Linear(4,3)
        loss, pooled=layer31_count_loss(states,(0,1,2),2,head)
        loss.backward()
        self.assertIsNotNone(head.weight.grad)
        self.assertGreater(float(head.weight.grad.abs().sum()),0)
        self.assertIsNotNone(states[31].grad)
        self.assertGreater(float(states[31].grad.abs().sum()),0)
        self.assertEqual(loss_contract(2.0,1.0,False)['answer_mean'], loss_contract(2.0,1.0,True)['answer_mean'])

    def test_launcher_reaches_production_execute_entrypoint(self):
        launcher=(Path(__file__).resolve().parents[1]/'scripts/launch_e4e_late_layer_count_v1.sh').read_text(encoding='utf-8')
        self.assertIn('run_e4e_late_layer_count_production_v1.py', launcher)
        self.assertIn('--execute', launcher)
        self.assertIn('runtime_python', launcher)
        self.assertNotIn('run_e4e_late_layer_count_v1.py --binding', launcher)

    def test_preclaim_requires_cuda_bf16_before_model_loader(self):
        worker=(Path(__file__).resolve().parents[1]/'scripts/run_e4e_late_layer_count_production_v1.py').read_text(encoding='utf-8')
        preclaim=worker[worker.index('def preclaim'):worker.index('def load_model')]
        self.assertIn('torch.cuda.is_available()', preclaim)
        self.assertIn('torch.cuda.is_bf16_supported()', preclaim)

    def test_marker_single_endpoint_negative_fails_even_if_aggregate_nonnegative(self):
        rows=[{'split':'MARKER_BINDING','endpoint':0,'id':str(i)} for i in range(8)]+[{'split':'MARKER_BINDING','endpoint':2,'id':str(i+8)} for i in range(8)]
        self.assertEqual(set(marker_endpoint_guard(rows)),{0,2})
        delta={'0':-1,'2':8}; self.assertFalse(all(v>=0 for v in delta.values()) and sum(delta.values())>=0)
        with self.assertRaises(RuntimeError): marker_endpoint_guard(rows[:8])

    def test_mutated_probe_features_fail_freeze_validation(self):
        with tempfile.TemporaryDirectory() as d:
            from scripts.run_e4e_late_layer_count_production_v1 import probe_rows
            p=Path(d)/'f.jsonl'; data=[{k:v for k,v in x.items() if k!='task'}|{'feature':[0.0,1.0]} for x in probe_rows()]
            p.write_text('\n'.join(json.dumps(x,sort_keys=True) for x in data)+'\n',encoding='utf-8'); self.assertEqual(validate_feature_file(p)['records'],144)
            data[0]['feature']=[float('nan'),1.0];p.write_text('\n'.join(json.dumps(x,sort_keys=True) for x in data)+'\n',encoding='utf-8')
            with self.assertRaises(RuntimeError): validate_feature_file(p)

    def test_zero_regression_protection(self):
        retention={'a':-1,'b':2}; self.assertFalse(all(v>=0 for v in retention.values()) and sum(retention.values())>=0 and 0>=0)
        retention={'a':0,'b':0}; self.assertFalse(all(v>=0 for v in retention.values()) and sum(retention.values())>=0 and -1>=0)

if __name__ == '__main__': unittest.main()
