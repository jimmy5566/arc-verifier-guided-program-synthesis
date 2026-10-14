from __future__ import annotations
import unittest

from scripts.e4e_late_layer_global_count import final_query_grid_layout, layer31_count_loss, loss_contract
from scripts.run_e4e_late_layer_count_production_v1 import audit_slots, schedule

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

if __name__ == '__main__': unittest.main()