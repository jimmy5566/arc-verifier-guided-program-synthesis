import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

from scripts.decision_token_supervision import count_terminal_label_index, matched_weights
from training_data.pipeline import task_to_sample


class DecisionTokenSupervisionTests(unittest.TestCase):
    def test_real_diverse_count_slots_are_supervised_and_mass_matched(self):
        schedule = json.loads((ROOT / 'experiments/capability_repair_baseline_v1/connected_components_object_extraction_capability_pilot_v1/TREATMENT_SCHEDULE.json').read_text())['episodes']
        count_slots = [row for row in schedule if row['role'] == 'PRIMARY_CONNECTED_COMPONENT_COUNT']
        self.assertEqual(len(schedule), 384)
        self.assertEqual(len(count_slots), 48)
        for row in count_slots:
            sample = task_to_sample(row['task'])
            index, count = count_terminal_label_index(row['task'])
            control, treatment = matched_weights(sample['labels'], index, multiplier=2.0)
            self.assertEqual(sample['labels'][index], row['task']['test'][-1]['output'][1][count - 1])
            self.assertEqual(set(control), set(treatment))
            self.assertAlmostEqual(sum(control.values()), sum(treatment.values()), places=12)
            self.assertEqual(treatment[index], 2.0)
            self.assertTrue(all(weight > 0 for weight in treatment.values()))

    def test_equal_weights_match_uniform_causal_loss_on_cpu(self):
        try:
            import torch
        except ImportError:
            self.skipTest('torch is not installed in this CPU test environment')
        from scripts.e04_e_equal_slot_loss import causal_slot_mean_ce, backward_weighted_slot_term
        torch.manual_seed(11)
        labels = torch.tensor([[-100, 1, 2, 3]], dtype=torch.long)
        logits = torch.randn((1, 4, 16), dtype=torch.float32, requires_grad=True)
        uniform, _ = causal_slot_mean_ce(logits, labels)
        weighted_logits = logits.detach().clone().requires_grad_(True)
        weighted, _, mass = backward_weighted_slot_term(weighted_logits, labels, [0.0, 1.0, 1.0, 1.0], expected_slots=1)
        self.assertAlmostEqual(float(uniform.detach()), float(weighted), places=6)
        self.assertEqual(mass, 3.0)
        self.assertTrue(torch.isfinite(weighted_logits.grad).all())

    def test_non_count_output_is_rejected(self):
        task = {'train': [{'input': [[0]], 'output': [[0]]}], 'test': [{'input': [[0]], 'output': [[0]]}]}
        with self.assertRaisesRegex(ValueError, 'COUNT_OUTPUT_SHAPE'):
            count_terminal_label_index(task)


if __name__ == '__main__':
    unittest.main()
