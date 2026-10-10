import json
import shutil
import tempfile
import unittest
from pathlib import Path
from scripts import e04_orientation_marker_counterfactual_v2 as e

class E04V2CounterfactualTest(unittest.TestCase):
    def test_complete_cpu_freeze_and_regeneration(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'cohort'
            manifest=e.freeze(out)
            self.assertEqual(manifest['validation_task_rows'],768)
            self.assertEqual(manifest['views_per_canonical_base'],16)
            self.assertTrue(e.validate_frozen(out)['target_blind_prompts'])
            self.assertTrue(e.deterministic_regeneration_report(out)['all_byte_identical'])
    def test_marker_blind_guard_fails_closed(self):
        inputs,targets=e.task_rows('VALIDATION')
        group=[r for r in targets if r['condition']=='ROTATION_TARGET' and r['canonical_base_id']==targets[0]['canonical_base_id']]
        for target in group: target['target']=group[0]['target']
        with self.assertRaisesRegex(e.V2Failure,'MARKER_BLIND_PREDICTOR_NOT_EXCLUDED'):
            e.validate_tasks(inputs,targets,48)
    def test_prompt_has_no_query_target_or_condition_metadata(self):
        inputs,targets=e.task_rows('VALIDATION')
        prompt=inputs[0]['prompt']
        self.assertEqual(set(prompt),{'train','test'})
        self.assertEqual(set(prompt['test'][0]),{'input'})
        self.assertNotIn('condition',json.dumps(prompt).lower())
    def test_fixed_role_schedule_fails_closed(self):
        train=e.canonical_bases('TRAIN')
        bad=[dict(row, color_role_tuple=[1,2,3,4]) for row in train]
        with self.assertRaisesRegex(e.V2Failure,'ROLE_COLOR_IMBALANCE'):
            e.validate_bases(bad,e.canonical_bases('VALIDATION'))

    def test_condition_task_observability_fails_closed(self):
        inputs,targets=e.task_rows('VALIDATION')
        first=targets[0]['canonical_base_id']
        source=next(row['prompt']['train'] for row,target in zip(inputs,targets) if target['canonical_base_id']==first and target['condition']=='ROTATION_TARGET' and target['control_marker_turn']==0)
        for row,target in zip(inputs,targets):
            if target['canonical_base_id']==first:
                row['prompt']['train']=source
        with self.assertRaisesRegex(e.V2Failure,'CONDITION_NATIVE_TASK_UNOBSERVABLE'):
            e.validate_tasks(inputs,targets,48)
    def test_color_schedule_is_role_balanced(self):
        e.validate_bases(e.canonical_bases('TRAIN'),e.canonical_bases('VALIDATION'))

if __name__ == '__main__': unittest.main()
