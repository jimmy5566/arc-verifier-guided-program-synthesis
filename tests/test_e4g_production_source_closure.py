from __future__ import annotations
import hashlib, json, subprocess, sys, tempfile, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / 'experiments/capability_repair_baseline_v1/connected_components_object_extraction_capability_pilot_v1/deconfounded_count_curriculum_causal_pilot_v1/e4g_inside_contains_atomic_plasticity_v1'
WORKER = ROOT / 'scripts/run_e4g_inside_contains_production_v1.py'
SCORER = ROOT / 'scripts/score_e4g_inside_contains_v1.py'
LAUNCHER = ROOT / 'scripts/launch_e4g_inside_contains_v1.sh'

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def binding(path):
    names = {
        'freeze':'CPU_FREEZE_RECEIPT.json',
        'ood':'STRUCTURAL_OOD_INPUTS_AND_COMMITMENTS.json',
        'preflight':'CPU_PRODUCTION_PATH_RECEIPT.json',
        'schedule':'TRAINING_SCHEDULE.json',
        'protection':'PROTECTION_PANEL_REFERENCES.json',
        'trainfit':'TRAIN_FIT_INPUTS_AND_COMMITMENTS.json',
        'scoring':'SCORING_CONTRACT.json',
        'protocol':'PROTOCOL.json',
    }
    inputs = {}
    for alias, name in names.items():
        p = PKG / name
        inputs[alias] = {'path': str(p.relative_to(ROOT)).replace('\\','/'), 'sha256': sha(p)}
    commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    manifest = ROOT / 'experiments/capability_repair_baseline_v1/CHECKPOINT_MANIFEST_REMOTE_V1.json'
    return {
        'experiment_id':'E4G_INSIDE_CONTAINS_ATOMIC_PLASTICITY_POSITIVE_CONTROL_V1',
        'execution_source_commit':commit,
        'inputs':inputs,
        'worker_path':str(WORKER.relative_to(ROOT)).replace('\\','/'),
        'worker_sha256':sha(WORKER),
        'output_root':'C:/tmp/e4g-test-unused',
        'runtime_python':sys.executable,
        'hard_cap_seconds':7200,
        'retry':False,
        'nonce':'cpu-fixture',
        'v7':{'manifest_sha256':sha(manifest),'adapter_model_sha256':'1639bf6665ccfbb99cf547f59fcd581663c95b6bad770ae7846fee90f66be1c3'},
        'training':{'torch_seed':20261013,'learning_rate':5e-5},
        'runtime':{'max_new_tokens':192,'context_window':16384,'physical_batch_size':1},
    }

class E4GProductionClosureTests(unittest.TestCase):
    def test_worker_preclaim_and_scorer_target_guard_without_model(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / 'binding.json'
            p.write_text(json.dumps(binding(p)),encoding='utf-8')
            got = subprocess.run([sys.executable,str(WORKER),'--binding',str(p),'--cpu-self-test'],cwd=ROOT,text=True,capture_output=True,check=True)
            self.assertIn('PASS_E4G_COMPLETE_PRODUCTION_PATH_CPU_FIXTURE',got.stdout)
            receipt = p.parent / 'E4G_CPU_COMPLETE_PATH_RECEIPT.json'
            self.assertTrue(receipt.is_file())
            data = json.loads(receipt.read_text())
            self.assertEqual(data['status'],'PASS_E4G_COMPLETE_PRODUCTION_PATH_CPU_FIXTURE')
            self.assertFalse(data['model_loaded']); self.assertFalse(data['gpu_used']); self.assertFalse(data['targets_opened'])
            self.assertEqual(data['stage_order'],['preclaim','target_blind_generation','complete_raw_freeze','sidecar','score','terminal'])
            self.assertTrue((p.parent / 'e4g_cpu_fixture_output' / 'TERMINAL_RECEIPT.json').is_file())
        got = subprocess.run([sys.executable,str(SCORER),'--cpu-self-test'],cwd=ROOT,text=True,capture_output=True,check=True)
        self.assertIn('PASS_E4G_SCORER_FAIL_CLOSED_TARGET_GUARD',got.stdout)

    def test_launcher_only_dispatches_production_entrypoint(self):
        text = LAUNCHER.read_text(encoding='utf-8')
        self.assertIn('run_e4g_inside_contains_production_v1.py',text)
        self.assertIn('--execute',text)
        self.assertIn('--cpu-self-test',text)
        subprocess.run(['bash','-n','scripts/launch_e4g_inside_contains_v1.sh'],cwd=ROOT,check=True)

if __name__ == '__main__':
    unittest.main()
