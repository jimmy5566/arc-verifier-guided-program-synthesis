import importlib.util
import json
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts/freeze_connected_components_object_extraction_execution_binding_v1.py'


def test_cpu_binding_has_one_job_frozen_batch_policy_and_no_target_access():
    spec = importlib.util.spec_from_file_location('binding', SCRIPT)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    with tempfile.TemporaryDirectory() as raw:
        out = Path(raw) / 'binding'; doc = module.freeze(out)
        assert doc['status'] == 'CPU_PREFLIGHT_READY_EXPLICIT_USER_GPU_AUTHORIZATION_REQUIRED'
        assert doc['training']['maximum_detached_jobs'] == 1 and doc['training']['retry'] is False
        assert doc['evaluation']['fallback_ladder'] == [32, 16, 8, 4, 1]
        assert doc['evaluation']['total_evaluations'] == 392
        assert doc['target_isolation']['raw_prediction_freeze_required_before_target_generation'] is True
        preflight = json.loads((out / 'CPU_PREFLIGHT_V1.json').read_text())
        assert all(preflight[k] is False for k in ('model_loaded', 'gpu_used', 'optimizer_constructed', 'targets_accessed'))
