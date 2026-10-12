import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts/amend_connected_components_object_extraction_capability_pilot_v1_measurement_contract.py'


def test_additive_measurement_contract_preserves_schedules_and_targets():
    spec = importlib.util.spec_from_file_location('amend', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()
    base = ROOT / 'experiments/capability_repair_baseline_v1/connected_components_object_extraction_capability_pilot_v1'
    m = json.loads((base / 'measurement_contract_v1/AMENDED_MANIFEST.json').read_text())
    assert m['status'] == 'PASS_CPU_ONLY_ADDITIVE_MEASUREMENT_CONTRACT'
    assert m['model_loaded'] is False and m['gpu_used'] is False and m['targets_accessed'] is False
    assert m['immutable_schedule_sha256']['control'] == module.fsha(base / 'CONTROL_SCHEDULE.json')
    assert m['immutable_schedule_sha256']['treatment'] == module.fsha(base / 'TREATMENT_SCHEDULE.json')
    train = json.loads((base / 'measurement_contract_v1/TRAIN_FIT_INPUT_PROMPTS.json').read_text())
    assert len(train['records']) == 48
    assert all('output' not in row['task']['test'][0] for row in train['records'])
    protected = json.loads((base / 'measurement_contract_v1/PROTECTED_CAPABILITY_INPUT_PROMPTS.json').read_text())
    assert len(protected['records']) == 16
    assert protected['source_protocol'] == 'E04_ORIENTATION_V3_FIXED_INDEPENDENT_DEMONSTRATION_BASELINE'
    assert protected['independent_canonical_bases_per_condition'] == 4
    assert all('output' not in row['prompt']['test'][0] for row in protected['records'])
    assert all(set(row['prompt']) == {'train', 'test'} for row in protected['records'])
