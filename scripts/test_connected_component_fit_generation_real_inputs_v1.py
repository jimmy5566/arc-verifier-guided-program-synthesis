"""CPU-only regression for the target-free real-input execution material."""
import importlib.util
import json
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('component_inputs', ROOT / 'scripts' / 'prepare_connected_component_fit_generation_inputs_v1.py')
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def main():
    cohorts, missing = module.build()
    assert len(cohorts) == 72 and len(missing) == 48
    assert len({(row['checkpoint'], row['cohort_record_id']) for row in missing}) == 48
    assert all('output' not in case for row in cohorts.values() for case in row['task']['test'])
    with tempfile.TemporaryDirectory() as raw:
        cohort_path, missing_path = Path(raw) / 'cohorts.json', Path(raw) / 'missing.json'
        result = __import__('subprocess').run([
            __import__('sys').executable, str(ROOT / 'scripts' / 'prepare_connected_component_fit_generation_inputs_v1.py'),
            '--cohort-out', str(cohort_path), '--missing-out', str(missing_path),
        ], check=True, capture_output=True, text=True)
        report = json.loads(result.stdout)
        assert report['cohort_records'] == 72 and report['missing_records'] == 48
        assert json.loads(cohort_path.read_text())['target_content_present'] is False
        assert json.loads(missing_path.read_text())['target_content_present'] is False
    print('PASS_REAL_INPUTS_V1')


if __name__ == '__main__':
    main()
