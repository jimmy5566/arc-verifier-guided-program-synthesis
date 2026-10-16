import json
import subprocess
import sys

import pytest

from scripts import run_cpu_selector_completion as wrapper


@pytest.mark.parametrize('module, artifact', [
    ('scripts.score_reused_rankings_cpu', 'CPU_REUSED_SCORECARD.json'),
    ('scripts.analyze_frozen_eval60_search_cpu', 'RESULT.json'),
])
def test_production_command_and_terminal_artifact(monkeypatch, tmp_path, module, artifact):
    out = tmp_path/'run'
    monkeypatch.setattr(sys, 'argv', ['wrapper', '--output', str(out), '--worker-module', module, '--', '--base', 'frozen'])
    monkeypatch.setattr(subprocess, 'check_output', lambda *a, **kw: 'a'*40)
    class Child:
        pid = 123
        def __init__(self, command, **kw):
            assert command == [sys.executable, '-m', module, '--output', str(out/'rankings'), '--base', 'frozen']
            (out/'rankings').mkdir()
            (out/'rankings'/artifact).write_text('{}')
        def wait(self, timeout):
            assert timeout == 1800
            return 0
    monkeypatch.setattr(subprocess, 'Popen', Child)
    assert wrapper.main() == 0
    state = json.loads((out/'PIPELINE_STATE.json').read_text())
    assert state['status'] == 'COMPLETE'
    assert state['gpu_used'] is False and state['model_loaded'] is False
    assert state['scorecard'].endswith(artifact)


def test_failure_receipt_and_wall_bound(monkeypatch, tmp_path):
    out = tmp_path/'failed'
    monkeypatch.setattr(sys, 'argv', ['wrapper', '--output', str(out)])
    monkeypatch.setattr(subprocess, 'check_output', lambda *a, **kw: 'a'*40)
    class Child:
        pid = 456
        def __init__(self, *a, **kw): pass
        def wait(self, timeout): return 7
    monkeypatch.setattr(subprocess, 'Popen', Child)
    assert wrapper.main() == 1
    assert json.loads((out/'PIPELINE_STATE.json').read_text())['reason'].endswith('CPU_SCORER_EXIT_7')
    monkeypatch.setattr(sys, 'argv', ['wrapper', '--output', str(tmp_path/'bad'), '--max-wall-seconds', '1801'])
    with pytest.raises(RuntimeError, match='1800'):
        wrapper.main()
